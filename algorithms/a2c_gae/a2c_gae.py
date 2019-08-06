import gym
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.optim import Adam
from torch.distributions.normal import Normal
import os
from network import Network
from utils import load_checkpoint, save_checkpoint, set_global_seed
from runningmeanstd import RunningMeanStd
from tensorboardX import SummaryWriter


MEAN_STD = 0
MIN_MAX = 1


parser = argparse.ArgumentParser(description='A2C')
parser.add_argument('-n', '--train-steps', default=1000000, type=int,
                    help='number of training steps')
parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float,
                    help='learning rate of the network')
parser.add_argument('-gamma', '--discount-factor', default=0.99, type=float,
                    help='discount factor')
parser.add_argument('-tn', '--n-test-episodes', default=10, type=int,
                    help='number of test episodes')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--resume', default='', help='previous checkpoint from'
                    ' which training is to be resumed')
parser.add_argument('-env', '--environment', default='Ant-v2',
                    help='OpenAI Mujoco environment name')
parser.add_argument('-s', '--seed', default=0, type=int, help='seed')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    help='Batch size')
parser.add_argument('--horizon', default=500, type=int, help='Horizon')
parser.add_argument('--gae', default=0.95, type=float, help='GAE Lambda')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--entropy-coeff', default=0.01, type=float,
                    help='Entropy coefficient')
parser.add_argument('--use-lr-decay', action='store_true')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mseloss = nn.MSELoss()


def batch_actor_critic(logps, rewards, values, dones, gamma, lam, horizon,
                       adv_rms, return_rms):
    running_adv = 0
    n = len(rewards)
    advs = np.zeros(n)
    returns = np.zeros(n)
    last_index = n - 1

    for i in range(n - 1, -1, -1):

        if dones[i] or i == n - 1:
            running_adv = 0
            last_index = i
            delta = rewards[i] - values[i]
            returns[i] = rewards[i]
        else:
            delta = rewards[i] + gamma * values[i + 1] - values[i]
            returns[i] = rewards[i] + gamma * values[i + 1]

        # returns[i] = return_rms.normalize(np.array([returns[i]]))[0]
        running_adv = delta + gamma * lam * running_adv
        if i + horizon < last_index:
            running_adv -= (((gamma * lam) ** (horizon + 1)) *
                            advs[i + horizon + 1])
        advs[i] = running_adv
        # advs[i] = adv_rms.normalize(np.array([advs[i]]))[0]

    # returns = (returns - returns.mean()) / (returns.std() + 1e-6)

    value_loss = 0
    policy_loss = 0

    for (v, r) in zip(values, returns):
        value_loss += (v - r) ** 2

    for logp, adv in zip(logps, advs):
        policy_loss += (-logp * adv).sum()

    policy_loss /= len(logps)
    value_loss /= len(logps)

    return policy_loss, value_loss


def prepare_input(x):
    return torch.from_numpy(x).float().to(device)


def test(env, action_dim, net, state_rms, n_test_episodes, render=True):
    average_reward = 0
    average_steps = 0
    mean_indices = torch.LongTensor([2 * x for x in range(action_dim)])
    logstd_indices = torch.LongTensor([2 * x + 1 for x in range(action_dim)])
    mean_indices = mean_indices.to(device)
    logstd_indices = logstd_indices.to(device)

    for i in range(n_test_episodes):
        done = False
        state = env.reset()
        episode_reward = 0
        steps = 0

        while not done:
            state = state_rms.normalize(state)
            out, _ = net(torch.from_numpy(state).float().to(device))
            mean = torch.index_select(out, 0, mean_indices)
            logstd = torch.index_select(out, 0, logstd_indices)
            # print(mean)
            # print(torch.exp(logstd))
            action_dist = Normal(mean, torch.exp(logstd))
            action = action_dist.sample()
            state, reward, done, info = env.step(action.cpu().numpy())
            episode_reward += reward
            steps += 1
            if render:
                env.render()

        average_reward += episode_reward
        average_steps += steps
        if render:
            print('Episode reward:', episode_reward)
            print('Episode length:', steps)

    average_reward /= n_test_episodes
    average_steps /= n_test_episodes

    print('Average reward: %f' % (average_reward))
    print('Average steps: %f' % (average_steps))
    env.close()

    return average_reward


def main():
    args = parser.parse_args()
    num_training_steps = args.train_steps
    lr = args.learning_rate
    gamma = args.discount_factor
    n_test_episodes = args.n_test_episodes
    checkpoint_file = args.resume
    test_only = args.test_only
    env_name = args.environment
    seed = args.seed
    batch_size = args.batch_size
    horizon = args.horizon
    lam = args.gae
    visualize = args.visualize
    entropy_coeff = args.entropy_coeff
    use_lr_decay = args.use_lr_decay

    env = gym.make(env_name)
    set_global_seed(seed)
    env.seed(seed)

    input_shape = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    net = Network(input_shape, action_dim).to(device)
    total_steps = 0
    total_episodes = 0

    optimizer = Adam(net.parameters(), lr=lr)
    adv_rms = RunningMeanStd(dim=1)
    return_rms = RunningMeanStd(dim=1)
    state_rms = RunningMeanStd(dim=input_shape)

    if checkpoint_file:
        (total_steps, total_episodes,
         net, optimizer, state_info,
         adv_info, return_info) = load_checkpoint(checkpoint_file, net,
                                                  optimizer, 'state', 'adv',
                                                  'return')
        state_mean, state_var, state_min, state_max = state_info
        adv_mean, adv_var, adv_min, adv_max = adv_info
        return_mean, return_var, return_min, return_max = return_info
        state_rms.set_state(state_mean, state_var, state_min, state_max,
                            total_steps)
        adv_rms.set_state(adv_mean, adv_var, adv_min, adv_max,
                          total_steps)
        return_rms.set_state(return_mean, return_var, return_min, return_max,
                             total_steps)

    checkpoint_dir = os.path.join(env_name, 'a2c_checkpoints_lr2e-3-b32-decay')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if test_only:
        avg_reward = test(env, action_dim, net, state_rms, n_test_episodes,
                          visualize)
        print('Average episode reward:', avg_reward)
        return

    # Summary writer for tensorboardX
    writer = {}
    writer['writer'] = SummaryWriter()

    s = env.reset()

    reward_buf = []
    ep_reward = 0
    ep_len = 0
    niter = 0
    done = False

    mean_indices = torch.LongTensor([2 * x for x in range(action_dim)])
    logstd_indices = torch.LongTensor([2 * x + 1 for x in range(action_dim)])
    mean_indices = mean_indices.to(device)
    logstd_indices = logstd_indices.to(device)

    prev_best = 0

    total_epochs = int(num_training_steps / batch_size) + 1

    while total_steps < num_training_steps:
        values = []
        rewards = []
        dones = []
        logps = []
        entropies = []
        niter += 1
        for _ in range(batch_size):
            s = state_rms.normalize(s, mode=MEAN_STD)
            out, v = net(prepare_input(s))
            mean = torch.index_select(out, 0, mean_indices)
            logstd = torch.index_select(out, 0, logstd_indices)
            action_dist = Normal(mean, torch.exp(logstd))
            a = action_dist.sample()
            s, r, done, _ = env.step(a.cpu().numpy())
            logp = action_dist.log_prob(a)
            entropy = action_dist.entropy()
            ep_reward += r
            ep_len += 1
            total_steps += 1

            if done:
                writer['iter'] = total_steps + 1
                writer['writer'].add_scalar('data/ep_reward',
                                            ep_reward,
                                            total_steps)
                writer['writer'].add_scalar('data/ep_len',
                                            ep_len,
                                            total_steps)
                reward_buf.append(ep_reward)
                ep_reward = 0
                ep_len = 0
                total_episodes += 1
                if len(reward_buf) > 100:
                    reward_buf = reward_buf[-100:]
                done = False
                s = env.reset()

            values.append(v)
            rewards.append(r)
            dones.append(done)
            logps.append(logp)
            entropies.append(entropy.sum())

        policy_loss, value_loss = batch_actor_critic(logps, rewards, values,
                                                     dones, gamma, lam, horizon,
                                                     adv_rms, return_rms)
        optimizer.zero_grad()
        policy_entropy = torch.stack(entropies).mean()
        loss = policy_loss + 0.5 * value_loss - entropy_coeff * policy_entropy
        loss.backward()
        optimizer.step()

        if use_lr_decay:
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                param_group['lr'] = (lr - lr * (total_steps /
                                     num_training_steps) / total_epochs)

        writer['iter'] = total_steps
        writer['writer'].add_scalar('data/last_100_ret',
                                    np.array(reward_buf).mean(),
                                    total_steps)
        writer['writer'].add_scalar('data/policy_loss',
                                    policy_loss,
                                    total_steps)
        writer['writer'].add_scalar('data/value_loss',
                                    value_loss,
                                    total_steps)
        writer['writer'].add_scalar('data/loss',
                                    loss,
                                    total_steps)

        print(total_episodes, 'episodes,',
              total_steps, 'steps,',
              np.array(reward_buf).mean(), 'reward')

        save_checkpoint(
                {'total_steps': total_steps,
                 'total_episodes': total_episodes,
                 'state_dict': net.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'state': [state_rms.mean, state_rms.var,
                           state_rms.min, state_rms.max],
                 'adv': [adv_rms.mean, adv_rms.var,
                         adv_rms.min, adv_rms.max],
                 'return': [return_rms.mean, return_rms.var,
                            return_rms.min, return_rms.max]
                 }, filename=os.path.join(checkpoint_dir, str(niter) +
                                          '.pth.tar'))

        if np.array(reward_buf).mean() > prev_best:
            save_checkpoint(
                    {'total_steps': total_steps,
                     'total_episodes': total_episodes,
                     'state_dict': net.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     }, filename=os.path.join(checkpoint_dir, 'best.pth.tar'))


if __name__ == '__main__':
    main()
