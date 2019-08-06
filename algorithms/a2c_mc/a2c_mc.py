import gym
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.optim import Adam
from torch.distributions.normal import Normal
import os
import errno
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='A2C')
parser.add_argument('-n', '--train-steps', default=1000000, type=int,
                    help='number of training steps')
parser.add_argument('-lr', '--learning-rate', default=3e-4, type=float,
                    help='learning rate of the network')
parser.add_argument('-gamma', '--discount-factor', default=0.95, type=float,
                    help='discount factor')
parser.add_argument('-tn', '--test_episodes', default=10, type=int,
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mseloss = nn.MSELoss()


class Network(nn.Module):
    """The policy and value networks share weights."""
    def __init__(self, input_shape, action_dim):
        super(Network, self).__init__()
        self.hidden1 = nn.Linear(input_shape, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.tanh = nn.Tanh()
        self.pi = nn.Linear(64, action_dim * 2)
        self.vf = nn.Linear(64, 1)

    def forward(self, x):
        out = self.hidden1(x)
        out = self.tanh(out)
        out = self.hidden2(out)
        out = self.tanh(out)
        pi = self.pi(out)
        vf = self.vf(out)
        return pi, vf


def batch_actor_critic(logps, rewards, values, dones, gamma, horizon):
    returns = []
    running_reward = 0
    n = len(rewards)
    last_index = n - 1

    for i in range(n - 1, -1, -1):
        if dones[i]:
            running_reward = 0
            last_index = i
        running_reward = rewards[i] + gamma * running_reward
        if i + horizon < last_index:
            running_reward -= ((gamma ** (horizon + 1)) *
                               rewards[i + horizon + 1])
        returns.append(running_reward)

    returns.reverse()
    returns = np.array(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)

    value_loss = 0
    policy_loss = 0
    advantages = []

    for (v, r) in zip(values, returns):
        value_loss += (v - r) ** 2
        advantages.append(r - v)

    for logp, adv in zip(logps, advantages):
        policy_loss += (-logp * adv).sum()

    policy_loss /= len(logps)
    value_loss /= len(logps)

    return policy_loss, value_loss


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def load_checkpoint(filename, net, optimizer):
    if not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT,
                                os.strerror(errno.ENOENT),
                                filename)
    print('Loading checkpoint ', filename)
    checkpoint = torch.load(filename)
    total_steps = checkpoint['total_steps']
    total_episodes = checkpoint['total_episodes']
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return total_steps, total_episodes, net, optimizer


def prepare_input(x):
    return torch.from_numpy(x).float().to(device)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def main():
    args = parser.parse_args()
    num_training_steps = args.train_steps
    lr = args.learning_rate
    gamma = args.discount_factor
    test_episodes = args.test_episodes
    checkpoint_file = args.resume
    test_only = args.test_only
    env_name = args.environment
    seed = args.seed
    batch_size = args.batch_size
    horizon = args.horizon

    env = gym.make(env_name)
    set_global_seed(seed)
    env.seed(seed)

    input_shape = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    net = Network(input_shape, action_dim).to(device)
    total_steps = 0
    total_episodes = 0

    # Summary writer for tensorboardX
    writer = {}
    writer['writer'] = SummaryWriter()

    optimizer = Adam(net.parameters(), lr=lr)

    if checkpoint_file:
        total_steps, total_episodes, net, optimizer = load_checkpoint(
            checkpoint_file, net, optimizer)

    checkpoint_dir = os.path.join(env_name, 'a2c_checkpoints')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    s = env.reset()

    reward_buf = []
    ep_reward = 0
    ep_len = 0
    niter = 0
    done = False

    mean_indices = torch.LongTensor([2 * x for x in range(action_dim)]).to(device)
    logstd_indices = torch.LongTensor([2 * x + 1 for x in range(action_dim)]).to(device)


    while total_steps < num_training_steps:
        values = []
        rewards = []
        dones = []
        logps = []
        niter += 1
        for _ in range(batch_size):
            out, v = net(prepare_input(s))
            mean = torch.index_select(out, 0, mean_indices)
            logstd = torch.index_select(out, 0, logstd_indices)
            action_dist = Normal(mean, torch.exp(logstd))
            a = action_dist.sample()
            s, r, done, _ = env.step(a.cpu().numpy())
            logp = action_dist.log_prob(a)
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

        policy_loss, value_loss = batch_actor_critic(logps, rewards, values,
                                                     dones, gamma, horizon)
        optimizer.zero_grad()
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()

        writer['iter'] = total_steps + 1
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
                 'logstd': logstd}, filename=os.path.join(checkpoint_dir,
                                                          str(niter)
                                                          + '.pth.tar'))


if __name__ == '__main__':
    main()
