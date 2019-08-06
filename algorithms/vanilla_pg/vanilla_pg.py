import gym
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.optim import Adam
from torch.autograd import Variable
from torch.distributions.normal import Normal
import os
import errno
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PG')
parser.add_argument('-n', '--train-steps', default=1000000, type=int,
                    help='number of training steps')
parser.add_argument('-lr', '--learning-rate', default=3e-4, type=float,
                    help='learning rate of the network')
parser.add_argument('-epi', '--episodes-per-iteration', default=1, type=int,
                    help='Number of episodes per training iteration')
parser.add_argument('-gamma', '--discount-factor', default=0.95, type=float,
                    help='discount factor')
parser.add_argument('-tn', '--test_episodes', default=10, type=int,
                    help='number of test episodes')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--resume', default='', help='previous checkpoint from'
                    ' which training is to be resumed')
parser.add_argument('-env', '--environment', default='Ant-v2',
                    help='OpenAI Mujoco environment name')
parser.add_argument('-l', '--layers', nargs='+',
                    help='hidden layer dimensions', required=True)
parser.add_argument('-s', '--seed', default=0, type=int, help='seed')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='Batch size')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):
    """Policy network takes as input the intermediate layer dimensions and
    outputs the means for each dimension of the action. In this code, the log
    of the standard deviation of the normal distribution (logstd as seen in the
    code below) is not part of the network output; it is a separate variable
    that can be trained."""
    def __init__(self, input_shape, output_shape, hidden_layers):
        super(Network, self).__init__()
        modules = []
        modules.append(nn.Linear(input_shape, hidden_layers[0]))
        modules.append(nn.Tanh())
        for i in range(len(hidden_layers) - 1):
            modules.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(hidden_layers[-1], output_shape))
        self.sequential = nn.Sequential(*modules)
        for m in self.sequential.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.5)

    def forward(self, x):
        return self.sequential(x)


def find_discounted_rewards(rewards, gamma):
    batch_discounted_rewards = []
    for batch in range(len(rewards)):
        discounted_rewards = []
        running_reward = 0
        for i in range(len(rewards[batch]) - 1, -1, -1):
            running_reward = rewards[batch][i] + gamma * running_reward
            discounted_rewards.append(running_reward)
        discounted_rewards = np.array(discounted_rewards)
        discounted_rewards = ((discounted_rewards - discounted_rewards.mean())
                              / (discounted_rewards.std() + 1e-6))
        discounted_rewards = list(reversed(discounted_rewards))
        batch_discounted_rewards.append(discounted_rewards)
    return np.array(batch_discounted_rewards).flatten()


def test(env, net, logstd, test_episodes, render=False):
    """Function to check the performance of the existing network and logstd
    on a few test trajectories."""
    average_reward = 0
    average_steps = 0
    for i in range(test_episodes):
        done = False
        state = env.reset()
        episode_reward = 0
        steps = 0
        while not done:
            means = net(torch.from_numpy(state).float().to(device))
            action_dist = Normal(means, torch.exp(logstd))
            action = action_dist.sample()
            logp = action_dist.log_prob(action)
            state, reward, done, info = env.step(action.cpu().numpy())
            episode_reward += reward
            steps += 1
            if render:
                env.render()

        average_reward += episode_reward
        average_steps += steps
        if render:
            print('Episode reward: ', episode_reward)
    average_reward /= test_episodes
    average_steps /= test_episodes
    print('Average reward: %f' % (average_reward))
    print('Average steps: %f' % (average_steps))
    env.close()
    return average_reward


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


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
    logstd = checkpoint['logstd']
    optimizer.add_param_group({"name": "logstd", "params": logstd})
    optimizer.load_state_dict(checkpoint['optimizer'])
    return total_steps, total_episodes, net, optimizer, logstd


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def main():
    args = parser.parse_args()
    num_training_steps = args.train_steps
    lr = args.learning_rate
    episodes_per_iteration = args.episodes_per_iteration
    gamma = args.discount_factor
    test_episodes = args.test_episodes
    checkpoint_file = args.resume
    test_only = args.test_only
    env_name = args.environment
    layers = args.layers
    seed = args.seed
    batch_size = args.batch_size

    env = gym.make(env_name)
    set_global_seed(seed)
    env.seed(seed)

    input_shape = env.observation_space.shape[0]
    output_shape = env.action_space.shape[0]
    hidden_layers = [int(x) for x in layers]

    net = Network(input_shape, output_shape, hidden_layers).to(device)
    total_steps = 0
    total_episodes = 0

    # Summary writer for tensorboardX
    writer = {}
    writer['writer'] = SummaryWriter()

    optimizer = Adam(net.parameters(), lr=lr)

    if checkpoint_file:
        total_steps, total_episodes, net, optimizer, logstd = load_checkpoint(
            checkpoint_file, net, optimizer)

    else:
        logstd = Variable(torch.ones(output_shape).cuda(), requires_grad=True)
        optimizer.add_param_group({"name": "logstd", "params": logstd})

    if test_only:
        test(env, net, logstd, test_episodes, render=True)
        return

    # Path to the directory where the checkpoints and log file will be saved.
    # If there is no existing directory, we create one.
    checkpoint_dir = os.path.join(env_name, 'pg_checkpoints')
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    running_reward = 0
    state = env.reset()
    done = False
    return_buf = []

    while total_steps < num_training_steps:
        batch_rewards = []
        batch_logps = []
        rewards = []
        steps = 0

        while steps < batch_size:
            # The network outputs the mean values for each action dimension
            means = net(torch.from_numpy(state).float().to(device))
            # We now have a normal distribution for each action dimension,
            # since we have the means from the network and the trainable
            # parameter, logstd.
            action_dist = Normal(means, torch.exp(logstd))
            # We sample an action from the normal distribution.
            action = action_dist.sample()
            # We compute the log-likelihood of that action.
            logp = action_dist.log_prob(action)
            state, reward, done, info = env.step(action.cpu())
            rewards.append(reward)
            batch_logps.append(logp)
            total_steps += 1
            steps += 1
            if done or steps == batch_size:
                batch_rewards.append(rewards)
                total_episodes += 1
                return_buf.append(np.array(rewards).sum())
                if len(return_buf) > 100:
                    return_buf = return_buf[-100:]
                writer['iter'] = total_steps + 1
                writer['writer'].add_scalar('data/reward',
                                            np.array(return_buf).mean(),
                                            total_steps)

            running_reward = 0.1 * np.array(rewards).sum() + 0.9 * running_reward

        batch_discounted_rewards = find_discounted_rewards(batch_rewards,
                                                           gamma)

        loss = 0
        for (r, logp) in zip(batch_discounted_rewards, batch_logps):
            # The loss for policy gradient at each time step equals the
            # product of the negative log-likelihood of the action taken
            # and the discounted reward at that step.
            loss += (-logp * r).sum()

        optimizer.zero_grad()
        loss /= batch_size
        loss.backward()
        optimizer.step()

        # average_reward = test(env, net, logstd, test_episodes)
        print(total_episodes, 'episodes,',
              total_steps, 'steps,',
              np.array(return_buf).mean(), 'reward')

        if ((total_episodes) % (episodes_per_iteration * 100) == 0 or
           total_steps >= num_training_steps):
            save_checkpoint(
                {'total_steps': total_steps,
                 'total_episodes': total_episodes,
                 'state_dict': net.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'logstd': logstd}, filename=os.path.join(checkpoint_dir,
                                                          str(total_episodes)
                                                          + '.pth.tar'))
            # print('Logstd: ', logstd)
            print('\n')

    return


if __name__ == '__main__':
    main()
