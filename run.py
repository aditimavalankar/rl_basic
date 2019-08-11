import gym
import torch
import argparse
from utils.helper import set_global_seed
from algorithms import vanilla_pg, a2c_mc, a2c_critic, a2c_gae, ddpg


parser = argparse.ArgumentParser(description='RL-algorithms')
parser.add_argument('--alg', default='vanilla_pg',
                    help='Training algorithm: Choose from vanilla_pg,'
                    'a2c_mc, a2c_gae, a2c_critic, ddpg')
parser.add_argument('-n', '--n-training-steps', default=1000000, type=int,
                    help='number of training steps')
parser.add_argument('-alr', '--actor-lr', default=1e-4, type=float,
                    help='learning rate of the actor')
parser.add_argument('-clr', '--critic-lr', default=1e-3, type=float,
                    help='learning rate of the critic')
parser.add_argument('--gamma', default=0.99, type=float,
                    help='discount factor')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--resume', default='', help='previous checkpoint from'
                    ' which training is to be resumed')
parser.add_argument('-env', '--env-name', default='Walker2d-v2',
                    help='OpenAI Mujoco environment name')
parser.add_argument('-s', '--seed', default=4, type=int, help='seed')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='Batch size')
parser.add_argument('--horizon', default=2048, type=int, help='Horizon')
parser.add_argument('--gae', default=0.95, type=float, help='GAE Lambda')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--entropy-coeff', default=0.01, type=float,
                    help='Entropy coefficient')
parser.add_argument('--use-lr-decay', action='store_true')
parser.add_argument('-tn', '--n-test-episodes', default=10, type=int,
                    help='number of test episodes')
parser.add_argument('--tau', default=1e-3, type=float,
                    help='Target network soft update parameter')
parser.add_argument('--l2-reg', default=1e-2, type=float,
                    help='L2 regularization for weights of non-final layers')
parser.add_argument('--hidden-layer-size', default=128, type=int,
                    help='Hidden layer dimension')
parser.add_argument('--updates-per-step', default=5, type=int,
                    help='Number of updates per step')
parser.add_argument('--noise_scale', type=float, default=0.3,
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.3,
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=100,
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--log-reward-buffer-size', type=int, default=10,
                    help='Size of the reward buffer for logging')


def main():
    args = parser.parse_args()
    alg = args.alg

    env = gym.make(args.env_name)
    env.seed(args.seed)
    set_global_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = vars(args)
    args['env'] = env
    args['device'] = device

    if alg == 'vanilla_pg':
        vanilla_pg.runner.run(args)

    elif alg == 'a2c_mc':
        a2c_mc.runner.run(args)

    elif alg == 'a2c_critic':
        a2c_critic.runner.run(args)

    elif alg == 'a2c_gae':
        a2c_gae.runner.run(args)

    elif alg == 'ddpg':
        ddpg.runner.run(args)

    else:
        print('Enter valid algorithm.')
        return


if __name__ == '__main__':
    main()
