import torch
import numpy as np
import errno
import os


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


# def load_checkpoint(filename, net, optimizer, **kwargs):
def load_checkpoint(filename, **args):
    if not os.path.isfile(filename):
        raise FileNotFoundError(errno.ENOENT,
                                os.strerror(errno.ENOENT),
                                filename)
    print('Loading checkpoint ', filename)
    checkpoint = torch.load(filename)
    # total_steps = checkpoint['total_steps']
    # total_episodes = checkpoint['total_episodes']
    # net.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    params = [checkpoint[a] for a in args['params']]
    # state_mean, state_var, state_min, state_max = checkpoint['state']
    # reward_mean, reward_var, reward_min, reward_max = checkpoint['reward']
    # return (total_steps, total_episodes, net, optimizer) + other_params
    return params


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def np2tensor(x, device='cpu'):
    return torch.from_numpy(x).float().to(device)
