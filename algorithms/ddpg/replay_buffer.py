from collections import namedtuple
import random
import torch
from torch.autograd import Variable

# Taken from https://github.com/ikostrikov/pytorch-ddpg-naf

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer():
    def __init__(self, max_size=1e6):
        self.max_size = max_size
        self.buffer = []
        self.position = 0

    def update(self, *args):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.max_size

    def sample(self, size):
        batch = random.sample(self.buffer, size)
        batch = Transition(*zip(*batch))

        states = Variable(torch.cat(batch.state))
        actions = Variable(torch.cat(batch.action))
        rewards = Variable(torch.cat(batch.reward)).unsqueeze(1)
        terminals = Variable(torch.cat(batch.done)).unsqueeze(1)
        next_states = Variable(torch.cat(batch.next_state))

        return (states, actions, rewards, terminals, next_states)

    def __len__(self):
        return len(self.buffer)
