from collections import namedtuple
import random

# Taken from https://github.com/ikostrikov/pytorch-ddpg-naf

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer():
    def __init__(self, max_size=1e6):
        self.max_size = max_size
        # self.buffer = {'state': np.array([]),
        #                'action': np.array([]),
        #                'reward': np.array([]),
        #                'next_state': np.array([]),
        #                'done': np.array([])}
        self.buffer = []
        self.position = 0
        # self.size = 0

    # def insert(self, experience):
    #     if self.size == 0:
    #         self.buffer['state'] = np.array([experience[0]])
    #         self.buffer['action'] = np.array([experience[1]])
    #         self.buffer['reward'] = np.array([experience[2]])
    #         self.buffer['next_state'] = np.array([experience[3]])
    #         self.buffer['done'] = np.array([experience[4]])
    #     else:
    #         self.buffer['state'] = np.append(self.buffer['state'],
    #                                          [experience[0]], axis=0)
    #         self.buffer['action'] = np.append(self.buffer['action'],
    #                                           [experience[1]], axis=0)
    #         self.buffer['reward'] = np.append(self.buffer['reward'],
    #                                           [experience[2]], axis=0)
    #         self.buffer['next_state'] = np.append(self.buffer['next_state'],
    #                                               [experience[3]], axis=0)
    #         self.buffer['done'] = np.append(self.buffer['done'],
    #                                         [int(experience[4])], axis=0)
    #     self.size += 1

    def update(self, *args):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.max_size

    def sample(self, size):
        # indices = np.random.choice(np.arange(self.size), size=size)
        # batch = {'state': self.buffer['state'][indices],
        #          'action': self.buffer['action'][indices],
        #          'reward': self.buffer['reward'][indices],
        #          'next_state': self.buffer['next_state'][indices],
        #          'done': self.buffer['done'][indices]
        #          }
        # return batch
        return random.sample(self.buffer, size)

    def __len__(self):
        return len(self.buffer)
