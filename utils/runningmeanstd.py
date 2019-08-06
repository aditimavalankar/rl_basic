import numpy as np
from utils.helper import np2tensor
import torch


MEAN_STD = 0
MIN_MAX = 1


class RunningMeanStd():
    """Computes and stores running mean and standard deviation."""
    def __init__(self, dim=1, device=None):
        super(RunningMeanStd, self).__init__()
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.std = np.ones(dim)
        self.n = 0
        self.min = 100 * np.ones(dim)
        self.max = -100 * np.ones(dim)
        self.dim = dim
        self.device = device
        if self.device is not None:
            self.mean = np2tensor(self.mean, device=self.device)
            self.var = np2tensor(self.var, device=self.device)
            self.std = np2tensor(self.std, device=self.device)
            self.min = np2tensor(self.min, device=self.device)
            self.max = np2tensor(self.max, device=self.device)

    def recompute(self, x):
        new_mean = (self.n * self.mean + x) / (self.n + 1)
        new_var = ((self.n * (self.var + self.mean ** 2) + x ** 2) /
                   (self.n + 1) - new_mean ** 2)
        self.mean = new_mean
        self.var = new_var
        if self.device is None:
            self.std = np.sqrt(self.var)
            if np.all(x > self.max):
                self.max = x
            if np.all(x < self.min):
                self.min = x
        else:
            self.std = torch.sqrt(self.var)
            if torch.all(x > self.max):
                self.max = x
            if torch.all(x < self.min):
                self.min = x
        self.n += 1

    def normalize(self, x, mode=MEAN_STD):
        self.recompute(x)
        if mode == MEAN_STD:
            return (x - self.mean) / (self.std + 1e-8)
        return (x - self.min) / (self.max - self.min + 1e-8)

    def denormalize(self, x):
        return x * (self.std + 1e-8) + self.mean

    def set_state(self, mean, var, n):
        self.mean = mean
        self.var = var
        self.std = np.sqrt(self.var)
        self.n = n

    def set_params(self, rms):
        self.mean, self.var, self.min, self.max = rms
