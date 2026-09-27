"""Small utilities: seeding, running normalization, logging."""
import numpy as np
import random


class RunningNorm:
    def __init__(self, dim, eps=1e-8):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.n = 0
        self.eps = eps

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.var += delta * (x - self.mean)

    def __call__(self, x):
        std = np.sqrt(self.var / max(self.n, 1) + self.eps)
        return (x - self.mean) / std


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass