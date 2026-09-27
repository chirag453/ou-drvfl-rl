"""Uncertainty-aware exploration: UCB and randomized value functions."""
import numpy as np


class UCBExploration:
    def __init__(self, beta=0.1):
        self.beta = beta

    def select(self, qbar, u, n_actions):
        # qbar, u: arrays of length n_actions
        return int(np.argmax(qbar + self.beta * u))


class RandomizedValueExploration:
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def select(self, q_heads, n_actions):
        k = self.rng.integers(0, len(q_heads))
        return int(np.argmax(q_heads[k]))