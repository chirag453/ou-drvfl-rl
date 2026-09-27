"""Ensemble of RLS critics with randomized priors and bootstrap masks."""
import numpy as np
from .rls_critic import RLSCritic


class RLSCriticEnsemble:
    def __init__(self, dim, K=5, ridge=1e-3, forgetting=0.995,
                 covariance="full", block_sizes=None, prior_scale=0.3,
                 bootstrap_prob=0.8, seed=0):
        rng = np.random.default_rng(seed)
        self.K = K
        self.bootstrap_prob = bootstrap_prob
        self.prior_scale = prior_scale
        self.critics = [
            RLSCritic(dim, ridge=ridge, forgetting=forgetting,
                      covariance=covariance, block_sizes=block_sizes)
            for _ in range(K)
        ]
        self.priors = [rng.normal(0.0, 1.0, size=dim) for _ in range(K)]
        self.rng = rng

    def predict_all(self, x):
        x = np.asarray(x)
        return np.array([
            c.predict(x) + self.prior_scale * float(x @ self.priors[k])
            for k, c in enumerate(self.critics)
        ])

    def mean_and_disagreement(self, x):
        qs = self.predict_all(x)
        qbar = qs.mean()
        u = np.sqrt(((qs - qbar) ** 2).sum() / (self.K - 1) + 1e-12)
        return qbar, u, qs

    def update_all(self, x, targets):
        for k, c in enumerate(self.critics):
            if self.rng.random() < self.bootstrap_prob:
                prior = self.prior_scale * float(np.asarray(x) @ self.priors[k])
                c.update(x, targets[k] - prior)