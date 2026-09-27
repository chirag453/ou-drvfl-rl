"""Check the finite-time growth bound P_t <= lambda^{-t} delta^{-1} I."""
import numpy as np
from src.rls_critic import RLSCritic


def test_growth_bound():
    rng = np.random.default_rng(1)
    D, lam, delta = 10, 0.99, 1e-2
    c = RLSCritic(D, ridge=delta, forgetting=lam, covariance="full")
    for t in range(50):
        c.update(rng.normal(size=D), rng.normal())
        bound = lam ** -(t + 1) * (1.0 / delta)
        assert np.linalg.eigvalsh(c.P).max() <= bound + 1e-9