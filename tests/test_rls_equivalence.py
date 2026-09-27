"""Numerically verify Theorem 1: RLS recursion == batch ridge solution."""
import numpy as np
from src.rls_critic import RLSCritic


def test_rls_equals_batch():
    rng = np.random.default_rng(0)
    D, T, lam, delta = 20, 200, 1.0, 1e-3
    X = rng.normal(size=(T, D))
    y = rng.normal(size=T)

    # RLS
    c = RLSCritic(D, ridge=delta, forgetting=lam, covariance="full")
    for t in range(T):
        c.update(X[t], y[t])

    # Batch ridge
    A = X.T @ X + delta * np.eye(D)
    theta_batch = np.linalg.solve(A, X.T @ y)

    assert np.allclose(c.theta, theta_batch, atol=1e-6)