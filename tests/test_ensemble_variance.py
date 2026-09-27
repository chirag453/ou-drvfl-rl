"""Monte-Carlo check of Theorem 4 (variance decomposition)."""
import numpy as np


def test_variance_decomposition():
    rng = np.random.default_rng(2)
    K, N = 5, 200_000
    # Positively correlated heads via a shared factor.
    rho = 0.4
    shared = rng.normal(size=N)
    Q = np.array([rho * shared + np.sqrt(1 - rho ** 2) * rng.normal(size=N)
                  for _ in range(K)])
    qbar = Q.mean(axis=0)
    lhs = qbar.var()
    within = Q.var(axis=1).mean()
    covs = [np.cov(Q[i], Q[j])[0, 1] for i in range(K) for j in range(K) if i != j]
    between = np.mean(covs)
    rhs = within / K + (1 - 1 / K) * between
    assert abs(lhs - rhs) < 5e-3