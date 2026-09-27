"""Regularized recursive least squares critic (full / block / diagonal)."""
import numpy as np


class RLSCritic:
    """One RLS head with optional block-diagonal or diagonal covariance.

    Update (Theorem 1):
        g   = P x / (lambda + x^T P x + eps)
        theta <- theta + g * clip(y - x^T theta, -c, c)
        P   <- (1/lambda) (P - g x^T P)
        P   <- (P + P^T) / 2
    """

    def __init__(self, dim, ridge=1e-3, forgetting=0.995,
                 covariance="full", block_sizes=None,
                 innovation_clip=10.0, denom_floor=1e-8):
        self.dim = dim
        self.ridge = ridge
        self.forgetting = forgetting
        self.covariance = covariance
        self.innovation_clip = innovation_clip
        self.denom_floor = denom_floor
        self.theta = np.zeros(dim)

        if covariance == "full":
            self.P = (1.0 / ridge) * np.eye(dim)
        elif covariance == "diagonal":
            self.P = (1.0 / ridge) * np.ones(dim)
        elif covariance == "block":
            assert block_sizes is not None
            self.block_sizes = block_sizes
            self.P = [(1.0 / ridge) * np.eye(b) for b in block_sizes]
        else:
            raise ValueError(covariance)

    def predict(self, x):
        return float(x @ self.theta)

    def update(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        if self.covariance == "full":
            Px = self.P @ x
            denom = self.forgetting + x @ Px + self.denom_floor
            g = Px / denom
            innov = np.clip(y - x @ self.theta, -self.innovation_clip, self.innovation_clip)
            self.theta += g * innov
            self.P = (self.P - np.outer(g, Px)) / self.forgetting
            self.P = 0.5 * (self.P + self.P.T)
        elif self.covariance == "diagonal":
            Px = self.P * x
            denom = self.forgetting + x @ Px + self.denom_floor
            g = Px / denom
            innov = np.clip(y - x @ self.theta, -self.innovation_clip, self.innovation_clip)
            self.theta += g * innov
            self.P = (self.P - g * Px) / self.forgetting
        else:  # block
            offset = 0
            g_all = np.zeros(self.dim)
            Px_all = np.zeros(self.dim)
            for i, b in enumerate(self.block_sizes):
                xb = x[offset:offset + b]
                Pb = self.P[i]
                Pxb = Pb @ xb
                denom = self.forgetting + xb @ Pxb + self.denom_floor
                gb = Pxb / denom
                g_all[offset:offset + b] = gb
                Px_all[offset:offset + b] = Pxb
                self.P[i] = (Pb - np.outer(gb, Pxb)) / self.forgetting
                offset += b
            innov = np.clip(y - x @ self.theta, -self.innovation_clip, self.innovation_clip)
            self.theta += g_all * innov