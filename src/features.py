"""Deep random vector functional link (dRVFL) feature map with direct links."""
import numpy as np


class DeepRVFLFeatures:
    """Fixed deep random feature hierarchy with direct input-output links.

    Implements the recursion in Definition (Deep Random Feature Hierarchy):

        h^(1) = sigma(W^(1) z + b^(1))
        h^(l) = sigma(W^(l) [z; h^(l-1)] + b^(l))
        phi(z) = [1; z; h^(1); ...; h^(L)]
    """

    def __init__(self, input_dim, depth=2, width=256, seed=0, activation="relu"):
        rng = np.random.default_rng(seed)
        self.input_dim = input_dim
        self.depth = depth
        self.width = width
        self.activation = activation

        self.W, self.b = [], []
        prev = input_dim
        for _ in range(depth):
            W = rng.normal(0.0, np.sqrt(2.0 / (prev + input_dim)), size=(width, prev + input_dim))
            b = np.zeros(width)
            self.W.append(W)
            self.b.append(b)
            prev = width

    def _act(self, x):
        if self.activation == "relu":
            return np.maximum(x, 0.0)
        if self.activation == "tanh":
            return np.tanh(x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def __call__(self, z):
        z = np.asarray(z, dtype=np.float64).ravel()
        h_prev = None
        hs = []
        for l in range(self.depth):
            inp = z if h_prev is None else np.concatenate([z, h_prev])
            h = self._act(self.W[l] @ inp + self.b[l])
            hs.append(h)
            h_prev = h
        return np.concatenate([np.array([1.0]), z] + hs)

    @property
    def output_dim(self):
        return 1 + self.input_dim + self.depth * self.width