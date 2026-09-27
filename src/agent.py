"""Top-level OU-dRVFL-RL agent: RLS critic ensemble + optional gradient actor."""
import numpy as np
from .features import DeepRVFLFeatures
from .ensemble import RLSCriticEnsemble


class OUDRVFLAgent:
    def __init__(self, obs_dim, n_actions, cfg, seed=0):
        self.features = DeepRVFLFeatures(
            input_dim=obs_dim,
            depth=cfg["feature"]["depth"],
            width=cfg["feature"]["width"],
            seed=seed,
            activation=cfg["feature"].get("activation", "relu"),
        )
        D = self.features.output_dim
        self.ensemble = RLSCriticEnsemble(
            dim=D,
            K=cfg["critic"]["ensemble_size"],
            ridge=cfg["critic"]["ridge"],
            forgetting=cfg["critic"]["forgetting"],
            covariance=cfg["critic"]["covariance"],
            prior_scale=cfg["critic"]["prior_scale"],
            bootstrap_prob=cfg["critic"]["bootstrap_prob"],
            seed=seed,
        )
        self.n_actions = n_actions
        self.gamma = cfg["training"].get("gamma", 0.99)

    def act(self, obs, mode="ucb", beta=0.1):
        phi = self.features(obs)
        qbars, us = [], []
        for a in range(self.n_actions):
            x = np.concatenate([phi, np.eye(self.n_actions)[a]])
            qbar, u, _ = self.ensemble.mean_and_disagreement(x)
            qbars.append(qbar)
            us.append(u)
        qbars = np.array(qbars)
        us = np.array(us)
        if mode == "ucb":
            return int(np.argmax(qbars + beta * us))
        return int(np.argmax(qbars))

    def update(self, obs, action, reward, next_obs, done):
        phi = self.features(obs)
        phi_next = self.features(next_obs)
        x = np.concatenate([phi, np.eye(self.n_actions)[action]])
        targets = []
        for k in range(self.ensemble.K):
            q_next = max(
                self.ensemble.predict_all(np.concatenate([phi_next, np.eye(self.n_actions)[a2]]))[k]
                for a2 in range(self.n_actions)
            )
            targets.append(reward + self.gamma * (1.0 - done) * q_next)
        self.ensemble.update_all(x, targets)