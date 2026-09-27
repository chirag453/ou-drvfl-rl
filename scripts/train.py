"""Train OU-dRVFL-RL on a configured environment."""
import argparse
import yaml
from src.envs import make_env
from src.agent import OUDRVFLAgent
from src.utils import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(args.seed)

    env = make_env(cfg["env"])
    obs, _ = env.reset(seed=args.seed)
    agent = OUDRVFLAgent(
        obs_dim=env.observation_space.shape[0],
        n_actions=env.action_space.n,
        cfg=cfg, seed=args.seed,
    )

    total = cfg["training"]["total_steps"]
    for step in range(total):
        a = agent.act(obs, mode=cfg["exploration"]["mode"], beta=cfg["exploration"]["beta"])
        next_obs, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        agent.update(obs, a, r, next_obs, terminated)
        obs = next_obs
        if done:
            obs, _ = env.reset()
    print("Done.")


if __name__ == "__main__":
    main()