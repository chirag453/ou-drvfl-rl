"""Evaluate a trained OU-dRVFL-RL run."""
import argparse
import numpy as np
from src.envs import make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--episodes", type=int, default=100)
    args = ap.parse_args()
    # Placeholder: load checkpoints from `args.run`, run evaluation episodes.
    print(f"Evaluating {args.run} for {args.episodes} episodes (stub).")


if __name__ == "__main__":
    main()