"""Generate learning curves and calibration plots from seed-level data."""
import argparse, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"Writing figures to {args.out} (stub).")


if __name__ == "__main__":
    main()