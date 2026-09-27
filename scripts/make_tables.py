"""Aggregate seed-level CSVs into paper tables."""
import argparse, os
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # Placeholder: iterate over CSVs in args.results and emit LaTeX/CSV tables.
    print(f"Writing tables to {args.out} (stub).")


if __name__ == "__main__":
    main()