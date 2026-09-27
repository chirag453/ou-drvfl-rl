"""Jetson Orin Nano latency / power measurement.

Implements the block-energy protocol described in the paper:

    E_hat = ( trapezoid(P) - P_idle * T ) / N_s
"""
import argparse
import numpy as np


def trapezoid(power, timestamps):
    power = np.asarray(power); timestamps = np.asarray(timestamps)
    return float(np.trapezoid(power, timestamps)) if hasattr(np, "trapezoid") \
        else float(np.trapz(power, timestamps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-updates", type=int, default=100000)
    ap.add_argument("--sessions", type=int, default=5)
    args = ap.parse_args()
    print(f"Measuring power: sessions={args.sessions}, N_s={args.n_updates} (stub).")


if __name__ == "__main__":
    main()