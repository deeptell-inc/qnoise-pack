#!/usr/bin/env python
"""Idealized branching model: E[max_b w_b] rate and multiplicity-M sweep.

Simulates the exact twirl-idealized occupancy process
  N'_k = (N_k - S_k) + 2 S_{k-1},  S_k ~ Binom(N_k, 1/2)
(each branch survives w.p. 1/2 or splits into two children with k+1),
vectorized over R realizations. max_b w_b = 2^{-k_min}.

Outputs: rate of E[max w] (windows), rate of E[sqrt(Pi)], and the
M-multiplicity sweep (max over M independent realizations) that models the
partner-slot enhancement at f > 1/2. Reproduces the panel adjudication.
"""

import argparse
import json

import numpy as np


def simulate(R, tmax, rng, kcap=None):
    kcap = kcap or (tmax + 1)
    N = np.zeros((R, kcap), dtype=np.int64)
    N[:, 0] = 1
    maxw = np.zeros((tmax, R))
    sqpi = np.zeros((tmax, R))
    w_pow = 2.0 ** (-np.arange(kcap))
    for t in range(tmax):
        S = rng.binomial(N, 0.5)
        N = N - S
        N[:, 1:] += 2 * S[:, :-1]
        occ = N > 0
        kmin = np.argmax(occ, axis=1)
        maxw[t] = 2.0 ** (-kmin.astype(float))
        pi = (N * w_pow[None, :] ** 2).sum(axis=1)
        sqpi[t] = np.sqrt(pi)
    return maxw, sqpi


def rate(series, ts, lo, hi):
    m = [(t, np.log(series[t - 1].mean())) for t in ts if lo <= t <= hi]
    tt, yy = zip(*m)
    return -np.polyfit(tt, yy, 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=200000)
    ap.add_argument("--tmax", type=int, default=32)
    ap.add_argument("--M", type=int, nargs="*", default=[1, 4, 16, 64, 256])
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    maxw, sqpi = simulate(args.R, args.tmax, rng)
    ts = list(range(1, args.tmax + 1))
    out = dict(
        R=args.R, tmax=args.tmax,
        rate_maxw_full=rate(maxw, ts, 1, args.tmax),
        rate_maxw_late=rate(maxw, ts, 8, args.tmax),
        rate_sqpi_late=rate(sqpi, ts, 8, args.tmax),
    )
    print(f"E[max w] rate: full={out['rate_maxw_full']:.4f}, "
          f"[8,{args.tmax}]={out['rate_maxw_late']:.4f}  "
          f"(beta* ln2 = {0.2271 * np.log(2):.4f})")
    print(f"E[sqrt Pi] rate [8,{args.tmax}] = {out['rate_sqpi_late']:.4f}  "
          f"(ln(4/3)/2 = {np.log(4 / 3) / 2:.4f})")

    # M-multiplicity sweep: max over M independent realizations per sample
    out["M_sweep"] = {}
    for M in args.M:
        RM = (args.R // M) * M
        mw = maxw[:, :RM].reshape(args.tmax, RM // M, M).max(axis=2)
        r = rate(mw, ts, 1, args.tmax)
        out["M_sweep"][M] = r
        print(f"M={M:>4}: E[max over M] rate (full) = {r:.4f}")
    with open("maxw_model.json", "w") as f:
        json.dump(out, f, indent=1)
    print("-> maxw_model.json")


if __name__ == "__main__":
    main()
