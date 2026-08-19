#!/usr/bin/env python
"""Circuit-level bootstrap CIs and fit-window sensitivity for the alpha fits.

Referee request (Nature-style report, item 3): the quoted
alpha = 0.142 +/- 0.014 is the mean/SD over 12 descriptive fits, not an
ensemble confidence interval.  This script supplies both missing pieces:

  1. Nonparametric bootstrap over circuit realizations.  Within each
     (n, a) cell, the per-t circuit sets are resampled with replacement
     (circuits at different t are independent draws), the mean
     eta_needle(t)/eta_needle(0) curve is rebuilt, and the
     (1-c) e^{-alpha t} + c model is refit.  Percentile CIs [2.5, 97.5]
     are reported per cell and for the 12-fit grand mean.
  2. Fit-window sensitivity.  Each cell is refit with the maximum
     included T-count restricted to t <= {16, 24, 32, full}.

Deterministic (fixed bootstrap seed).  Output: data/alpha_robustness.json.
"""

import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

DATA = Path(__file__).parent.parent / "data"

# the 12 manuscript cells (file, [a values])
CELLS = [
    ("results_oneshot_n8_x0.json", [5, 6]),
    ("lean_oneshot_n10.json", [6, 8]),
    ("lean_oneshot_n12.json", [8, 9]),
    ("lean_oneshot_n14.json", [8, 10]),
    ("lean_oneshot_n16.json", [10, 12]),
    ("lean_oneshot_n18.json", [10, 12]),
]

N_BOOT = 2000
WINDOWS = [16, 24, 32, None]  # max t included; None = full range


def logmodel(t, alpha, c):
    return np.log((1 - c) * np.exp(-alpha * t) + c)


def fit_curve(ts, etas_n):
    y = np.log(np.clip(etas_n, 1e-12, None))
    po, _ = curve_fit(logmodel, np.array(ts, float), y, p0=[0.14, 1e-2],
                      bounds=([0, 1e-9], [3, 1]), maxfev=20000)
    return po[0]


def cell_samples(recs, a):
    """Return (ts, per-t arrays of per-circuit eta_needle)."""
    ts = sorted({r["t"] for r in recs})
    per_t = [np.array([r["eta_needle"] for r in recs
                       if r["t"] == t and r["a"] == a]) for t in ts]
    return ts, per_t


def fit_mean(ts, per_t, tmax=None):
    etas = np.array([v.mean() for v in per_t])
    e0 = max(etas[0], 1e-12)
    sel = [i for i, t in enumerate(ts) if tmax is None or t <= tmax]
    return fit_curve([ts[i] for i in sel], (etas / e0)[sel])


def main():
    t0 = time.time()
    rng = np.random.default_rng(20260819)
    out = {"n_boot": N_BOOT, "cells": [], "windows_t_max": []}

    boot_matrix = []  # cell x N_BOOT alpha samples
    for fn, alist in CELLS:
        recs = json.load(open(DATA / fn))
        n = recs[0]["n"]
        for a in alist:
            ts, per_t = cell_samples(recs, a)
            alpha_pt = fit_mean(ts, per_t)
            boots = np.empty(N_BOOT)
            for b in range(N_BOOT):
                res = [v[rng.integers(0, len(v), len(v))] for v in per_t]
                try:
                    boots[b] = fit_mean(ts, res)
                except Exception:
                    boots[b] = np.nan
            boots = boots[np.isfinite(boots)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            wins = {}
            for w in WINDOWS:
                key = str(w) if w else "full"
                try:
                    wins[key] = fit_mean(ts, per_t, tmax=w)
                except Exception:
                    wins[key] = float("nan")
            out["cells"].append(dict(
                n=n, a=a, f=a / n, n_circuits=int(len(per_t[0])),
                alpha=alpha_pt, ci_lo=float(lo), ci_hi=float(hi),
                boot_median=float(np.median(boots)),
                boot_sd=float(np.std(boots)), windows=wins))
            boot_matrix.append(boots[:min(len(boots), N_BOOT)])
            print(f"n={n:2d} a={a:2d}: alpha={alpha_pt:.4f} "
                  f"CI=[{lo:.4f},{hi:.4f}]  windows={ {k: round(v,4) for k,v in wins.items()} }")

    # grand mean: bootstrap iterations are independent across cells
    m = min(len(b) for b in boot_matrix)
    grand = np.mean([b[:m] for b in boot_matrix], axis=0)
    glo, ghi = np.percentile(grand, [2.5, 97.5])
    out["grand_mean"] = dict(
        point=float(np.mean([c["alpha"] for c in out["cells"]])),
        ci_lo=float(glo), ci_hi=float(ghi), boot_sd=float(np.std(grand)))
    for w in WINDOWS:
        key = str(w) if w else "full"
        vals = [c["windows"][key] for c in out["cells"]
                if np.isfinite(c["windows"][key])]
        out["windows_t_max"].append(dict(
            t_max=key, n_cells=len(vals), mean_alpha=float(np.mean(vals)),
            sd=float(np.std(vals))))
        print(f"window t<={key}: mean alpha = {np.mean(vals):.4f} "
              f"+/- {np.std(vals):.4f} ({len(vals)} cells)")
    print(f"grand mean {out['grand_mean']['point']:.4f} "
          f"CI=[{glo:.4f},{ghi:.4f}]  ({time.time()-t0:.0f}s)")

    with open(DATA / "alpha_robustness.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", DATA / "alpha_robustness.json")


if __name__ == "__main__":
    main()
