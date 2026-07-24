#!/usr/bin/env python
"""Theorem-2 systematics: fit eta_needle(t) = (1-c) exp(-alpha t) + c.

Ingests every results/lean JSON present in the working directory and
produces a fit table (alpha per n, mode, a) plus fig_theorem2_alpha.png:
  (a) eta_needle vs t, log-y, fits overlaid  (f >= 0.5 rows)
  (b) alpha vs f, grouped by (n, mode)       -> is alpha f-independent?
The Theorem-2 conjecture eta <= e^{-c t} holds if alpha is bounded away
from 0 uniformly in n (and the floor c shrinks with n: finite-size effect).
"""

import glob
import json
import re

import numpy as np
from scipy.optimize import curve_fit


def load_all():
    runs = {}
    for path in glob.glob("results_*_n*_x0.json") + glob.glob(
            "lean_*_n*.json"):
        m = re.match(r"(?:results|lean)_(\w+?)_n(\d+)", path)
        if not m:
            continue
        mode, n = m.group(1), int(m.group(2))
        runs[(n, mode)] = json.load(open(path))
    return runs


def model(t, alpha, c):
    return np.log((1 - c) * np.exp(-alpha * t) + c)


def fit_cell(ts, etas):
    """Fit on log eta; returns (alpha, alpha_err, floor)."""
    y = np.log(np.clip(etas, 1e-12, None))
    try:
        popt, pcov = curve_fit(model, ts, y, p0=[0.15, max(etas[-1], 1e-4)],
                               bounds=([0, 1e-6], [3, 1]), maxfev=20000)
        return popt[0], float(np.sqrt(pcov[0, 0])), popt[1]
    except Exception:
        return np.nan, np.nan, np.nan


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = load_all()
    if not runs:
        print("no result JSONs found")
        return

    fits = []
    for (n, mode), recs in sorted(runs.items()):
        ts = sorted({r["t"] for r in recs})
        a_vals = sorted({r["a"] for r in recs})
        for a in a_vals:
            f = a / n
            if f < 0.5 - 1e-9:
                continue  # below/at the one-shot wall; not the decay regime
            etas, sds = [], []
            for t in ts:
                v = [r["eta_needle"] for r in recs
                     if r["t"] == t and r["a"] == a]
                etas.append(np.mean(v))
                sds.append(np.std(v))
            # normalize by t=0 value so interleaved 1/L offset drops out
            e0 = max(etas[0], 1e-12)
            etas_n = np.array(etas) / e0
            alpha, err, floor = fit_cell(np.array(ts, float), etas_n)
            fits.append(dict(n=n, mode=mode, a=a, f=f, alpha=alpha,
                             alpha_err=err, floor=floor,
                             ts=ts, etas=list(etas_n), sds=list(sds)))

    print(f"{'n':>3} {'mode':>12} {'a':>3} {'f':>5} "
          f"{'alpha':>8} {'+/-':>7} {'floor':>9}")
    for r in fits:
        print(f"{r['n']:>3} {r['mode']:>12} {r['a']:>3} {r['f']:>5.2f} "
              f"{r['alpha']:>8.3f} {r['alpha_err']:>7.3f} "
              f"{r['floor']:>9.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    shown = {}
    for r in fits:
        if abs(r["f"] - 0.75) > 0.08:
            continue
        key = (r["n"], r["mode"])
        color = cmap(len(shown) % 10) if key not in shown else shown[key]
        shown[key] = color
        ts = np.array(r["ts"], float)
        ax.semilogy(ts, r["etas"], "o", ms=4, color=color,
                    label=f"n={r['n']} {r['mode']}")
        tt = np.linspace(0, ts.max(), 200)
        ax.semilogy(tt, np.exp(model(tt, r["alpha"], r["floor"])), "-",
                    color=color, alpha=0.6)
    ax.set_xlabel("t (T gates)")
    ax.set_ylabel(r"$\eta_{\rm needle}(t)\,/\,\eta_{\rm needle}(0)$")
    ax.set_title(r"$f \approx 0.75$: decay + saturation floor")
    ax.legend(fontsize=8)

    ax = axes[1]
    for (n, mode) in sorted({(r["n"], r["mode"]) for r in fits}):
        sel = [r for r in fits if r["n"] == n and r["mode"] == mode
               and np.isfinite(r["alpha"])]
        ax.errorbar([r["f"] for r in sel], [r["alpha"] for r in sel],
                    yerr=[r["alpha_err"] for r in sel], marker="o",
                    ls="--", capsize=3, label=f"n={n} {mode}")
    ax.set_xlabel("f = |A|/n")
    ax.set_ylabel(r"$\alpha$ (per T gate)")
    ax.set_title(r"Theorem-2 fit: $\eta \sim e^{-\alpha t}$")
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("fig_theorem2_alpha.png", dpi=150)
    print("figure -> fig_theorem2_alpha.png")

    with open("alpha_fits.json", "w") as fjson:
        json.dump(fits, fjson, indent=1)
    print("fits -> alpha_fits.json")


if __name__ == "__main__":
    main()
