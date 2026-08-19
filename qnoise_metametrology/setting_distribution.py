#!/usr/bin/env python
"""Per-setting Fisher-information distribution vs the eta_needle proxy.

Referee request (Nature-style report, item 5): eta_needle = s_max^2/F_Q^A
is a proxy; the achievable classical Fisher information of the best Pauli
setting also carries the binomial variance factor 1 - <P>^2.  This script
regenerates the n=10 oneshot circuits (identical seeds as
lean_oneshot_n10.json) and records, per (a, t, seed):

  * the full normalized per-setting sensitivity distribution s_P^2 /
    sum s^2: top-k cumulative shares (k = 1, 4, 16, 64) and m_eff;
  * the achievable classical FI per setting F_P = s_P^2 / (1 - <P>^2)
    at theta = 0, its maximum over settings, and
    eta_ach = max_P F_P / F_Q^A alongside the proxy eta_needle;
  * the variance factor 1 - <P>^2 at the argmax setting;
  * cross-check: recomputed eta_needle must match the stored dataset.

Deterministic.  Output: data/setting_distribution_n10.json.
"""

import json
import time
from pathlib import Path

import numpy as np

from prototype_phase_diagram import pauli_coeffs, reduced_density
from prototype_lean import synth_blocks, make_phi

DATA = Path(__file__).parent.parent / "data"

N = 10
A_LIST = [6, 8]
TS = [0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
SEEDS = range(8)
TOPK = [1, 4, 16, 64]


def qfi_from_states(rho, drho):
    w, v = np.linalg.eigh(rho)
    d = v.conj().T @ drho @ v
    fq = 0.0
    for i in range(len(w)):
        for j in range(len(w)):
            den = w[i] + w[j]
            if den > 1e-12:
                fq += 2.0 * abs(d[i, j]) ** 2 / den
    return float(fq)


def cell(n, t, a, seed):
    rng = np.random.default_rng([seed, t, a, 0])
    blocks = synth_blocks(n, t, max(t, 1), rng)
    phi, _ = make_phi(blocks, "oneshot", n)

    dth = 1e-3
    e = {}
    for th in (dth, -dth, 0.0):
        rho = reduced_density(phi(th), n, a)
        e[th] = pauli_coeffs(rho, a).reshape(-1).real * 2 ** a
    s = (e[dth][1:] - e[-dth][1:]) / (2 * dth)
    s2 = s ** 2
    tot = float(s2.sum())

    delta = 1e-4
    rho0 = reduced_density(phi(0.0), n, a)
    drho = (reduced_density(phi(delta), n, a)
            - reduced_density(phi(-delta), n, a)) / (2 * delta)
    fqa = qfi_from_states(rho0, drho)

    var = np.clip(1.0 - e[0.0][1:] ** 2, 1e-12, 1.0)
    fi = s2 / var
    kmax = int(np.argmax(fi))
    order = np.argsort(s2)[::-1]
    shares = {k: float(s2[order[:k]].sum() / max(tot, 1e-300)) for k in TOPK}

    eta_needle = float(s2.max() / fqa) if fqa > 1e-9 else 0.0
    eta_ach = float(fi[kmax] / fqa) if fqa > 1e-9 else 0.0
    return dict(n=n, t=t, a=a, seed=seed, fqa=fqa,
                s_max_sq=float(s2.max()), s_sum_sq=tot,
                m_eff=float(tot ** 2 / max(np.sum(s2 ** 2), 1e-300)),
                eta_needle=eta_needle, eta_ach=eta_ach,
                fi_max=float(fi[kmax]),
                var_at_best=float(var[kmax]),
                var_at_smax=float(var[int(np.argmax(s2))]),
                topk_shares={str(k): v for k, v in shares.items()})


def main():
    t0 = time.time()
    stored = json.load(open(DATA / "lean_oneshot_n10.json"))
    ref = {(r["t"], r["a"], r["seed"]): r["eta_needle"] for r in stored}

    recs, mismatches = [], 0
    for a in A_LIST:
        for t in TS:
            for seed in SEEDS:
                r = cell(N, t, a, seed)
                key = (t, a, seed)
                if key in ref and ref[key] > 1e-9:
                    if abs(r["eta_needle"] - ref[key]) / ref[key] > 1e-6:
                        mismatches += 1
                        print(f"MISMATCH {key}: {r['eta_needle']:.6g} "
                              f"vs stored {ref[key]:.6g}")
                recs.append(r)
        done = [r for r in recs if r["a"] == a]
        print(f"a={a}: {len(done)} cells ({time.time()-t0:.0f}s)")

    print(f"\ncross-check vs lean_oneshot_n10.json: {mismatches} mismatches")
    print(f"\n{'a':>2} {'t':>3} {'eta_needle':>11} {'eta_ach':>9} "
          f"{'ratio':>6} {'top1':>6} {'top16':>6} {'top64':>6} {'m_eff':>8}")
    summary = []
    for a in A_LIST:
        for t in TS:
            sel = [r for r in recs if r["a"] == a and r["t"] == t]
            en = np.mean([r["eta_needle"] for r in sel])
            ea = np.mean([r["eta_ach"] for r in sel])
            row = dict(
                a=a, t=t,
                eta_needle=float(en), eta_ach=float(ea),
                ach_over_needle=float(ea / en) if en > 1e-12 else None,
                top1=float(np.mean([r["topk_shares"]["1"] for r in sel])),
                top4=float(np.mean([r["topk_shares"]["4"] for r in sel])),
                top16=float(np.mean([r["topk_shares"]["16"] for r in sel])),
                top64=float(np.mean([r["topk_shares"]["64"] for r in sel])),
                m_eff=float(np.mean([r["m_eff"] for r in sel])),
                var_at_best=float(np.mean([r["var_at_best"] for r in sel])))
            summary.append(row)
            print(f"{a:>2} {t:>3} {en:>11.4g} {ea:>9.4g} "
                  f"{(ea/en if en>1e-12 else float('nan')):>6.2f} "
                  f"{row['top1']:>6.3f} {row['top16']:>6.3f} "
                  f"{row['top64']:>6.3f} {row['m_eff']:>8.1f}")

    out = dict(n=N, ts=TS, a_list=A_LIST, n_seeds=len(list(SEEDS)),
               cross_check_mismatches=mismatches,
               records=recs, summary=summary)
    with open(DATA / "setting_distribution_n10.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", DATA / "setting_distribution_n10.json",
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
