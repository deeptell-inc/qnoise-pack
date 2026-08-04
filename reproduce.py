#!/usr/bin/env python
"""qnoise-pack reproduction driver.

Fast tier (default, ~5-10 min): verification suites + refits of every
number quoted in the manuscript, checked against the quoted values.
Full tier (--full): regenerates the sweep data themselves (hours).

Exit code 0 = all checks passed.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).parent
PKG = ROOT / "qnoise_metametrology"
DATA = ROOT / "data"
PY = sys.executable

FAILURES = []


def run(cmd, cwd=PKG):
    print(f"\n$ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


def check(label, value, expected, tol):
    ok = abs(value - expected) <= tol
    print(f"  [{'OK' if ok else 'FAIL'}] {label}: {value:.4f} "
          f"(expected {expected} +/- {tol})")
    if not ok:
        FAILURES.append(label)
    return ok


def logmodel(t, alpha, c):
    return np.log((1 - c) * np.exp(-alpha * t) + c)


def fit_eta(recs, a, key="eta_needle"):
    ts = sorted({r["t"] for r in recs})
    etas = [np.mean([r[key] for r in recs if r["t"] == t and r["a"] == a])
            for t in ts]
    e0 = max(etas[0], 1e-12)
    y = np.log(np.clip(np.array(etas) / e0, 1e-12, None))
    po, _ = curve_fit(logmodel, np.array(ts, float), y, p0=[0.14, 1e-2],
                      bounds=([0, 1e-9], [3, 1]), maxfev=20000)
    return float(po[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--skip-slow", action="store_true",
                    help="skip the live verification suites (data-only refits)")
    args = ap.parse_args()

    # ---- 1. live verification suites -----------------------------------
    if not args.skip_slow:
        run([PY, "verify_proofs.py"])          # Weingarten, 120/120, rank tol
        run([PY, "branch_tree.py", "--test"])  # symplectic vs dense, n=6
        with tempfile.TemporaryDirectory() as td:
            run([PY, PKG / "maxw_model.py", "--R", "50000", "--tmax", "32"],
                cwd=td)
            mw = json.load(open(Path(td) / "maxw_model.json"))
        check("E[sqrt Pi] rate (live maxw model)",
              mw["rate_sqpi_late"], 0.1438, 0.006)
        check("E[max w] rate (live, full window)",
              mw["rate_maxw_full"], 0.182, 0.008)
        check("M=256 dragged rate", mw["M_sweep"]["256"], 0.1373, 0.01)

    # ---- 2. alpha universality (10 fits, n=8-16, f>1/2) ----------------
    quoted = {(8, 6): 0.148, (8, 5): 0.124, (10, 6): 0.158, (10, 8): 0.135,
              (12, 8): 0.146, (12, 9): 0.133, (14, 8): 0.172, (14, 10): 0.128,
              (16, 10): 0.152, (16, 12): 0.128}
    alphas = []
    for fn, alist in [("results_oneshot_n8_x0.json", [5, 6]),
                      ("lean_oneshot_n10.json", [6, 8]),
                      ("lean_oneshot_n12.json", [8, 9]),
                      ("lean_oneshot_n14.json", [8, 10]),
                      ("lean_oneshot_n16.json", [10, 12])]:
        recs = json.load(open(DATA / fn))
        n = recs[0]["n"]
        for a in alist:
            al = fit_eta(recs, a)
            alphas.append(al)
            check(f"alpha n={n} a={a}", al, quoted[(n, a)], 0.003)
    check("mean alpha", float(np.mean(alphas)), 0.142, 0.004)

    # ---- 3. purity rates (A-restricted, window t in [4,16]) ------------
    for fn, alist, exp in [("family_oneshot_n10.json", [6, 8], [.285, .282]),
                           ("family_oneshot_n12.json", [8, 9], [.285, .290])]:
        recs = json.load(open(DATA / fn))
        for a, e in zip(alist, exp):
            win = [4, 6, 8, 12, 16]
            pur = [np.mean([r["purity"] for r in recs
                            if r["t"] == t and r["a"] == a]) for t in win]
            r = -np.polyfit(win, np.log(pur), 1)[0]
            check(f"purity rate {fn} a={a}", float(r), e, 0.004)

    # ---- 4. commuting-family alphas ------------------------------------
    for fn, alist, exp in [("family_oneshot_n10.json", [6, 8], [.107, .072]),
                           ("family_oneshot_n12.json", [8, 9], [.090, .073])]:
        recs = json.load(open(DATA / fn))
        for a, e in zip(alist, exp):
            al = fit_eta(recs, a, key="eta_family")
            check(f"alpha_family {fn} a={a}", al, e, 0.004)

    # ---- 5. branch-tree operator purity, n=100 / n=1000 ----------------
    for fn, exp in [("branchtree24_n100.json", 0.2903),
                    ("branchtree_n1000.json", 0.2794)]:
        trajs = json.load(open(DATA / fn))
        tmax = max(x["t"] for x in trajs[0])
        ks = np.arange(2, tmax + 1)
        logs = [np.log(np.mean([tr[k - 1]["purity"] for tr in trajs]))
                for k in ks]
        check(f"branch purity rate {fn}",
              float(-np.polyfit(ks, logs, 1)[0]), exp, 0.003)
        cols = [x["collisions"] for tr in trajs for x in tr
                if x["collisions"] >= 0]
        check(f"branch collisions {fn}", float(sum(cols)), 0.0, 0.5)

    # ---- 6. ensemble dependence (T / T-layer / CCZ) --------------------
    recs = json.load(open(DATA / "ensemble_alpha_n8.json"))
    for kind, exp in [("t1", 0.139), ("tlayer4", 0.542), ("ccz", 0.552)]:
        evs = sorted({r["events"] for r in recs if r["kind"] == kind})
        etas = [np.mean([r["eta_needle"] for r in recs
                         if r["kind"] == kind and r["events"] == e])
                for e in evs]
        e0 = max(etas[0], 1e-12)
        y = np.log(np.clip(np.array(etas) / e0, 1e-12, None))
        po, _ = curve_fit(logmodel, np.array(evs, float), y, p0=[0.2, 1e-2],
                          bounds=([0, 1e-9], [5, 1]), maxfev=20000)
        check(f"ensemble alpha/event {kind}", float(po[0]), exp, 0.01)

    # ---- 7. identifiability / sweet spot (lean2, corrected tolerance) --
    recs = json.load(open(DATA / "lean2_oneshot_n10.json"))

    def cellmean(key, t, a, norm=False):
        v = [r[key] / (r["gram_m"] if norm else 1)
             for r in recs if r["t"] == t and r["a"] == a]
        return float(np.mean(v))

    check("rank/m a=5 t=0", cellmean("gram_rank", 0, 5, True), 0.34, 0.01)
    check("rank/m a=5 t=6", cellmean("gram_rank", 6, 5, True), 0.98, 0.01)
    check("rank/m a=3 t=0", cellmean("gram_rank", 0, 3, True), 0.03, 0.01)
    check("rank/m a=3 t=16", cellmean("gram_rank", 16, 3, True), 1.00, 0.005)
    fr = cellmean("fqa", 16, 3) / max(cellmean("fq_full", 16, 3), 1e-12)
    check("F_Q^A/F_Q a=3 t=16 (separation claim)", fr, 0.03, 0.02)
    check("eta a=5 t=6 (window)", cellmean("eta_needle", 6, 5), 0.424, 0.01)
    check("eta a=5 t=32", cellmean("eta_needle", 32, 5), 0.014, 0.005)

    # ---- 8. noise recovery (P_id / rel err spot checks) ----------------
    recs = json.load(open(DATA / "noise_recovery_n10_a5.json"))

    def cell(t, S, q):
        return [r for r in recs
                if r["t"] == t and r["S"] == S and r["q"] == q][0]

    check("P_id t=6 S=1e4 q=0", cell(6, 10000, 0.0)["p_identify"],
          0.925, 0.01)
    check("rel_err t=6 S=1e4 q=0", cell(6, 10000, 0.0)["rel_err_theta"],
          0.167, 0.02)
    check("P_id t=32 S=1e4 q=0", cell(32, 10000, 0.0)["p_identify"],
          1.00, 0.005)
    check("P_id t=0 S=1e5 q=0 (ceiling)", cell(0, 100000, 0.0)["p_identify"],
          0.350, 0.01)
    check("P_id t=0 S=1e5 q=2% (= rank/m)",
          cell(0, 100000, 0.02)["p_identify"], 0.34375, 0.005)
    check("P_id t=6 S=1e4 q=2% (robustness)",
          cell(6, 10000, 0.02)["p_identify"], 0.8875, 0.01)
    # estimation-error inflation t=32 vs t=6 (manuscript: 3.3x at S=1e4,
    # 4.1x at S=1e5; NOT "4-5x" / "an order of magnitude")
    check("rel_err ratio t32/t6 S=1e4",
          cell(32, 10000, 0.0)["rel_err_theta"]
          / cell(6, 10000, 0.0)["rel_err_theta"], 3.276, 0.05)
    check("rel_err ratio t32/t6 S=1e5",
          cell(32, 100000, 0.0)["rel_err_theta"]
          / cell(6, 100000, 0.0)["rel_err_theta"], 4.060, 0.05)

    # ---- 8b. S*_id interpolation (SM-E table, coverage gap G3) ---------
    recs = json.load(open(DATA / "noise_recovery_n10_a5.json"))
    sstar_expected = {2: 6874, 6: 2653, 12: 2581, 16: 2607, 32: 2264}
    for t, exp in sstar_expected.items():
        row = sorted([(r["S"], r["p_identify"]) for r in recs
                      if r["t"] == t and r["q"] == 0.0])
        Ss = np.array([x[0] for x in row], float)
        Ps = np.array([x[1] for x in row])
        i = int(np.argmax(Ps >= 0.5))
        s = np.interp(0.5, [Ps[i - 1], Ps[i]],
                      np.log10([Ss[i - 1], Ss[i]])) if i > 0 \
            else np.log10(Ss[0])
        check(f"S*_id t={t}", float(10 ** s), exp, exp * 0.02)

    # ---- 8c. f=0.8 noise contrast (coverage gap G5) --------------------
    recs = json.load(open(DATA / "noise_recovery_n10_a8.json"))
    v = [r for r in recs if r["t"] == 0 and r["S"] == 1000
         and r["q"] == 0.0][0]
    check("P_id f=0.8 t=0 S=1e3 (no t=0 failure at high f)",
          v["p_identify"], 0.994, 0.01)

    # ---- 8d. shipped occupancy-model rates (coverage gap G1) -----------
    mw = json.load(open(DATA / "maxw_model.json"))
    for M, exp in [("1", 0.18217), ("4", 0.17420), ("16", 0.16387),
                   ("64", 0.15152), ("256", 0.13730)]:
        check(f"shipped M-sweep rate M={M}", mw["M_sweep"][M], exp, 0.002)

    # ---- 8e. fair-budget comparison (charged calibration) --------------
    recs = json.load(open(DATA / "fair_budget_n10_a5.json"))

    def fb(t, K, S):
        return [r for r in recs if r["t"] == t and r["K"] == K
                and r["S"] == S][0]

    check("fair-budget P_id t=16 K=128", fb(16, 128, 4096)["p_identify"],
          0.4125, 0.005)
    check("fair-budget P_id t=16 K=512", fb(16, 512, 4096)["p_identify"],
          0.75, 0.005)
    check("fair-budget P_id t=16 K=1023", fb(16, 1023, 4096)["p_identify"],
          0.9812, 0.005)
    check("fair-budget P_id t=6 K=1023", fb(6, 1023, 4096)["p_identify"],
          0.8812, 0.005)
    check("fair-budget charged B at K=1023",
          float(fb(16, 1023, 4096)["budget"]), 175992832, 1)

    # ---- 9. extensive generator ----------------------------------------
    recs = json.load(open(DATA / "results_oneshot_n8_sumx.json"))
    eta0 = np.mean([r["eta_needle"] for r in recs
                    if r["t"] == 0 and r["a"] == 6])
    check("sumx eta(0) a=6 (needle multiplicity 1/n)", float(eta0),
          0.130, 0.005)
    for a, exp, tol in [(5, 0.1698, 0.003), (6, 0.0611, 0.003)]:
        al = fit_eta(recs, a)
        check(f"sumx alpha a={a} (coverage gap G4)", al, exp, tol)

    # ---- 10. interleaved eta(0) = 1/L ----------------------------------
    # per-cell deviation counts |eta(0) - 1/L| > 0.02 out of 8 seeds
    # (manuscript SM-C: identity holds cleanly only at high access a=8,
    # 22/24 seeds; lower-access cells are dominated by degenerate
    # realizations -- up to 8/8 at a=3)
    dev_expected = {
        ("leanL2_interleaved_n10.json", 6): 5,
        ("leanL2_interleaved_n10.json", 8): 1,
        ("lean_interleaved_n10.json", 3): 8,
        ("lean_interleaved_n10.json", 5): 7,
        ("lean_interleaved_n10.json", 6): 5,
        ("lean_interleaved_n10.json", 8): 1,
        ("leanL8_interleaved_n10.json", 6): 6,
        ("leanL8_interleaved_n10.json", 8): 0,
    }
    for fn, L in [("leanL2_interleaved_n10.json", 2),
                  ("lean_interleaved_n10.json", 4),
                  ("leanL8_interleaved_n10.json", 8)]:
        recs = json.load(open(DATA / fn))
        med = float(np.median([r["eta_needle"] for r in recs
                               if r["t"] == 0 and r["a"] == 8]))
        check(f"interleaved median eta(0) L={L} a=8", med, 1.0 / L, 0.02)
        avail = sorted({r["a"] for r in recs})
        for a in avail:
            if (fn, a) not in dev_expected:
                continue
            v = np.array([r["eta_needle"] for r in recs
                          if r["t"] == 0 and r["a"] == a])
            dev = int(np.sum(np.abs(v - 1.0 / L) > 0.02))
            check(f"interleaved eta(0) deviations L={L} a={a}",
                  float(dev), float(dev_expected[(fn, a)]), 0.5)
    # interleaved/one-shot floor ratios (manuscript: 441, 7.7, 17 at
    # f = 0.5, 0.6, 0.8 -- NOT "10-30x")
    af = json.load(open(DATA / "alpha_fits.json"))
    n10 = [r for r in af if r["n"] == 10]
    inter = {r["a"]: r["floor"] for r in n10[:3]}   # first triple: L=4
    ones = {r["a"]: r["floor"] for r in n10[3:6]}   # second triple: one-shot
    for a, exp in [(5, 440.9), (6, 7.67), (8, 17.20)]:
        check(f"interleaved/one-shot floor ratio a={a}",
              inter[a] / ones[a], exp, exp * 0.01)

    # ---- 11. figures regenerate ----------------------------------------
    run([PY, PKG / "make_figures.py", "--data", DATA, "--out",
         ROOT / "figures"], cwd=ROOT)
    for f in ["fig1_phase_diagram.png", "fig2_sweetspot.png",
              "fig3_alpha.png"]:
        ok = (ROOT / "figures" / f).exists()
        print(f"  [{'OK' if ok else 'FAIL'}] figure {f}")
        if not ok:
            FAILURES.append(f)

    if args.full:
        print("\n--- FULL tier: regenerating sweeps (hours) ---")
        run([PY, "prototype_phase_diagram.py", "--n", "8", "--seeds", "8",
             "--mode", "both"])
        run([PY, "prototype_lean.py", "--n", "10", "--seeds", "16",
             "--mode", "oneshot", "--ident", "--out", "lean2"])
        run([PY, "branch_tree.py", "--n", "100", "--tmax", "32",
             "--seeds", "24", "--out", "branchtree24"])

    print(f"\n=== reproduction {'PASSED' if not FAILURES else 'FAILED'} "
          f"({len(FAILURES)} failures) ===")
    for f in FAILURES:
        print(f"  FAILED: {f}")
    sys.exit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
