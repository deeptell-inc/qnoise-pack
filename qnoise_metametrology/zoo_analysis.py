#!/usr/bin/env python
"""Summary statistics for the scrambler-family experiments (scrambler_zoo.py).

Reads data/zoo_*_n{N}.json and writes data/zoo_summary_n{N}.json:
  - decay rate of the best-Pauli fraction per layer / per non-Gaussian gate
    (fit of log eta to (eta0 - c) exp(-r x) + c on the seed-mean trace,
    with a bootstrap over circuit seeds),
  - late-depth plateaus and their ratio to the global-Haar reference,
  - selection-rule counts, light-cone arrival depths, rank filling.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

DATA = Path(__file__).resolve().parent.parent / "data"


def model(x, r, c, e0):
    return np.log((e0 - c) * np.exp(-r * x) + c)


def fit_rate(xs, ys):
    xs, ys = np.asarray(xs, float), np.clip(np.asarray(ys, float), 1e-12, None)
    try:
        p, _ = curve_fit(model, xs, np.log(ys), p0=[0.2, ys[-1], ys[0]],
                         bounds=([0, 1e-6, 1e-6], [5, 1, 1.5]), maxfev=20000)
        return float(p[0]), float(p[1])
    except Exception:
        return float("nan"), float("nan")


def trace(recs, xkey, val):
    xs = sorted({r[xkey] for r in recs})
    seeds = sorted({r["seed"] for r in recs})
    M = np.array([[val(next(r for r in recs if r[xkey] == x and r["seed"] == s))
                   for x in xs] for s in seeds])
    return np.array(xs), M                      # M[seed, x]


def rate_with_ci(recs, xkey, xmin=0, nboot=400, seed=0):
    xs, M = trace(recs, xkey, lambda r: r["eta_needle"])
    keep = xs >= xmin
    r, c = fit_rate(xs[keep], M.mean(0)[keep])
    rng = np.random.default_rng(seed)
    bs = [fit_rate(xs[keep], M[rng.integers(len(M), size=len(M))].mean(0)[keep])[0]
          for _ in range(nboot)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(rate=r, floor=c, ci=[float(lo), float(hi)])


def plateau(recs, xkey, val, last=3):
    xs, M = trace(recs, xkey, val)
    return float(M[:, -last:].mean()), float(M[:, -last:].mean(1).std())


def main(n):
    ratio = lambda r: r["fqa"] / r["fq_full"]
    eta = lambda r: r["eta_needle"]
    out = {"n": n}
    H = json.load(open(DATA / f"zoo_haar_n{n}.json"))
    a_list = sorted({r["a"] for r in H})
    out["haar"] = {str(a): dict(
        fq_ratio=float(np.mean([ratio(r) for r in H if r["a"] == a])),
        eta=float(np.mean([eta(r) for r in H if r["a"] == a])),
        m_eff=float(np.mean([r["m_eff"] for r in H if r["a"] == a])))
        for a in a_list}

    # ---- brickwork
    B = json.load(open(DATA / f"zoo_brickwork_n{n}.json"))
    bw = {}
    for a in a_list:
        sel = lambda g, p: [r for r in B if r["gate"] == g and r["a"] == a and r["place"] == p]
        hn, hf, cn = sel("haar", "near"), sel("haar", "far"), sel("clifford", "near")
        xs, F = trace(hf, "depth", ratio)
        arrive = int(xs[np.argmax(F.mean(0) > 1e-6)])
        xs, Fn = trace(hn, "depth", ratio)
        xs, En = trace(hn, "depth", eta)
        xs, K = trace(hn, "depth", lambda r: r["gram_rank"] / r["gram_m"])
        href = out["haar"][str(a)]
        cl = [r for r in B if r["gate"] == "clifford" and r["a"] == a]
        allnone = float(np.mean([min(abs(eta(r)), abs(eta(r) - 1)) < 1e-6 for r in cl]))
        bw[str(a)] = dict(
            lightcone_arrival_depth=arrive, distance=n - a,
            eta_rate_per_layer=rate_with_ci(hn, "depth", xmin=1),
            depth_fq_within_5pct_of_haar=int(xs[np.argmax(np.abs(Fn.mean(0) - href["fq_ratio"]) < 0.05 * href["fq_ratio"] + 0.0)]) if a < n else 0,
            depth_eta_within_1p5x_of_haar=int(xs[np.argmax(En.mean(0) < 1.5 * href["eta"])]),
            depth_full_rank=int(xs[np.argmax(K.mean(0) > 0.999)]),
            eta_plateau=plateau(hn, "depth", eta)[0],
            clifford_eta_is_0_or_1_fraction=allnone,
            clifford_rank_plateau=plateau(cn, "depth", lambda r: r["gram_rank"] / r["gram_m"])[0])
    out["brickwork"] = bw

    # ---- U(1)
    U = json.load(open(DATA / f"zoo_u1_n{n}.json"))
    u1 = {}
    for a in a_list:
        s = lambda g: [r for r in U if r["gate"] == g and r["a"] == a and r["place"] == "near"]
        pe_u, pe_h = plateau(s("u1"), "depth", eta)[0], plateau(s("haar"), "depth", eta)[0]
        pm_u, pm_h = plateau(s("u1"), "depth", lambda r: r["m_eff"])[0], plateau(s("haar"), "depth", lambda r: r["m_eff"])[0]
        act = sorted({r["n_active"] for r in s("u1") if r["depth"] >= 32})
        u1[str(a)] = dict(
            fq_plateau_u1=plateau(s("u1"), "depth", ratio)[0],
            fq_plateau_generic=plateau(s("haar"), "depth", ratio)[0],
            eta_plateau_u1=pe_u, eta_plateau_generic=pe_h, eta_ratio=pe_u / pe_h,
            m_eff_ratio=pm_u / pm_h,
            active_settings_u1=act, selection_rule_count=2 ** (2 * a - 1),
            all_settings=4 ** a - 1,
            eta_rate_u1=rate_with_ci(s("u1"), "depth", xmin=2),
            eta_rate_generic=rate_with_ci(s("haar"), "depth", xmin=2))
    out["u1"] = u1

    # ---- matchgate
    M = json.load(open(DATA / f"zoo_matchgate_n{n}.json"))
    mg = {}
    for a in a_list:
        s = [r for r in M if r["a"] == a]
        k0 = [r for r in s if r["k"] == 0]
        xs, K = trace(s, "k", lambda r: r["gram_rank"] / r["gram_m"])
        mg[str(a)] = dict(
            gaussian_rank=sorted({r["gram_rank"] for r in k0}), two_a=2 * a,
            gaussian_eta=float(np.mean([eta(r) for r in k0])),
            gaussian_eta_std=float(np.std([eta(r) for r in k0])),
            gaussian_fq_ratio=float(np.mean([ratio(r) for r in k0])),
            k_full_rank=int(xs[np.argmax(K.mean(0) > 0.999)]),
            eta_rate_per_cz=rate_with_ci(s, "k"),
            eta_plateau=plateau(s, "k", eta, last=2)[0],
            eta_plateau_over_haar=plateau(s, "k", eta, last=2)[0] / out["haar"][str(a)]["eta"],
            fq_plateau=plateau(s, "k", ratio, last=2)[0],
            active_settings=sorted({r["n_active"] for r in s if r["k"] >= 8}))
    out["matchgate"] = mg
    json.dump(out, open(DATA / f"zoo_summary_n{n}.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    o = main(n)
    print(json.dumps(o, indent=1))
