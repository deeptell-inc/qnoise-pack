#!/usr/bin/env python
"""Resource frontier: identification/estimation quality vs TOTAL cost.

Referee request (Nature-style report, item 2): present a fair resource
frontier that accounts for the number of calibrated settings K, the
calibration shot budget, the readout circuit depth, and the classical
post-processing time.

Protocol identical to fair_budget.py (same-total-budget accounting,
every shot charged), extended with:

  * a K sweep (16 ... 4^a - 1) at the magic-window operating point t=16,
    tracing the P_id / relative-error frontier against total budget B;
  * an explicit budget decomposition
        B = B_cal + B_sig + B_est,
        B_cal = S*K*2m  (m dictionary elements x 2 theta points),
        B_sig = 2*K*S,  B_est = S;
  * measured wall-clock classical time of the least-squares
    identification step (per trial mean);
  * readout-depth accounting (static, engineering columns): every Pauli
    setting on A is one layer of single-qubit basis rotations
    (entangling depth 0); no multi-setting joint readout is used.

Deterministic given seeds.  Output: data/resource_frontier_n10_a5.json.
"""

import argparse
import json
import time

import numpy as np

from fair_budget import exact_features, measure
from prototype_lean import synth_blocks, apply_block


def run_cell(n, a, t, seed, K_list, S_list, delta=0.1, theta_sig=0.05,
             n_trials=60):
    from qiskit.quantum_info import Statevector as SV
    rng = np.random.default_rng([seed, t, a, 0])
    blocks = synth_blocks(n, t, max(t, 1), rng)

    def evolve_all(vec):
        for b in blocks:
            vec = apply_block(vec, b, n, SV)
        return vec

    psi0 = np.zeros(2 ** n, dtype=complex)
    psi0[0] = 1.0
    u0 = evolve_all(psi0)
    u_mu = []
    for i in range(n):
        for ph in (1.0, 1j):
            w = np.zeros(2 ** n, dtype=complex)
            w[1 << i] = ph
            u_mu.append(evolve_all(w))
    m = len(u_mu)

    def phi(mu, th):
        return np.cos(th) * u0 - 1j * np.sin(th) * u_mu[mu]

    feats_p = np.array([exact_features(phi(mu, delta), n, a)
                        for mu in range(m)])
    feats_m = np.array([exact_features(phi(mu, -delta), n, a)
                        for mu in range(m)])
    n_feat = feats_p.shape[1]

    out = {}
    nrng = np.random.default_rng([seed, t, a, 77])
    for K in K_list:
        for S in S_list:
            succ, rel, tcl = 0, [], []
            for _ in range(n_trials):
                idx = nrng.choice(np.arange(1, n_feat), size=K,
                                  replace=False)
                Rcal = np.array([
                    (measure(feats_p[mu][idx], S, nrng)
                     - measure(feats_m[mu][idx], S, nrng)) / (2 * delta)
                    for mu in range(m)])
                mu_true = int(nrng.integers(m))
                Rsig = (measure(feats_p[mu_true][idx], S, nrng)
                        - measure(feats_m[mu_true][idx], S, nrng)) \
                    / (2 * delta)
                tc0 = time.perf_counter()
                c, *_ = np.linalg.lstsq(Rcal.T, Rsig, rcond=None)
                mu_hat = int(np.argmax(np.abs(c)))
                tcl.append(time.perf_counter() - tc0)
                if mu_hat == mu_true:
                    succ += 1
                s_hat = Rcal[mu_hat]
                jstar = int(np.argmax(np.abs(s_hat)))
                x_th = exact_features(phi(mu_true, theta_sig), n, a)[idx]
                x_00 = exact_features(phi(mu_true, 0.0), n, a)[idx]
                mmt = measure(x_th - x_00, S, nrng)
                th_hat = mmt[jstar] / s_hat[jstar]
                rel.append(abs(abs(th_hat) - theta_sig) / theta_sig)
            out[(K, S)] = (succ / n_trials, float(np.median(rel)),
                           float(np.mean(tcl)))
    return out, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--a", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--t", type=int, default=16)
    ap.add_argument("--K", type=int, nargs="*",
                    default=[16, 32, 64, 128, 256, 512, 1023])
    ap.add_argument("--S", type=int, nargs="*", default=[1024, 4096])
    args = ap.parse_args()

    t0 = time.time()
    agg, m = {}, None
    for seed in range(args.seeds):
        res, m = run_cell(args.n, args.a, args.t, seed, args.K, args.S)
        for k, v in res.items():
            agg.setdefault(k, []).append(v)
        print(f"seed {seed} done ({time.time()-t0:.0f}s)", flush=True)

    recs = []
    for (K, S), vals in sorted(agg.items()):
        b_cal = S * K * 2 * m
        b_sig = 2 * K * S
        b_est = S
        recs.append(dict(
            n=args.n, a=args.a, t=args.t, K=K, S=S, m=m,
            budget=b_cal + b_sig + b_est, budget_cal=b_cal,
            budget_sig=b_sig, budget_est=b_est,
            cal_share=b_cal / (b_cal + b_sig + b_est),
            p_identify=float(np.mean([v[0] for v in vals])),
            p_identify_seed_sd=float(np.std([v[0] for v in vals])),
            rel_err_theta=float(np.median([v[1] for v in vals])),
            t_classical_ms=float(np.mean([v[2] for v in vals]) * 1e3),
            readout_entangling_depth=0,
            readout_1q_layers=1))

    print(f"\nresource frontier  n={args.n} a={args.a} t={args.t} "
          f"(m={m} dictionary elements, {args.seeds} seeds x 60 trials)")
    print(f"{'K':>5} {'S':>6} {'B_total':>12} {'cal%':>5} {'P_id':>5} "
          f"{'sd':>5} {'rel_err':>8} {'t_cl(ms)':>9}")
    for r in recs:
        print(f"{r['K']:>5} {r['S']:>6} {r['budget']:>12,} "
              f"{100*r['cal_share']:>4.0f}% {r['p_identify']:>5.2f} "
              f"{r['p_identify_seed_sd']:>5.2f} {r['rel_err_theta']:>8.3f} {r['t_classical_ms']:>9.3f}")

    out_path = "../data/resource_frontier_n10_a5.json"
    with open(out_path, "w") as f:
        json.dump(recs, f, indent=1)
    print("wrote", out_path, f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
