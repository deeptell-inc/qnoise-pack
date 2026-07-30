#!/usr/bin/env python
"""Same-total-budget comparison of identification vs estimation.

Addresses the resource-accounting asymmetry (cross-family panel, Priority 1):
both tasks are restricted to the SAME K randomly chosen Pauli settings on A,
and every shot is charged --- calibration (m generators x 2 theta-points x K
settings x S shots), signal acquisition (2 x K x S), and estimation (1
setting x S). Total budget B = S * K * (2m + 2) + S.

Protocol per (t, seed, trial):
  1. draw K settings uniformly from the 4^a - 1 nontrivial Paulis on A;
  2. calibrate noisy K-dim reference responses for all m dictionary elements;
  3. nature picks G* uniformly; acquire its noisy K-dim response; identify
     by least squares over the calibrated references;
  4. estimate theta_sig with the identified generator's best calibrated
     setting AMONG THE K (no access to uncalibrated settings).

Outputs P_id and median relative error vs (t, K, S) with the charged budget.
"""

import argparse
import json
import time

import numpy as np

from prototype_phase_diagram import pauli_coeffs, reduced_density
from prototype_lean import synth_blocks, apply_block


def exact_features(state, n, a):
    rho = reduced_density(state, n, a)
    return pauli_coeffs(rho, a).reshape(-1).real * 2 ** a


def measure(x_true, S, rng):
    sig = np.sqrt(np.clip(1.0 - x_true ** 2, 0.0, 1.0) / S)
    return x_true + rng.normal(0.0, 1.0, x_true.shape) * sig


def run_cell(n, a, t, seed, K_list, S_list, delta=0.1, theta_sig=0.05,
             n_trials=20):
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
            succ, rel = 0, []
            for _ in range(n_trials):
                idx = nrng.choice(np.arange(1, n_feat), size=K,
                                  replace=False)
                # calibration: m x 2 theta-points x K settings, S shots each
                Rcal = np.array([
                    (measure(feats_p[mu][idx], S, nrng)
                     - measure(feats_m[mu][idx], S, nrng)) / (2 * delta)
                    for mu in range(m)])
                mu_true = int(nrng.integers(m))
                Rsig = (measure(feats_p[mu_true][idx], S, nrng)
                        - measure(feats_m[mu_true][idx], S, nrng)) \
                    / (2 * delta)
                c, *_ = np.linalg.lstsq(Rcal.T, Rsig, rcond=None)
                mu_hat = int(np.argmax(np.abs(c)))
                if mu_hat == mu_true:
                    succ += 1
                # estimation restricted to the K calibrated settings
                s_hat = Rcal[mu_hat]
                jstar = int(np.argmax(np.abs(s_hat)))
                x_th = exact_features(phi(mu_true, theta_sig), n, a)[idx]
                x_00 = exact_features(phi(mu_true, 0.0), n, a)[idx]
                mmt = measure(x_th - x_00, S, nrng)
                th_hat = mmt[jstar] / s_hat[jstar]
                rel.append(abs(abs(th_hat) - theta_sig) / theta_sig)
            budget = S * K * (2 * m + 2) + S
            out[(K, S)] = (succ / n_trials, float(np.median(rel)), budget)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--a", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--t", type=int, nargs="*", default=[0, 6, 16, 32])
    ap.add_argument("--K", type=int, nargs="*", default=[8, 32, 128])
    ap.add_argument("--S", type=int, nargs="*", default=[1024, 4096])
    ap.add_argument("--out", default="fair_budget")
    args = ap.parse_args()

    recs, t0 = [], time.time()
    for t in args.t:
        agg = {}
        for seed in range(args.seeds):
            res = run_cell(args.n, args.a, t, seed, args.K, args.S)
            for k, v in res.items():
                agg.setdefault(k, []).append(v)
        for (K, S), vals in agg.items():
            recs.append(dict(
                n=args.n, a=args.a, t=t, K=K, S=S,
                budget=vals[0][2],
                p_identify=float(np.mean([v[0] for v in vals])),
                rel_err_theta=float(np.median([v[1] for v in vals]))))
        print(f"t={t} done ({time.time() - t0:.0f}s)", flush=True)

    tag = f"{args.out}_n{args.n}_a{args.a}"
    with open(f"{tag}.json", "w") as f:
        json.dump(recs, f, indent=1)
    print(f"-> {tag}.json\n")
    print("t\\K,S " + " ".join(f"K={K},S={S}" for K in args.K
                               for S in args.S))
    for t in args.t:
        row = []
        for K in args.K:
            for S in args.S:
                r = [x for x in recs if x["t"] == t and x["K"] == K
                     and x["S"] == S][0]
                row.append(f"{r['p_identify']:.2f}|{r['rel_err_theta']:.2f}")
        print(f"t={t:>3} " + " ".join(f"{s:>12s}" for s in row))
    b = [x for x in recs if x["K"] == args.K[0] and x["S"] == args.S[0]][0]
    print(f"\n(example charged budget at K={args.K[0]}, S={args.S[0]}: "
          f"B = {b['budget']:,} shots total, ALL calibration included)")


if __name__ == "__main__":
    main()
