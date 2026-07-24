#!/usr/bin/env python
"""End-to-end meta-metrology under realistic observation noise.

Protocol (one-shot encoding, n qubits, access A of a qubits):
  1. CALIBRATE: for each dictionary generator G_mu in {X_i, Y_i}, inject
     theta = +/- delta and estimate all Pauli expectations on A with S shots
     per setting, with readout bit-flip error q per qubit
     (<P>_meas = (1-2q)^{weight(P)} <P>_true + shot noise).
     -> noisy reference responses Rhat_mu and sensitivities shat.
  2. IDENTIFY: nature encodes an unknown G* (random dictionary element)
     at theta = +/- delta; estimate its response Rhat_* the same way;
     solve Rhat_* = sum_mu c_mu Rhat_mu (least squares);
     declare mu_hat = argmax |c_mu|.
  3. ESTIMATE: using the identified generator's calibrated best setting j*,
     estimate an unknown theta_sig from S shots: theta_hat =
     x_meas(j*) / shat(j*).

Outputs per (t, S, q): identification success rate, |theta| estimation
relative error. Shows whether the sweet-spot window survives finite shots
and how the magic collapse inflates the shot cost.
"""

import argparse
import json
import time

import numpy as np

from prototype_phase_diagram import pauli_coeffs, reduced_density
from prototype_lean import synth_blocks, apply_block


def pauli_weights(a):
    idx = np.arange(4 ** a)
    w = np.zeros(4 ** a, dtype=int)
    for q in range(a):
        w += ((idx // 4 ** q) % 4 != 0).astype(int)
    return w


def exact_features(state, n, a):
    rho = reduced_density(state, n, a)
    return pauli_coeffs(rho, a).reshape(-1).real * 2 ** a  # index 0 = identity


def measure(x_true, weights, S, q, rng):
    """Readout-error-shrunk expectations + binomial shot noise (Gaussian)."""
    shrink = (1.0 - 2.0 * q) ** weights
    x = shrink * x_true
    sig = np.sqrt(np.clip(1.0 - x ** 2, 0.0, 1.0) / S)
    return x + rng.normal(0.0, 1.0, x.shape) * sig


def run_cell(n, a, t, seed, S_list, q_list, delta=0.1, theta_sig=0.05,
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
    # dictionary {X_i, Y_i}: G_mu|0..0> = (1 or 1j) * |bit i flipped>
    u_mu = []
    for i in range(n):
        for ph in (1.0, 1j):
            w = np.zeros(2 ** n, dtype=complex)
            w[1 << i] = ph
            u_mu.append(evolve_all(w))
    m = len(u_mu)

    # exact features at theta = +/- delta for every encoder, plus theta_sig
    weights = pauli_weights(a)

    def phi(mu, th):
        return np.cos(th) * u0 - 1j * np.sin(th) * u_mu[mu]

    feats_p = np.array([exact_features(phi(mu, delta), n, a)
                        for mu in range(m)])
    feats_m = np.array([exact_features(phi(mu, -delta), n, a)
                        for mu in range(m)])

    results = {}
    noise_rng = np.random.default_rng([seed, t, a, 99])
    for S in S_list:
        for q in q_list:
            succ, rel_err = 0, []
            for _ in range(n_trials):
                # calibration: noisy responses for all mu
                R_cal = np.array([
                    (measure(feats_p[mu], weights, S, q, noise_rng)
                     - measure(feats_m[mu], weights, S, q, noise_rng))
                    / (2 * delta) for mu in range(m)])
                # unknown signal generator
                mu_true = int(noise_rng.integers(m))
                R_sig = (measure(feats_p[mu_true], weights, S, q, noise_rng)
                         - measure(feats_m[mu_true], weights, S, q,
                                   noise_rng)) / (2 * delta)
                c, *_ = np.linalg.lstsq(R_cal.T, R_sig, rcond=None)
                mu_hat = int(np.argmax(np.abs(c)))
                if mu_hat == mu_true:
                    succ += 1
                # estimation with the identified generator's best setting
                s_hat = R_cal[mu_hat]
                jstar = int(np.argmax(np.abs(s_hat)))
                x_th = exact_features(phi(mu_true, theta_sig), n, a)
                x0 = exact_features(phi(mu_true, 0.0), n, a)
                meas = measure(x_th - x0, weights, S, q, noise_rng)
                th_hat = meas[jstar] / s_hat[jstar]
                rel_err.append(abs(abs(th_hat) - theta_sig) / theta_sig)
            results[(S, q)] = (succ / n_trials, float(np.median(rel_err)))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--a", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--t", type=int, nargs="*", default=[0, 2, 6, 12, 16, 32])
    ap.add_argument("--shots", type=int, nargs="*",
                    default=[100, 1000, 10000, 100000])
    ap.add_argument("--readout", type=float, nargs="*", default=[0.0, 0.02])
    ap.add_argument("--out", default="noise_recovery")
    args = ap.parse_args()

    all_rec, t0 = [], time.time()
    for t in args.t:
        agg = {}
        for seed in range(args.seeds):
            res = run_cell(args.n, args.a, t, seed, args.shots, args.readout)
            for k, v in res.items():
                agg.setdefault(k, []).append(v)
        for (S, q), vals in agg.items():
            succ = float(np.mean([v[0] for v in vals]))
            err = float(np.median([v[1] for v in vals]))
            all_rec.append(dict(n=args.n, a=args.a, t=t, S=S, q=q,
                                p_identify=succ, rel_err_theta=err))
        print(f"t={t} done ({time.time() - t0:.0f}s)", flush=True)

    tag = f"{args.out}_n{args.n}_a{args.a}"
    with open(f"{tag}.json", "w") as f:
        json.dump(all_rec, f, indent=1)
    print(f"-> {tag}.json\n")
    print(f"=== n={args.n}, a={args.a} (f={args.a / args.n}): "
          f"P[identify] | median rel.err(theta) ===")
    hdr = "t\\S,q " + " ".join(f"{S},q={q:>4}" for S in args.shots
                               for q in args.readout)
    print(hdr)
    for t in args.t:
        row = []
        for S in args.shots:
            for q in args.readout:
                r = [x for x in all_rec if x["t"] == t and x["S"] == S
                     and x["q"] == q][0]
                row.append(f"{r['p_identify']:.2f}|{r['rel_err_theta']:.2f}")
        print(f"t={t:>3} " + " ".join(f"{s:>12s}" for s in row))


if __name__ == "__main__":
    main()
