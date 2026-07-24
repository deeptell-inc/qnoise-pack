#!/usr/bin/env python
"""Lean large-n sweep for the scrambled-signal learnability phase diagram.

Differences from prototype_phase_diagram.py (kept intact for n<=8 reference):
  - never builds 2^n x 2^n matrices: Clifford blocks are synthesized circuits
    applied to statevectors, T gates are diagonal phases, the encoder
    e^{-i theta X_0} is a 2-term combination. Scales to n = 12+.
  - drops the uniform-budget regression (established flat control at n=8);
    keeps the physics metrics:
      F_Q^A, F_Q, eta_needle = s_max^2 / F_Q^A,
      m_eff = (sum s^2)^2 / sum s^4   (effective number of Pauli settings
                                       carrying the sensitivity -- the
                                       quantity in the Theorem-2 conjecture)
  - optional --ident: Theorem-1 identifiability sweep. Dictionary
    {X_i, Y_i}_{i<n} (2n generators with nonzero variance on |0..0>),
    exact first-order responses R_mu = -i Tr_env[G_hat_mu, rho0_hat] on A,
    Gram matrix spectrum -> identifiability rank/conditioning per (t, a).
    Prediction: magic HELPS identifiability (fills the rank) while it HURTS
    extraction (eta_needle) -> intermediate-t sweet spot.

Usage:
  python prototype_lean.py --n 10 --seeds 8 --mode both --ident
  python prototype_lean.py --n 12 --seeds 8 --mode both
"""

import argparse
import json
import time
from itertools import product as iproduct

import numpy as np
from qiskit.quantum_info import random_clifford

from prototype_phase_diagram import (
    pauli_coeffs, reduced_density, qfi_from_states, aggregate)


# ----------------------------------------------------------- state pipeline

def synth_blocks(n, t, layers, rng):
    """Clifford blocks as synthesized circuits + T-gate positions.

    Circuit = B_layers ... B_1 B_0 with t T gates split across the `layers`
    slots after B_0..B_{layers-1}. Returns list of (circuit, t_qubits) pairs
    applied in order; encoding slots sit between consecutive pairs.
    """
    tk = [t // layers + (1 if i < t % layers else 0) for i in range(layers)]
    blocks = []
    for k in range(layers + 1):
        circ = random_clifford(
            n, seed=int(rng.integers(2 ** 31))).to_circuit()
        tqs = ([int(rng.integers(n)) for _ in range(tk[k])]
               if k < layers else [])
        blocks.append((circ, tqs))
    return blocks


def apply_block(vec, block, n, sv_cls):
    """Apply one (clifford circuit, T positions) block to a statevector."""
    circ, tqs = block
    v = sv_cls(vec).evolve(circ).data
    for q in tqs:
        phase = np.exp(1j * np.pi / 4 * ((np.arange(len(v)) >> q) & 1))
        v = phase * v
    return v


def apply_x0(vec):
    """X on qubit 0 (LSB): swap amplitude pairs."""
    return vec.reshape(-1, 2)[:, ::-1].reshape(-1)


def make_phi(blocks, mode, n):
    """Return phi(theta) evaluator using only vector operations."""
    from qiskit.quantum_info import Statevector as SV
    psi0 = np.zeros(2 ** n, dtype=complex)
    psi0[0] = 1.0

    if mode == "oneshot":
        u0, u1 = psi0, apply_x0(psi0)
        for b in blocks:
            u0 = apply_block(u0, b, n, SV)
            u1 = apply_block(u1, b, n, SV)

        def phi(theta):
            return np.cos(theta) * u0 - 1j * np.sin(theta) * u1
        return phi, (u0, u1)

    def phi(theta):  # interleaved
        v = apply_block(psi0, blocks[0], n, SV)
        c, s = np.cos(theta), -1j * np.sin(theta)
        for b in blocks[1:]:
            v = c * v + s * apply_x0(v)
            v = apply_block(v, b, n, SV)
        return v
    return phi, None


# ---------------------------------------------------------------- metrics

def sensitivity_spectrum(phi, n, a, dth=1e-3):
    """s_j = d<P_j>/dtheta for all nontrivial Paulis on A; spread stats."""
    rp = reduced_density(phi(dth), n, a)
    rm = reduced_density(phi(-dth), n, a)
    ep = pauli_coeffs(rp, a).reshape(-1).real * 2 ** a
    em = pauli_coeffs(rm, a).reshape(-1).real * 2 ** a
    s = (ep[1:] - em[1:]) / (2 * dth)
    s2 = s ** 2
    tot = float(s2.sum())
    m_eff = float(tot ** 2 / max(np.sum(s2 ** 2), 1e-300))
    return dict(s_max_sq=float(s2.max()), s_sum_sq=tot, m_eff=m_eff)


def identifiability_gram(blocks, n, a):
    """Theorem-1 numerics (oneshot only): Gram of first-order responses.

    Dictionary G_mu in {X_i, Y_i}: R_mu = -i Tr_env(|u_mu><u0| - |u0><u_mu|)
    with u_mu = U G_mu |0..0>. Returns spectrum stats of the m x m Gram
    matrix of the responses on A (m = 2n), HS inner product.
    """
    from qiskit.quantum_info import Statevector as SV
    psi0 = np.zeros(2 ** n, dtype=complex)
    psi0[0] = 1.0

    def evolve_all(vec):
        for b in blocks:
            vec = apply_block(vec, b, n, SV)
        return vec

    u0 = evolve_all(psi0)
    responses = []
    for i in range(n):
        for pauli in ("x", "y"):
            v = psi0.copy().reshape(-1)
            # G_mu |0..0>: X_i -> flip bit i; Y_i -> i * flip bit i (|0>-><1|)
            idx = 1 << i
            w = np.zeros_like(v)
            w[idx] = 1.0 if pauli == "x" else 1j
            u_mu = evolve_all(w)
            psi_a0 = u0.reshape(2 ** (n - a), 2 ** a)
            psi_am = u_mu.reshape(2 ** (n - a), 2 ** a)
            cross = np.einsum("ei,ej->ij", psi_am, psi_a0.conj())
            R = -1j * (cross - cross.conj().T)
            responses.append(R.reshape(-1))
    M = np.array(responses)
    gram = (M @ M.conj().T).real
    ev = np.linalg.eigvalsh(gram)
    ev = np.clip(ev, 0.0, None)
    lam_max = float(ev[-1])
    # absolute tolerance anchored to the Weingarten mean E[diag] = 2g:
    # a lam_max-relative tolerance miscounts rank when the Gram is ~ 0
    # (all needles invisible), inflating rank with float noise.
    d_A, d = 2 ** a, 2 ** n
    two_g = 2 * d * (d_A ** 2 - 1) / (d_A * (d ** 2 - 1))
    rank_tol = max(lam_max * 1e-8, two_g * 1e-6)
    return dict(gram_lam_min=float(ev[0]), gram_lam_max=lam_max,
                gram_rank=int(np.sum(ev > rank_tol)), gram_m=len(ev),
                gram_eff_rank=float(np.sum(ev) ** 2
                                    / max(np.sum(ev ** 2), 1e-300)))


def run_cell(n, t, a, seed, mode, layers, ident):
    rng = np.random.default_rng([seed, t, a, 0 if mode == "oneshot" else 1])
    blocks = synth_blocks(n, t, layers if mode == "interleaved"
                          else max(t, 1), rng)
    phi, _ = make_phi(blocks, mode, n)

    delta = 1e-4
    rho_0 = reduced_density(phi(0.0), n, a)
    drho = (reduced_density(phi(delta), n, a)
            - reduced_density(phi(-delta), n, a)) / (2 * delta)
    fqa = qfi_from_states(rho_0, drho)
    psi_p, psi_m, psi_0 = phi(delta), phi(-delta), phi(0.0)
    drho_full = (np.outer(psi_p, psi_p.conj())
                 - np.outer(psi_m, psi_m.conj())) / (2 * delta) \
        if n <= 10 else None
    if drho_full is not None:
        fq_full = qfi_from_states(np.outer(psi_0, psi_0.conj()), drho_full)
    else:  # pure state: F_Q = 4 (<dphi|dphi> - |<phi|dphi>|^2), FD form
        dpsi = (psi_p - psi_m) / (2 * delta)
        ov = np.vdot(psi_0, dpsi)
        fq_full = float(4 * (np.vdot(dpsi, dpsi).real - abs(ov) ** 2))

    spec = sensitivity_spectrum(phi, n, a)
    eta_needle = spec["s_max_sq"] / fqa if fqa > 1e-9 else 0.0
    rec = dict(n=n, t=t, a=a, seed=seed, mode=mode, fqa=fqa,
               fq_full=fq_full, eta_needle=float(eta_needle), **spec)
    if ident and mode == "oneshot":
        rec.update(identifiability_gram(blocks, n, a))
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--mode", choices=["oneshot", "interleaved", "both"],
                   default="both")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--t", type=int, nargs="*",
                   default=[0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    p.add_argument("--a", type=int, nargs="*", default=None)
    p.add_argument("--ident", action="store_true")
    p.add_argument("--out", default="lean")
    args = p.parse_args()

    if args.a is None:
        args.a = {10: [3, 5, 6, 8], 12: [3, 6, 8, 9]}.get(
            args.n, [max(2, args.n // 4), args.n // 2,
                     args.n // 2 + 1, 3 * args.n // 4])

    modes = (["oneshot", "interleaved"] if args.mode == "both"
             else [args.mode])
    for mode in modes:
        records, t0 = [], time.time()
        cells = list(iproduct(args.t, args.a, range(args.seeds)))
        for i, (t, a, seed) in enumerate(cells):
            records.append(run_cell(args.n, t, a, seed, mode, args.layers,
                                    args.ident))
            if (i + 1) % 20 == 0 or i == len(cells) - 1:
                print(f"[{mode}] {i + 1}/{len(cells)} cells, "
                      f"{time.time() - t0:.0f}s", flush=True)
        tag = f"{args.out}_{mode}_n{args.n}"
        with open(f"{tag}.json", "w") as f:
            json.dump(records, f, indent=1)
        print(f"records -> {tag}.json")

        agg_f, agg_ff = aggregate(records, "fqa"), aggregate(records,
                                                             "fq_full")
        agg_e = aggregate(records, "eta_needle")
        print(f"\n=== {mode} n={args.n}: F_Q^A/F_Q | log10(eta_needle) ===")
        print("a\\t " + " ".join(f"{t:>12d}" for t in args.t))
        for a in args.a:
            row = [f"{agg_f[(t, a)][0] / max(agg_ff[(t, a)][0], 1e-12):.2f}"
                   f"/{np.log10(max(agg_e[(t, a)][0], 1e-12)):+.1f}"
                   for t in args.t]
            print(f"a={a}  " + " ".join(f"{s:>12s}" for s in row))
        if args.ident and mode == "oneshot":
            agg_r = aggregate(records, "gram_rank")
            agg_l = aggregate(records, "gram_lam_min")
            print(f"\n--- identifiability: mean rank(Gram)/m | "
                  f"log10 lam_min ---")
            m = records[0]["gram_m"]
            for a in args.a:
                row = [f"{agg_r[(t, a)][0] / m:.2f}"
                       f"/{np.log10(max(agg_l[(t, a)][0], 1e-12)):+.1f}"
                       for t in args.t]
                print(f"a={a}  " + " ".join(f"{s:>12s}" for s in row))


if __name__ == "__main__":
    main()
