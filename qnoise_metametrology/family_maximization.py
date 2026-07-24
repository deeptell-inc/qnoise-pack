#!/usr/bin/env python
"""Commuting-family maximization + purity-rate check (Theorem-2 upgrade).

For each (t, a, seed) cell of the one-shot t-doped Clifford ensemble:
  - full sensitivity vector s_j over all Paulis on A
  - eta_needle  = max_j s_j^2 / F_Q^A              (single setting, as before)
  - eta_family  = F_C(B_fam) / F_Q^A where B_fam is the joint eigenbasis of
                  a GREEDY max-sensitivity commuting family (mutually
                  commuting Paulis = one measurement setting). F_C is the
                  classical Fisher information of that projective basis, so
                  eta_family <= 1 rigorously (naive sum_fam s_j^2
                  double-counts correlated estimators and can exceed F_Q).
                  Greedy over top-K by s^2 -- a lower bound on the true
                  best-family value.
  - purity      = sum_j p_j^2 with p_j = s_j^2 / sum s^2
                  (Theorem-2 chain predicts E[purity] ~ (3/4)^t under
                  Clifford twirling, hence eta <= C * (3/4)^{t/2},
                  alpha_theory = ln(4/3)/2 = 0.1438)

Usage: python family_maximization.py --n 10 --a 6 8 --seeds 8
"""

import argparse
import json
import time
from itertools import product as iproduct

import numpy as np

from prototype_phase_diagram import pauli_coeffs, reduced_density, \
    qfi_from_states
from prototype_lean import synth_blocks, make_phi


def pauli_xz_bits(a):
    """x, z bit arrays for all 4^a Paulis, index axis order [p_{a-1}..p_0]."""
    idx = np.arange(4 ** a)
    digits = np.array([(idx // 4 ** q) % 4 for q in range(a)])  # (a, 4^a)
    x = ((digits == 1) | (digits == 2)).astype(np.uint8)
    z = ((digits == 2) | (digits == 3)).astype(np.uint8)
    return x, z  # shape (a, 4^a)


def greedy_family(s2, x, z, top_k=4096):
    """Greedy max-weight mutually-commuting family. Returns member indices."""
    order = np.argsort(s2)[::-1][:top_k]
    fam = []
    for j in order:
        if s2[j] <= 0:
            break
        ok = True
        for k in fam:
            # symplectic form: commute iff x_j.z_k + z_j.x_k = 0 mod 2
            if (int(np.dot(x[:, j], z[:, k]) + np.dot(z[:, j], x[:, k]))
                    % 2):
                ok = False
                break
        if ok:
            fam.append(j)
    return fam


PAULI2 = [np.eye(2, dtype=complex),
          np.array([[0, 1], [1, 0]], dtype=complex),
          np.array([[0, -1j], [1j, 0]], dtype=complex),
          np.array([[1, 0], [0, -1]], dtype=complex)]


def pauli_matrix(idx, a):
    """Pauli matrix on a qubits from flat index (axis order [p_{a-1}..p_0])."""
    M = np.ones((1, 1), dtype=complex)
    for q in range(a - 1, -1, -1):
        M = np.kron(M, PAULI2[(idx // 4 ** q) % 4])
    return M


def family_classical_fisher(fam, rho0, drho, a, rng):
    """F_C of the joint eigenbasis of the commuting family (one setting)."""
    M = np.zeros((2 ** a, 2 ** a), dtype=complex)
    for j in fam:
        M += rng.normal() * pauli_matrix(j, a)
    _, V = np.linalg.eigh(M)  # generic weights -> common eigenbasis
    p = np.einsum("ki,ij,jk->k", V.conj().T, rho0, V).real
    dp = np.einsum("ki,ij,jk->k", V.conj().T, drho, V).real
    mask = p > 1e-12
    return float(np.sum(dp[mask] ** 2 / p[mask]))


def run_cell(n, t, a, seed, x, z, dth=1e-3):
    rng = np.random.default_rng([seed, t, a, 0])  # same circuits as lean
    blocks = synth_blocks(n, t, max(t, 1), rng)
    phi, _ = make_phi(blocks, "oneshot", n)

    rp = reduced_density(phi(dth), n, a)
    rm = reduced_density(phi(-dth), n, a)
    ep = pauli_coeffs(rp, a).reshape(-1).real * 2 ** a
    em = pauli_coeffs(rm, a).reshape(-1).real * 2 ** a
    s = (ep - em) / (2 * dth)  # includes identity slot 0 (s=0 there)
    s2 = s ** 2
    s2[0] = 0.0
    tot = float(s2.sum())

    delta = 1e-4
    rho0 = reduced_density(phi(0.0), n, a)
    drho = (reduced_density(phi(delta), n, a)
            - reduced_density(phi(-delta), n, a)) / (2 * delta)
    fqa = qfi_from_states(rho0, drho)

    if fqa < 1e-9 or tot < 1e-18:
        return dict(n=n, t=t, a=a, seed=seed, fqa=fqa, eta_needle=0.0,
                    eta_family=0.0, fam_size=0, purity=0.0)
    fam = greedy_family(s2, x, z)
    fc = family_classical_fisher(fam, rho0, drho, a,
                                 np.random.default_rng([seed, t, a, 7]))
    p = s2 / tot
    return dict(n=n, t=t, a=a, seed=seed, fqa=fqa,
                eta_needle=float(s2.max() / fqa),
                eta_family=float(min(fc / fqa, 1.0)),
                fam_size=len(fam),
                purity=float(np.sum(p ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--a", type=int, nargs="*", default=[6, 8])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--t", type=int, nargs="*",
                    default=[0, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64])
    ap.add_argument("--out", default="family")
    args = ap.parse_args()

    xz = {a: pauli_xz_bits(a) for a in args.a}
    records, t0 = [], time.time()
    cells = list(iproduct(args.t, args.a, range(args.seeds)))
    for i, (t, a, seed) in enumerate(cells):
        x, z = xz[a]
        records.append(run_cell(args.n, t, a, seed, x, z))
        if (i + 1) % 20 == 0 or i == len(cells) - 1:
            print(f"{i + 1}/{len(cells)} cells, {time.time() - t0:.0f}s",
                  flush=True)
    tag = f"{args.out}_oneshot_n{args.n}"
    with open(f"{tag}.json", "w") as f:
        json.dump(records, f, indent=1)
    print(f"-> {tag}.json")

    print(f"\n=== n={args.n}: log10 eta_needle | log10 eta_family | "
          f"purity vs (3/4)^t ===")
    for a in args.a:
        print(f"\n a={a} (f={a / args.n:.2f}):")
        print(f" {'t':>3} {'eta_needle':>11} {'eta_family':>11} "
              f"{'fam_size':>8} {'purity':>9} {'(3/4)^t':>9}")
        for t in args.t:
            cell = [r for r in records if r["t"] == t and r["a"] == a]
            en = np.mean([r["eta_needle"] for r in cell])
            ef = np.mean([r["eta_family"] for r in cell])
            fs = np.mean([r["fam_size"] for r in cell])
            pu = np.mean([r["purity"] for r in cell])
            print(f" {t:>3} {en:>11.4f} {ef:>11.4f} {fs:>8.1f} "
                  f"{pu:>9.4f} {0.75 ** t:>9.4f}")


if __name__ == "__main__":
    main()
