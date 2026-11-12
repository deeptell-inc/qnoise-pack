#!/usr/bin/env python
"""Numerical verification of the free-fermion results.

Proposition 7 (rank bound at the free-fermion point).  For an even fermionic
Gaussian state rho on n modes and A = the first a modes, the 2n responses
  T_nu = -i Tr_env([gamma_nu, rho])
span a space of dimension at most 2a.  Proof steps checked here:
  (S1) [X_i, rho0] = [gamma_{2i+1}, rho0], [Y_i, rho0] = [gamma_{2i+2}, rho0]
       on the vacuum (the one-local dictionary is Majorana-linear).
  (S2) rho = exp(-H)/Z with H = (i/4) sum h_{mu nu} gamma_mu gamma_nu gives
       gamma_k rho = sum_mu L_{k mu} rho gamma_mu,  L = (exp(i h))^T > 0.
  (S3) for nu outside A:  Tr_env(gamma_nu rho) = - Tr_env(rho gamma_nu).
  (S4) hence x_env = -(1 + L_ee)^{-1} L_eA x_A  with x_nu = Tr_env(rho gamma_nu),
       so every T_nu lies in span{rho_A gamma_mu : mu in A}.
  (S5) rank = 2a on pure Gaussian states prepared by matchgate circuits.

Degree chain (operator-level benchmark for CZ doping).  Gaussian unitaries
preserve the weight p_d of an operator on Majorana monomials of degree d and,
averaged over Haar SO(2n), spread it uniformly inside each degree sector; a CZ
on neighbouring modes permutes monomials.  The expected degree distribution
therefore obeys p(k+1) = P p(k) with P fixed by counting local patterns; in
particular a degree-1 operator leaves degree 1 with probability 2/n per CZ.

Majorana convention (qubit q = bit q):  gamma_{2j+1} = Z_0..Z_{j-1} X_j,
gamma_{2j+2} = Z_0..Z_{j-1} Y_j.
"""
import itertools
from math import comb

import numpy as np
from scipy.linalg import expm, logm
from scipy.stats import special_ortho_group

from prototype_phase_diagram import pauli_coeffs
from scrambler_zoo import CZ, apply2, brick_layer, initial_batch, matchgate, responses

I2 = np.eye(2)
PX = np.array([[0, 1], [1, 0]], complex)
PY = np.array([[0, -1j], [1j, 0]])
PZ = np.diag([1.0, -1.0]).astype(complex)


def on_qubit(op, q, n):
    return np.kron(np.kron(np.eye(2 ** (n - 1 - q)), op), np.eye(2 ** q))


def majoranas(n):
    g = []
    for j in range(n):
        string = np.eye(2 ** n, dtype=complex)
        for i in range(j):
            string = string @ on_qubit(PZ, i, n)
        g += [string @ on_qubit(PX, j, n), string @ on_qubit(PY, j, n)]
    return g


def ptrace_env(M, n, a):
    """Trace out qubits a..n-1 (A = low bits)."""
    T = M.reshape(2 ** (n - a), 2 ** a, 2 ** (n - a), 2 ** a)
    return np.einsum("eiej->ij", T)


def pauli_degree(n):
    """Majorana degree of every Pauli string, indexed like pauli_coeffs
    (axes [p_{n-1}, ..., p_0], letters 0..3 = I, X, Y, Z)."""
    idx = np.arange(4 ** n)
    deg = np.zeros(4 ** n, int)
    q = np.zeros(4 ** n, int)                # parity of Majoranas at higher sites
    for site in range(n - 1, -1, -1):
        letter = (idx // 4 ** site) % 4
        xy = (letter == 1) | (letter == 2)
        two = ((letter == 3) & (q == 0)) | ((letter == 0) & (q == 1))
        deg += xy + 2 * two
        q ^= xy
    return deg


def gaussian_unitary(O, g):
    """U with U gamma_mu U^dag = sum_nu O[mu, nu] gamma_nu, O in SO(2n).
    (Transpose of the manuscript's index convention; immaterial for the
    Haar-averaged degree chain, since the Haar measure is transpose-invariant.)"""
    A = np.real(logm(O))
    A = (A - A.T) / 2
    H = sum(A[m, k] * g[m] @ g[k] for m in range(len(g)) for k in range(len(g)))
    return expm(-H / 4)


def degree_transition(n):
    """P[d', d] = fraction of degree-d Majorana monomials sent to degree d'
    by CZ on two neighbouring modes (exact pattern count)."""
    g2 = majoranas(2)                         # 4 local Majoranas
    deg2 = pauli_degree(2)
    shift = {}
    for L in itertools.product([0, 1], repeat=4):
        M = np.eye(4, dtype=complex)
        for b, gm in zip(L, g2):
            if b:
                M = M @ gm
        c = np.abs(pauli_coeffs(CZ @ M @ CZ.conj().T, 2).reshape(-1))
        shift[L] = int(deg2[np.argmax(c)]) - sum(L)
    N = 2 * n
    P = np.zeros((N + 1, N + 1))
    for d in range(N + 1):
        for L, dl in shift.items():
            ell = sum(L)
            if 0 <= d - ell <= N - 4:
                P[d + dl, d] += comb(N - 4, d - ell) / comb(N, d)
    return P


def main():
    rng = np.random.default_rng(1)

    # ---- S1: dictionary is Majorana-linear on the vacuum
    n = 4
    g = majoranas(n)
    rho0 = np.zeros((2 ** n, 2 ** n), complex)
    rho0[0, 0] = 1
    for i in range(n):
        X, Y = on_qubit(PX, i, n), on_qubit(PY, i, n)
        assert np.allclose(X @ rho0 - rho0 @ X, g[2 * i] @ rho0 - rho0 @ g[2 * i])
        assert np.allclose(Y @ rho0 - rho0 @ Y, g[2 * i + 1] @ rho0 - rho0 @ g[2 * i + 1])
    # pauli_degree agrees with explicit monomials
    deg = pauli_degree(n)
    for S in [(0,), (1, 2), (0, 3, 5), (2, 4, 6, 7), tuple(range(8))]:
        M = np.eye(2 ** n, dtype=complex)
        for s in S:
            M = M @ g[s]
        c = np.abs(pauli_coeffs(M, n).reshape(-1))
        assert deg[np.argmax(c)] == len(S) and np.isclose(c.max(), 1)
    print("S1 ok: one-local dictionary = Majorana-linear on the vacuum")

    # ---- S2-S4 on a full-rank Gaussian state
    n = 5
    g = majoranas(n)
    N = 2 * n
    h = rng.normal(size=(N, N))
    h = (h - h.T) / 2
    H = 0.25j * sum(h[m, k] * g[m] @ g[k] for m in range(N) for k in range(N))
    rho = expm(-H)
    rho /= np.trace(rho)
    L = expm(1j * h).T
    assert np.allclose(L, L.conj().T) and np.linalg.eigvalsh(L).min() > 0
    for k in range(N):
        assert np.allclose(g[k] @ rho, sum(L[k, m] * rho @ g[m] for m in range(N)), atol=1e-10)
    print("S2 ok: gamma_k rho = sum_mu L[k,mu] rho gamma_mu with L Hermitian > 0")
    for a in (2, 3):
        x = [ptrace_env(rho @ g[v], n, a) for v in range(N)]
        y = [ptrace_env(g[v] @ rho, n, a) for v in range(N)]
        for v in range(2 * a, N):
            assert np.allclose(y[v], -x[v], atol=1e-12)
        e, A = list(range(2 * a, N)), list(range(2 * a))
        C = -np.linalg.solve(np.eye(len(e)) + L[np.ix_(e, e)], L[np.ix_(e, A)])
        for i, v in enumerate(e):
            assert np.allclose(x[v], sum(C[i, j] * x[m] for j, m in enumerate(A)), atol=1e-10)
        T = np.array([(-1j * (y[v] - x[v])).reshape(-1) for v in range(N)])
        rank = int((np.linalg.svd(T, compute_uv=False) > 1e-9).sum())
        assert rank == 2 * a, (rank, a)
    print("S3, S4 ok: y_nu = -x_nu outside A; x_env = -(1+L_ee)^-1 L_eA x_A; rank = 2a")

    # ---- S5: pure Gaussian states from matchgate circuits
    n = 6
    for a in (2, 3, 4, 5):
        for _ in range(5):
            S = initial_batch(n, [0] * n)
            for D in range(2 * n):
                S = brick_layer(S, D, n, matchgate, rng)
            _, R = responses(S, list(range(a)), n)
            sv = np.linalg.svd(R.reshape(R.shape[0], -1), compute_uv=False)
            assert int((sv > 1e-9).sum()) == 2 * a
    print("S5 ok: rank = 2a on pure Gaussian states (n=6, a=2..5)")

    # ---- degree chain: exact transition matrix vs Haar-Gaussian simulation
    n = 5
    g = majoranas(n)
    N = 2 * n
    P = degree_transition(n)
    assert np.allclose(P.sum(0), 1) and np.isclose(P[3, 1], 2 / n) and np.isclose(P[1, 1], 1 - 2 / n)
    deg = pauli_degree(n)
    cz = np.eye(2 ** n, dtype=complex).reshape(1, -1)
    Ucz = apply2(np.eye(2 ** n, dtype=complex), CZ, 1, 2, n).T
    kmax, samples = 4, 300
    acc = np.zeros((kmax + 1, N + 1))
    for _ in range(samples):
        Op = g[0].copy()
        for k in range(kmax + 1):
            U = gaussian_unitary(special_ortho_group.rvs(N, random_state=rng), g)
            Op = U @ Op @ U.conj().T
            w = np.abs(pauli_coeffs(Op, n).reshape(-1)) ** 2
            acc[k] += np.bincount(deg, weights=w, minlength=N + 1)
            Op = Ucz @ Op @ Ucz.conj().T
    acc /= samples
    p = np.zeros(N + 1)
    p[1] = 1
    for k in range(kmax + 1):
        assert np.abs(acc[k] - p).max() < 0.03, (k, acc[k], p)
        p = P @ p
    print(f"degree chain ok: simulated degree weights match P^k p0 (n={n}, {samples} Haar-Gaussian samples)")
    print("   chain p_1(k):", np.round([np.linalg.matrix_power(P, k)[1, 1] for k in range(5)], 4),
          "| (1-2/n)^k:", np.round([(1 - 2 / n) ** k for k in range(5)], 4))
    print("free_fermion_theory: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
