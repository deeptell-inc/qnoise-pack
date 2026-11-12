#!/usr/bin/env python
"""Readout ceiling across scrambler families (decoding-limits map, column C).

Three experiments, all measuring the same observables as the doped-Clifford
study (available information F_Q^A/F_Q, best-Pauli fraction eta_needle,
effective number of settings m_eff, Gram rank of the {X_i, Y_i} dictionary):

  brickwork : 1D brickwork circuits of depth D, Haar or Clifford 2-qubit
              gates, subsystem A next to ("near") or away from ("far") the
              signal qubit.
  u1        : 1D brickwork of U(1)-symmetric Haar gates vs generic Haar
              gates, both on the Neel state.
  matchgate : free-fermion (matchgate) blocks interleaved with k CZ gates
              (non-Gaussian doping), on the vacuum.

The response is computed exactly: for G^2 = 1 and a basis state |psi0>,
d/dtheta U e^{-i theta G}|psi0> at theta=0 is -i U G|psi0>, so
  d rho_A = -i Tr_env( |uG><u0| - |u0><uG| ),  u0 = U|psi0>, uG = U G|psi0>.

Conventions follow prototype_phase_diagram.py: qubit q is bit q of the basis
index (qubit 0 = least significant bit).

Usage:
  python scrambler_zoo.py --selftest
  python scrambler_zoo.py --exp brickwork --n 10 --seeds 16 --out zoo_brickwork_n10.json
"""

import argparse
import json
import time

import numpy as np
from scipy.stats import unitary_group

from prototype_phase_diagram import pauli_coeffs, qfi_from_states

CZ = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)


# ------------------------------------------------------------------ gates

def haar2(rng):
    return unitary_group.rvs(4, random_state=rng)


def clifford2(rng):
    from qiskit.quantum_info import random_clifford
    return random_clifford(2, seed=int(rng.integers(2 ** 31))).to_matrix()


def u1_gate(rng):
    """Haar-random 2-qubit gate commuting with Z_1 + Z_2.
    Local basis index = 2*b_q1 + b_q2: phases on |00>, |11>, U(2) on
    span{|01>, |10>}."""
    U = np.zeros((4, 4), complex)
    U[0, 0] = np.exp(2j * np.pi * rng.random())
    U[3, 3] = np.exp(2j * np.pi * rng.random())
    U[1:3, 1:3] = unitary_group.rvs(2, random_state=rng)
    return U


def matchgate(rng):
    """Random matchgate G(A, B): A on span{|00>, |11>}, B on span{|01>, |10>},
    det A = det B (nearest-neighbour matchgates are fermionic Gaussian)."""
    A = unitary_group.rvs(2, random_state=rng)
    B = unitary_group.rvs(2, random_state=rng)
    B = B * np.sqrt(np.linalg.det(A) / np.linalg.det(B))
    U = np.zeros((4, 4), complex)
    U[np.ix_([0, 3], [0, 3])] = A
    U[np.ix_([1, 2], [1, 2])] = B
    return U


GATES = {"haar": haar2, "clifford": clifford2, "u1": u1_gate,
         "matchgate": matchgate}


# ------------------------------------------------------- state evolution

def apply2(S, U, q1, q2, n):
    """Apply 4x4 U (index 2*b_q1 + b_q2) to a batch S of shape (B, 2^n)."""
    B = S.shape[0]
    T = S.reshape((B,) + (2,) * n)
    a1, a2 = 1 + (n - 1 - q1), 1 + (n - 1 - q2)
    T = np.moveaxis(T, (a1, a2), (-2, -1))
    sh = T.shape
    T = (T.reshape(-1, 4) @ U.T).reshape(sh)
    return np.moveaxis(T, (-2, -1), (a1, a2)).reshape(B, -1)


def brick_layer(S, layer, n, gate, rng):
    for i in range(layer % 2, n - 1, 2):
        S = apply2(S, gate(rng), i, i + 1, n)
    return S


def initial_batch(n, bits):
    """Rows: |b>, then G_mu|b> for mu = X_0, Y_0, X_1, Y_1, ...
    (X_i|b> = |b^e_i>, Y_i|b> = i(-1)^{b_i}|b^e_i>)."""
    b = sum(int(v) << q for q, v in enumerate(bits))
    S = np.zeros((2 * n + 1, 2 ** n), complex)
    S[0, b] = 1.0
    for i in range(n):
        S[1 + 2 * i, b ^ (1 << i)] = 1.0
        S[2 + 2 * i, b ^ (1 << i)] = 1j * (-1) ** int(bits[i])
    return S


# ------------------------------------------------------------ observables

def g_const(n, a):
    d, dA = 2.0 ** n, 2.0 ** a
    return d * (dA ** 2 - 1) / (dA * (d ** 2 - 1))


def to_A(S, A, n):
    """(B, 2^n) -> (B, 2^(n-a), 2^a) with the qubits of A as low bits."""
    B = S.shape[0]
    T = S.reshape((B,) + (2,) * n)
    rest = [q for q in range(n) if q not in A]
    order = ([0] + [1 + (n - 1 - q) for q in rest[::-1]]
             + [1 + (n - 1 - q) for q in list(A)[::-1]])
    return np.transpose(T, order).reshape(B, 2 ** (n - len(A)), 2 ** len(A))


def responses(S, A, n):
    """rho_A and the exact responses d rho_A for every dictionary element."""
    M = to_A(S, A, n)
    p0 = M[0]
    rho = np.einsum("ei,ej->ij", p0, p0.conj())
    cross = np.einsum("mei,ej->mij", M[1:], p0.conj())
    R = -1j * (cross - np.conj(np.transpose(cross, (0, 2, 1))))
    return rho, R


def observe(S, A, n, gram=True):
    a = len(A)
    rho, R = responses(S, A, n)
    d = R[0]                                   # signal generator G = X_0
    fqa = qfi_from_states(rho, d)
    s = (pauli_coeffs(d, a).reshape(-1).real * 2 ** a)[1:]
    s2 = s ** 2
    tot = float(s2.sum())
    rec = dict(fqa=fqa, fq_full=4.0,
               eta_needle=float(s2.max() / fqa) if fqa > 1e-12 else 0.0,
               s_sum_sq=tot,
               m_eff=float(tot ** 2 / (s2 ** 2).sum()) if tot > 1e-12 else 0.0,
               n_active=int((s2 > 1e-10 * max(tot, 1e-300)).sum()))
    if gram:
        V = R.reshape(R.shape[0], -1)
        ev = np.linalg.eigvalsh((V @ V.conj().T).real)
        rec.update(gram_rank=int((ev > 2 * g_const(n, a) * 1e-6).sum()),
                   gram_m=int(len(ev)), gram_lam_min=float(ev[0]),
                   gram_lam_max=float(ev[-1]))
    return rec


# ------------------------------------------------------------ experiments

def subsystems(n, a):
    return {"near": list(range(a)), "far": list(range(n - a, n))}


def exp_brickwork(n, seeds, a_list, depths):
    recs = []
    for gname in ("haar", "clifford"):
        for seed in range(seeds):
            rng = np.random.default_rng([seed, n, 11, gname == "haar"])
            S = initial_batch(n, [0] * n)
            for D in range(max(depths) + 1):
                if D in depths:
                    for a in a_list:
                        for place, A in subsystems(n, a).items():
                            recs.append(dict(exp="brickwork", gate=gname, n=n,
                                             a=a, place=place, depth=D,
                                             seed=seed, **observe(S, A, n)))
                S = brick_layer(S, D, n, GATES[gname], rng)
    return recs


def exp_u1(n, seeds, a_list, depths):
    recs = []
    neel = [q % 2 for q in range(n)]
    for gname in ("u1", "haar"):
        for seed in range(seeds):
            rng = np.random.default_rng([seed, n, 22, gname == "u1"])
            S = initial_batch(n, neel)
            for D in range(max(depths) + 1):
                if D in depths:
                    for a in a_list:
                        for place, A in subsystems(n, a).items():
                            recs.append(dict(exp="u1", gate=gname, n=n, a=a,
                                             place=place, depth=D, seed=seed,
                                             **observe(S, A, n)))
                S = brick_layer(S, D, n, GATES[gname], rng)
    return recs


def exp_matchgate(n, seeds, a_list, k_list, block_depth=None):
    """Gaussian block (brickwork of matchgates) -> record -> CZ -> block ..."""
    block_depth = block_depth or 2 * n
    recs = []
    for seed in range(seeds):
        rng = np.random.default_rng([seed, n, 33])
        S = initial_batch(n, [0] * n)
        layer = 0
        for k in range(max(k_list) + 1):
            if k > 0:
                i = int(rng.integers(n - 1))
                S = apply2(S, CZ, i, i + 1, n)
            for _ in range(block_depth):
                S = brick_layer(S, layer, n, matchgate, rng)
                layer += 1
            if k in k_list:
                for a in a_list:
                    recs.append(dict(exp="matchgate", gate="matchgate+CZ", n=n,
                                     a=a, place="near", k=k, seed=seed,
                                     **observe(S, list(range(a)), n)))
    return recs


def exp_haar_reference(n, seeds, a_list):
    """Global Haar unitary: the deep-circuit limit of every family."""
    recs = []
    for seed in range(seeds):
        rng = np.random.default_rng([seed, n, 44])
        U = unitary_group.rvs(2 ** n, random_state=rng)
        S = initial_batch(n, [0] * n) @ U.T
        for a in a_list:
            recs.append(dict(exp="haar_global", gate="haar_global", n=n, a=a,
                             place="near", seed=seed,
                             **observe(S, list(range(a)), n)))
    return recs


# ---------------------------------------------------------------- selftest

def selftest():
    rng = np.random.default_rng(0)
    # 1. exact response equals a finite difference of rho_A(theta)
    n, A = 6, [0, 2, 3]
    S = initial_batch(n, [1, 0, 1, 0, 0, 1])
    for D in range(5):
        S = brick_layer(S, D, n, haar2, rng)
    _, R = responses(S, A, n)
    th = 1e-5
    def rho_th(t):
        v = (np.cos(t) * S[0] - 1j * np.sin(t) * S[1])[None]
        m = to_A(v, A, n)[0]
        return np.einsum("ei,ej->ij", m, m.conj())
    fd = (rho_th(th) - rho_th(-th)) / (2 * th)
    assert np.abs(fd - R[0]).max() < 1e-8, np.abs(fd - R[0]).max()

    # 2. Lemma 2: E<R,R> = 2g for global Haar unitaries
    n, a = 6, 4
    vals = []
    for _ in range(300):
        U = unitary_group.rvs(2 ** n, random_state=rng)
        _, R = responses(initial_batch(n, [0] * n) @ U.T, list(range(a)), n)
        vals.append(float(np.sum(np.abs(R[0]) ** 2)))
    se = np.std(vals) / np.sqrt(len(vals))
    assert abs(np.mean(vals) - 2 * g_const(n, a)) < 5 * se, (np.mean(vals), 2 * g_const(n, a))

    # 3. U(1) gates conserve total charge; matchgates conserve parity
    n = 6
    S = initial_batch(n, [0, 1, 0, 1, 0, 1])[:1]
    for D in range(8):
        S = brick_layer(S, D, n, u1_gate, rng)
    pops = np.abs(S[0]) ** 2
    charge = np.array([bin(i).count("1") for i in range(2 ** n)])
    assert abs(pops[charge == 3].sum() - 1) < 1e-10
    S = initial_batch(n, [0] * n)[:1]
    for D in range(8):
        S = brick_layer(S, D, n, matchgate, rng)
    assert abs((np.abs(S[0]) ** 2)[charge % 2 == 0].sum() - 1) < 1e-10

    # 4. matchgate circuits are Gaussian: U X_0 U^dag stays a sum of at most
    #    2n Majorana strings (dense check, n = 4)
    n = 4
    cols = np.eye(2 ** n, dtype=complex)
    for D in range(6):
        cols = brick_layer(cols, D, n, matchgate, rng)
    U = cols.T
    X0 = np.zeros((2 ** n, 2 ** n), complex)
    for i in range(2 ** n):
        X0[i ^ 1, i] = 1.0
    c = np.abs(pauli_coeffs(U @ X0 @ U.conj().T, n).reshape(-1)) ** 2
    assert int((c > 1e-12).sum()) <= 2 * n, int((c > 1e-12).sum())

    # 5. Clifford brickwork: every nonzero Pauli sensitivity is +-2
    n = 8
    S = initial_batch(n, [0] * n)
    for D in range(12):
        S = brick_layer(S, D, n, clifford2, rng)
    _, R = responses(S, list(range(6)), n)
    s = pauli_coeffs(R[0], 6).reshape(-1).real * 2 ** 6
    nz = np.abs(s[np.abs(s) > 1e-9])
    assert np.allclose(nz, 2.0), nz[:5]
    print("selftest PASSED (5 checks)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--exp", choices=["brickwork", "u1", "matchgate", "haar"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--a", type=int, nargs="*")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    n = args.n
    a_list = args.a or sorted({n // 2, round(0.6 * n), round(0.8 * n)})
    t0 = time.time()
    if args.exp == "brickwork":
        recs = exp_brickwork(n, args.seeds, a_list,
                             [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32])
    elif args.exp == "u1":
        recs = exp_u1(n, args.seeds, a_list,
                      [0, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128])
    elif args.exp == "matchgate":
        recs = exp_matchgate(n, args.seeds, a_list,
                             [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    else:
        recs = exp_haar_reference(n, args.seeds, a_list)
    out = args.out or f"zoo_{args.exp}_n{n}.json"
    json.dump(recs, open(out, "w"), indent=1)
    print(f"{len(recs)} records -> {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
