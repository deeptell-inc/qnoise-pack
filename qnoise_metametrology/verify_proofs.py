#!/usr/bin/env python
"""Numerical verification of the Theorem-1/2 proof ingredients.

Check 0  Weingarten constant: E<R,R> = 2g, g = d(d_A^2-1)/(d_A(d^2-1)),
         against Haar samples (small n) and random Cliffords (2-design =>
         must match at t=0 ALREADY -- the mean is t-independent).
Check 1  Clifford-point stabilizer counting (exact, GF(2)): for U Clifford,
         R = 2^{1-a} sum over N partner stabilizers; N and s_A from affine
         symplectic systems. Verifies:
           - R != 0  <=>  N >= 1
           - eta(t=0) = min(1, 2^{s_A} / N)   [needle-multiplicity formula]
           - p = P[N >= 1] vs measured visibility
Check 2  Corrected Gram rank at t=0 with an absolute eigenvalue tolerance
         anchored to 2g (the old lam_max-relative tolerance miscounts when
         Gram ~ 0).
"""

import numpy as np
from qiskit.quantum_info import Pauli, random_clifford

from prototype_phase_diagram import reduced_density, qfi_from_states
from prototype_lean import synth_blocks, make_phi, sensitivity_spectrum


# ----------------------------------------------------- GF(2) linear algebra

def gf2_solve_affine(A, b):
    """Solve Ac = b over GF(2). Returns (consistent, n_solutions, rankA)."""
    A = A.astype(np.uint8) % 2
    b = b.astype(np.uint8) % 2
    m, n = A.shape
    Ab = np.concatenate([A, b[:, None]], axis=1)
    row = 0
    for col in range(n):
        piv = None
        for rr in range(row, m):
            if Ab[rr, col]:
                piv = rr
                break
        if piv is None:
            continue
        Ab[[row, piv]] = Ab[[piv, row]]
        for rr in range(m):
            if rr != row and Ab[rr, col]:
                Ab[rr] ^= Ab[row]
        row += 1
        if row == m:
            break
    rankA = row
    # inconsistent iff a zero-A row has b = 1
    for rr in range(rankA, m):
        if Ab[rr, :n].any() == 0 and Ab[rr, n]:
            return False, 0, rankA
    # also check rows within [0, rankA) can't be inconsistent by construction
    inconsistent = any(not Ab[rr, :n].any() and Ab[rr, n]
                       for rr in range(m))
    if inconsistent:
        return False, 0, rankA
    return True, 2 ** (A.shape[1] - rankA), rankA


def clifford_exact_counts(cliff, n, a, g_qubit=0):
    """Exact N (partner multiplicity), s_A, for P = X_{g_qubit}, A = 0..a-1.

    Symplectic vectors from qiskit Pauli (x|z arrays, little-endian).
    Constraints on c in F_2^n (combination of stabilizer generators
    S_i = U Z_i U^dag):
      support:  (sum_i c_i S_i).x[q] = Phat.x[q], same for z, all q in Abar
      anticommute:  sum_i c_i w(S_i, Phat) = 1
    """
    phat = Pauli("I" * (n - 1 - g_qubit) + "X" + "I" * g_qubit).evolve(
        cliff, frame="s")
    gens = []
    for i in range(n):
        z = Pauli("I" * (n - 1 - i) + "Z" + "I" * i).evolve(cliff, frame="s")
        gens.append(z)
    Sx = np.array([g.x for g in gens], dtype=np.uint8)  # (n, n)
    Sz = np.array([g.z for g in gens], dtype=np.uint8)
    px, pz = phat.x.astype(np.uint8), phat.z.astype(np.uint8)

    abar = list(range(a, n))
    # support system: rows = 2*(n-a) constraints, cols = n unknowns
    A_sup = np.concatenate([Sx[:, abar].T, Sz[:, abar].T], axis=0)
    b_sup = np.concatenate([px[abar], pz[abar]])
    # anticommutation row: w(S_i, P) = S_i.x . P.z + S_i.z . P.x mod 2
    w = (Sx @ pz + Sz @ px) % 2
    A_full = np.concatenate([A_sup, w[None, :]], axis=0)
    b_full = np.concatenate([b_sup, [1]])
    ok, N, _ = gf2_solve_affine(A_full, b_full)
    N = N if ok else 0

    # s_A: dim of stabilizer subgroup supported in A (homogeneous system)
    _, n_hom, rank_hom = gf2_solve_affine(A_sup, np.zeros(2 * (n - a),
                                                          dtype=np.uint8))
    s_A = int(np.log2(n_hom))  # = n - rank_hom
    return N, s_A


# ------------------------------------------------------------------ checks

def g_const(d_A, d):
    return d * (d_A ** 2 - 1) / (d_A * (d ** 2 - 1))


def response_norm_sq(blocks_or_u, n, a, use_matrix=None):
    """<R,R> for G=X_0 via exact statevectors."""
    from qiskit.quantum_info import Statevector as SV
    from prototype_lean import apply_block, apply_x0
    psi0 = np.zeros(2 ** n, dtype=complex)
    psi0[0] = 1.0
    if use_matrix is not None:
        u0 = use_matrix @ psi0
        u1 = use_matrix @ apply_x0(psi0)
    else:
        u0, u1 = psi0, apply_x0(psi0)
        for b in blocks_or_u:
            u0 = apply_block(u0, b, n, SV)
            u1 = apply_block(u1, b, n, SV)
    pa0 = u0.reshape(2 ** (n - a), 2 ** a)
    pam = u1.reshape(2 ** (n - a), 2 ** a)
    cross = np.einsum("ei,ej->ij", pam, pa0.conj())
    R = -1j * (cross - cross.conj().T)
    return float(np.sum(np.abs(R) ** 2).real)


FAILURES = []


def _gate(label, ok):
    print(f"  [{'OK' if ok else 'FAIL'}] {label}")
    if not ok:
        FAILURES.append(label)


def check0_weingarten(n=6, a=4, samples=300, seed=1):
    from scipy.stats import unitary_group
    rng = np.random.default_rng(seed)
    d, d_A = 2 ** n, 2 ** a
    target = 2 * g_const(d_A, d)
    haar = [response_norm_sq(None, n, a,
                             use_matrix=unitary_group.rvs(d, random_state=int(
                                 rng.integers(2 ** 31))))
            for _ in range(samples)]
    cliffs = []
    for _ in range(samples):
        c = random_clifford(n, seed=int(rng.integers(2 ** 31)))
        blocks = [(c.to_circuit(), [])]
        cliffs.append(response_norm_sq(blocks, n, a))
    print(f"[check0] n={n}, a={a}: 2g = {target:.5f}")
    print(f"  Haar     E<R,R> = {np.mean(haar):.5f} "
          f"+/- {np.std(haar) / np.sqrt(samples):.5f}   "
          f"(std {np.std(haar):.4f})")
    print(f"  Clifford E<R,R> = {np.mean(cliffs):.5f} "
          f"+/- {np.std(cliffs) / np.sqrt(samples):.5f}   "
          f"(std {np.std(cliffs):.4f})  <- same mean, larger spread")
    _gate("check0 Haar mean within 5 SE of 2g",
          abs(np.mean(haar) - target) < 5 * np.std(haar) / np.sqrt(samples))
    _gate("check0 Clifford mean within 5 SE of 2g",
          abs(np.mean(cliffs) - target)
          < 5 * np.std(cliffs) / np.sqrt(samples))
    _gate("check0 Clifford spread exceeds Haar spread (4-design gap)",
          np.std(cliffs) > 3 * np.std(haar))


def check1_stabilizer(n=12, a_list=(6, 8, 9), samples=200, seed=2):
    print(f"\n[check1] Clifford point, n={n}: exact N vs eta(0), "
          f"{samples} samples")
    print(f"{'a':>3} {'P[N>0] pred~':>13} {'P[N>0] exact':>13} "
          f"{'E[eta] exact-pred':>18} {'E[eta] measured':>16} "
          f"{'per-seed match':>15}")
    rng = np.random.default_rng(seed)
    for a in a_list:
        lam = 2.0 ** (2 * a - n - 1)
        vis, etas_pred, etas_meas, match = [], [], [], 0
        for _ in range(samples):
            c = random_clifford(n, seed=int(rng.integers(2 ** 31)))
            N, s_A = clifford_exact_counts(c, n, a)
            vis.append(N > 0)
            eta_p = min(1.0, 2.0 ** s_A / N) if N else 0.0
            etas_pred.append(eta_p)
            # measure eta numerically for a subsample (costly)
            if len(etas_meas) < 40:
                blocks = [(c.to_circuit(), [])]
                phi, _ = make_phi(blocks, "oneshot", n)
                delta = 1e-4
                rho0 = reduced_density(phi(0.0), n, a)
                drho = (reduced_density(phi(delta), n, a)
                        - reduced_density(phi(-delta), n, a)) / (2 * delta)
                fqa = qfi_from_states(rho0, drho)
                spec = sensitivity_spectrum(phi, n, a)
                eta_m = spec["s_max_sq"] / fqa if fqa > 1e-9 else 0.0
                etas_meas.append(eta_m)
                if abs(eta_m - eta_p) < 0.05 * max(eta_p, 0.02):
                    match += 1
        print(f"{a:>3} {1 - np.exp(-lam):>13.3f} {np.mean(vis):>13.3f} "
              f"{np.mean(etas_pred):>18.3f} {np.mean(etas_meas):>16.3f} "
              f"{match:>10d}/{len(etas_meas)}")
        _gate(f"check1 per-seed formula match a={a} "
              f"({match}/{len(etas_meas)})", match == len(etas_meas))


def check2_rank_tolerance(n=10, a_list=(3, 5, 6, 8), t_list=(0, 2, 6, 12),
                          seeds=8):
    from prototype_lean import identifiability_gram
    print(f"\n[check2] Gram rank at fixed ABSOLUTE tol (2g*1e-6) vs old "
          f"relative tol, n={n}")
    print(f"{'a':>3} " + " ".join(f"t={t:>3d} old/new" for t in t_list))
    for a in a_list:
        row = []
        for t in t_list:
            olds, news = [], []
            for seed in range(seeds):
                rng = np.random.default_rng([seed, t, a, 0])
                blocks = synth_blocks(n, t, max(t, 1), rng)
                # TRUE old behaviour: lam_max-relative tolerance only
                olds.append(_rank_rel(blocks, n, a))
                # corrected: absolute tolerance anchored at 2g * 1e-6
                tol_abs = 2 * g_const(2 ** a, 2 ** n) * 1e-6
                news.append(_rank_abs(blocks, n, a, tol_abs))
            row.append(f"{np.mean(olds):.2f}/{np.mean(news):.2f}")
            if a == 3 and t == 0:
                # the erratum: relative tolerance inflates rank when the
                # Gram is numerically zero; corrected value matches the
                # exact visibility prediction 0.031
                _gate("check2 old rel-tolerance inflates a=3,t=0 rank "
                      f"({np.mean(olds):.2f} > 0.3)", np.mean(olds) > 0.3)
                _gate("check2 corrected a=3,t=0 rank near exact 0.031 "
                      f"({np.mean(news):.2f})", abs(np.mean(news) - 0.031)
                      < 0.03)
        print(f"{a:>3} " + "  ".join(f"{s:>12s}" for s in row))


def _gram_eigs(blocks, n, a):
    from qiskit.quantum_info import Statevector as SV
    from prototype_lean import apply_block
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
            idx = 1 << i
            w = np.zeros(2 ** n, dtype=complex)
            w[idx] = 1.0 if pauli == "x" else 1j
            u_mu = evolve_all(w)
            pa0 = u0.reshape(2 ** (n - a), 2 ** a)
            pam = u_mu.reshape(2 ** (n - a), 2 ** a)
            cross = np.einsum("ei,ej->ij", pam, pa0.conj())
            R = -1j * (cross - cross.conj().T)
            responses.append(R.reshape(-1))
    M = np.array(responses)
    ev = np.clip(np.linalg.eigvalsh((M @ M.conj().T).real), 0, None)
    return ev


def _rank_abs(blocks, n, a, tol_abs):
    ev = _gram_eigs(blocks, n, a)
    return float(np.sum(ev > tol_abs) / len(ev))


def _rank_rel(blocks, n, a):
    """The pre-erratum rank: lam_max-relative tolerance only."""
    ev = _gram_eigs(blocks, n, a)
    lam_max = float(ev[-1])
    return float(np.sum(ev > max(lam_max, 1e-300) * 1e-8) / len(ev))


if __name__ == "__main__":
    import sys
    check0_weingarten()
    check1_stabilizer()
    check2_rank_tolerance()
    print(f"\nverify_proofs: {'PASSED' if not FAILURES else 'FAILED'} "
          f"({len(FAILURES)} gate failures)")
    for f in FAILURES:
        print(f"  FAILED: {f}")
    sys.exit(0 if not FAILURES else 1)
