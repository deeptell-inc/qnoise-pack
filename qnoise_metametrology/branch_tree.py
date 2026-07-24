#!/usr/bin/env python
"""Heisenberg branch-tree simulator: Lemma 4 at n = 100+.

Tracks the Pauli branches of G_hat = U G U^dag through a t-doped Clifford
circuit purely in the symplectic (GF(2)) representation -- no statevector,
so n is limited only by branch count (2^t), not Hilbert-space dimension.

Per realization: after each T gate, records the Pauli-weight purity
Pi_t = sum_b w_b^2 (w_b = c_b^2 = 2^{-#splits}), the branch count, and the
collision count (distinct branches landing on the same Pauli string; exact
Pi requires sign-aware merging only when collisions exist -- at n >= 100
they are absent, which the simulator verifies rather than assumes).

Validates Lemma 4 [E(Pi_t) = (3/4)^t] with REAL circuits: actual Clifford
layers (not the idealized twirl) and actual collision statistics.

Unit test (--test): n = 6, dense-matrix cross-check of Pi per realization.

Usage:
  python branch_tree.py --test
  python branch_tree.py --n 100 --tmax 20 --seeds 8
"""

import argparse
import json
import time

import numpy as np
from qiskit.quantum_info import Pauli, random_clifford


def clifford_symplectic_image(cliff, n):
    """M (2n x 2n, uint8): row-vector (x|z) of P -> (x|z) of C P C^dag.

    Built directly from Pauli.evolve(frame='s') on the X_i / Z_i basis, so
    it is correct by construction regardless of tableau conventions.
    """
    M = np.zeros((2 * n, 2 * n), dtype=np.uint8)
    for i in range(n):
        for base, row in (("X", i), ("Z", n + i)):
            label = "I" * (n - 1 - i) + base + "I" * i
            img = Pauli(label).evolve(cliff, frame="s")
            M[row, :n] = img.x
            M[row, n:] = img.z
    return M


def evolve_branches(B, M):
    """B (K x 2n uint8) @ M mod 2, via float32 BLAS."""
    return (B.astype(np.float32) @ M.astype(np.float32)).astype(np.int64) \
        .astype(np.uint8) % 2


def collision_count(B):
    packed = np.packbits(B, axis=1)
    uniq = np.unique(packed, axis=0)
    return B.shape[0] - uniq.shape[0]


def run_realization(n, tmax, seed, check_every=(), g_qubit=0):
    rng = np.random.default_rng([seed, n, 4242])
    # initial branch: G = X_{g_qubit}
    B = np.zeros((1, 2 * n), dtype=np.uint8)
    B[0, g_qubit] = 1
    w = np.array([1.0], dtype=np.float64)

    gates = []  # (cliff_seed, t_qubit) record for the dense cross-check
    traj = []
    for k in range(tmax + 1):
        cseed = int(rng.integers(2 ** 31))
        cliff = random_clifford(n, seed=cseed)
        M = clifford_symplectic_image(cliff, n)
        B = evolve_branches(B, M)
        if k == tmax:
            gates.append((cseed, None))
            break
        q = int(rng.integers(n))
        gates.append((cseed, q))
        # T on qubit q: branches with x_q = 1 split (partner: z_q toggled)
        mask = B[:, q] == 1
        if mask.any():
            twin = B[mask].copy()
            twin[:, n + q] ^= 1
            B = np.concatenate([B, twin], axis=0)
            w = np.concatenate([w * np.where(mask, 0.5, 1.0), w[mask] * 0.5])
        pi = float(np.sum(w ** 2))
        col = collision_count(B) if (k + 1) in check_every else -1
        traj.append(dict(t=k + 1, purity=pi, branches=int(B.shape[0]),
                         collisions=col))
    return traj, gates


# ------------------------------------------------------------- dense check

def dense_purity(n, gates, g_qubit=0):
    """Exact Pi via dense matrices + Pauli transform (n <= 8)."""
    from qiskit.quantum_info import Operator
    from prototype_phase_diagram import pauli_coeffs, op_on_qubit, PAULIS
    U = np.eye(2 ** n, dtype=complex)
    for cseed, q in gates:
        C = Operator(random_clifford(n, seed=cseed)).data
        U = C @ U
        if q is not None:
            phase = np.exp(1j * np.pi / 4 * ((np.arange(2 ** n) >> q) & 1))
            U = phase[:, None] * U
    G = op_on_qubit(PAULIS[1], g_qubit, n)
    Ghat = U @ G @ U.conj().T
    c = pauli_coeffs(Ghat, n).reshape(-1)
    wts = np.abs(c) ** 2
    wts[0] = 0.0
    p = wts / wts.sum()
    return float(np.sum(p ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tmax", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--out", default="branchtree")
    args = ap.parse_args()

    if args.test:
        print("unit test: branch tree vs dense matrices, n=6, tmax=6")
        for seed in range(3):
            traj, gates = run_realization(6, 6, seed, check_every=set())
            pi_branch = traj[-1]["purity"]
            pi_dense = dense_purity(6, gates)
            ncol = collision_count  # noqa: F841 (visibility)
            status = "OK" if abs(pi_branch - pi_dense) < 1e-9 else \
                f"MISMATCH (collisions likely: branch={pi_branch:.6f} " \
                f"dense={pi_dense:.6f})"
            print(f"  seed {seed}: branch Pi={pi_branch:.6f}  "
                  f"dense Pi={pi_dense:.6f}  -> {status}")
        return

    checkpoints = {args.tmax // 2, args.tmax}
    all_traj, t0 = [], time.time()
    for seed in range(args.seeds):
        traj, _ = run_realization(args.n, args.tmax, seed,
                                  check_every=checkpoints)
        all_traj.append(traj)
        print(f"seed {seed}: t={args.tmax}, branches="
              f"{traj[-1]['branches']}, Pi={traj[-1]['purity']:.3e}, "
              f"collisions@checkpoints="
              f"{[x['collisions'] for x in traj if x['collisions'] >= 0]}, "
              f"{time.time() - t0:.0f}s", flush=True)

    tag = f"{args.out}_n{args.n}"
    with open(f"{tag}.json", "w") as f:
        json.dump(all_traj, f, indent=1)
    print(f"-> {tag}.json\n")
    print(f"=== n={args.n}: E[Pi_t] vs (3/4)^t ===")
    print(f"{'t':>3} {'mean Pi':>12} {'(3/4)^t':>12} {'ratio':>8}")
    for k in range(1, args.tmax + 1):
        pis = [tr[k - 1]["purity"] for tr in all_traj]
        th = 0.75 ** k
        print(f"{k:>3} {np.mean(pis):>12.4e} {th:>12.4e} "
              f"{np.mean(pis) / th:>8.3f}")
    # fitted rate
    ks = np.arange(2, args.tmax + 1)
    logs = [np.log(np.mean([tr[k - 1]['purity'] for tr in all_traj]))
            for k in ks]
    rate = -np.polyfit(ks, logs, 1)[0]
    print(f"\nfitted decay rate: {rate:.4f}  vs  ln(4/3) = {np.log(4/3):.4f}")


if __name__ == "__main__":
    main()
