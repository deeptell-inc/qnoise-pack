#!/usr/bin/env python
"""Learnability phase diagram prototype for scrambled-signal meta-metrology.

Sweeps (t, f) = (magic / number of T gates, access fraction |A|/n) for a
t-doped Clifford scrambler hiding a signal theta encoded by generator G.

Per cell, computes:
  - F_Q^A        : quantum Fisher information of the reduced state family
                   (information-theoretic resource; axis f)
  - Pauli spread : participation entropy of the effective generator
                   G_hat = (d/dtheta U_theta) U^dag, global and A-restricted
                   (complexity resource; axis t)
  - QELM readout : ridge regression on ALL Pauli expectations on A with shot
                   noise -> test MSE, Fisher efficiency eta = CRB / MSE,
                   condition number kappa of the feature matrix
                   (operational learnability)

Money-plot prediction (Region III): cells with high F_Q^A but eta collapsing
as t grows at fixed f.

Encoding modes:
  oneshot     |Phi> = U_scr e^{-i theta G} |0...0>   (Hayden-Preskill-like;
              expect a sharp f=1/2 no-cloning wall in F_Q^A)
  interleaved e^{-i theta G} interleaved between L scrambling blocks
              (chaos-metrology-like; signal re-imprinted, small-f survival)

Usage:
  python prototype_phase_diagram.py --mode both --n 8 --seeds 8
  python prototype_phase_diagram.py --smoke
"""

import argparse
import json
import time
from itertools import product as iproduct

import numpy as np
from qiskit.quantum_info import Operator, random_clifford

# Pauli matrices; qubit 0 = least significant bit (qiskit little-endian).
PAULIS = [
    np.eye(2, dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]


# ---------------------------------------------------------------- utilities

def op_on_qubit(sigma, q, n):
    """Embed a single-qubit operator on qubit q (LSB convention)."""
    return np.kron(np.kron(np.eye(2 ** (n - 1 - q)), sigma), np.eye(2 ** q))


def pauli_coeffs(mat, n):
    """All Pauli coefficients c_P = Tr(mat P)/2^n, shape (4,)*n flattened.

    Axis order [p_{n-1}, ..., p_0] (leftmost = highest qubit), matching
    qiskit label strings. Cost O(n 4^n).
    """
    # Interleave row/col axes per qubit: [r_{n-1}, c_{n-1}, r_{n-2}, ...]
    T = mat.reshape((2,) * (2 * n))
    perm = [ax for q in range(n) for ax in (q, n + q)]
    T = np.transpose(T, perm).reshape((4,) * n)  # combined index m = 2r + c
    # VT[m=2r+c, p] = P_p[c, r]  so contraction over m gives Tr(M P) per qubit
    VT = np.zeros((4, 4), dtype=complex)
    for p in range(4):
        for r in range(2):
            for c in range(2):
                VT[2 * r + c, p] = PAULIS[p][c, r]
    cur = T
    for _ in range(n):
        cur = np.tensordot(cur, VT, axes=([0], [0]))
    return cur / 2 ** n  # axes [p_{n-1}, ..., p_0]


def reduced_density(state, n, a):
    """rho_A for A = qubits 0..a-1 (LSB block)."""
    psi = state.reshape(2 ** (n - a), 2 ** a)  # env = high bits
    return np.einsum("ei,ej->ij", psi, psi.conj())


def qfi_from_states(rho, drho, tol=1e-10):
    """QFI = 2 sum |<i|drho|j>|^2 / (l_i + l_j)."""
    lam, vec = np.linalg.eigh(rho)
    d = vec.conj().T @ drho @ vec
    denom = lam[:, None] + lam[None, :]
    mask = denom > tol
    return float(2.0 * np.sum(np.abs(d[mask]) ** 2 / denom[mask]))


# ------------------------------------------------------------ circuit model

def build_layers(n, t, layers, rng):
    """Clifford blocks with t T gates distributed between them.

    Returns list of 2^n unitaries: [B_0, B_1, ..., B_{layers}] where the
    circuit (without encoding) is B_layers ... B_1 B_0 and each B_k already
    includes its share of interleaved T gates.
    """
    # Split t T-gates as evenly as possible across `layers` insertion slots.
    tk = [t // layers + (1 if i < t % layers else 0) for i in range(layers)]
    blocks = []
    for k in range(layers + 1):
        C = Operator(random_clifford(n, seed=int(rng.integers(2 ** 31)))).data
        if k < layers:
            for _ in range(tk[k]):
                q = int(rng.integers(n))
                phase = np.exp(1j * np.pi / 4 * ((np.arange(2 ** n) >> q) & 1))
                C = phase[:, None] * C  # T_q applied after C
        blocks.append(C)
    return blocks


def encoder_matrix(theta, G_mat, n):
    """e^{-i theta G} for G with G^2 acting simply; general via eigh cache."""
    # G is Hermitian; use eigendecomposition (cached by caller if needed).
    lam, vec = encoder_matrix._cache
    return (vec * np.exp(-1j * theta * lam)) @ vec.conj().T


def make_state_fn(blocks, G_mat, mode, n):
    """Returns Phi(theta) -> statevector, and the effective generator matrix."""
    lam, vec = np.linalg.eigh(G_mat)
    psi0 = np.zeros(2 ** n, dtype=complex)
    psi0[0] = 1.0

    def enc(theta):
        return (vec * np.exp(-1j * theta * lam)) @ vec.conj().T

    if mode == "oneshot":
        U_scr = blocks[0]
        for B in blocks[1:]:
            U_scr = B @ U_scr

        def phi(theta):
            return U_scr @ (enc(theta) @ psi0)

        G_eff = U_scr @ G_mat @ U_scr.conj().T
    else:  # interleaved: B_L E B_{L-1} E ... B_1 E B_0
        def u_theta(theta):
            E = enc(theta)
            U = blocks[0]
            for B in blocks[1:]:
                U = B @ (E @ U)
            return U

        def phi(theta):
            E = enc(theta)
            v = blocks[0] @ psi0
            for B in blocks[1:]:
                v = B @ (E @ v)
            return v

        delta = 1e-5
        dU = (u_theta(delta) - u_theta(-delta)) / (2 * delta)
        U0 = u_theta(0.0)
        G_eff = 1j * dU @ U0.conj().T
        G_eff = 0.5 * (G_eff + G_eff.conj().T)
    return phi, G_eff


# ------------------------------------------------------------------ metrics

def generator_spread_metrics(G_eff, n, a):
    """Pauli-space structure of the effective generator."""
    c = pauli_coeffs(G_eff, n)
    w = np.abs(c.reshape(-1)) ** 2
    w[0] = 0.0  # drop identity component (gauge)
    tot = w.sum()
    if tot < 1e-14:
        return dict(pn_global=0.0, s2_a=0.0, vis_weight=0.0)
    pi = w / tot
    pn_global = float(1.0 / np.sum(pi ** 2))  # participation number
    # A = qubits 0..a-1 = last a Pauli axes -> reshape (env, A)
    pi2 = pi.reshape(4 ** (n - a), 4 ** a)
    marg = pi2.sum(axis=0)
    s2_a = float(-np.log2(np.sum(marg ** 2) + 1e-300))
    vis_weight = float(1.0 - pi2[:, 0].sum())  # weight with P_A != I
    return dict(pn_global=pn_global, s2_a=s2_a, vis_weight=vis_weight)


def all_pauli_features(rho_a, a):
    """Expectations of all 4^a - 1 nontrivial Paulis on A. O(a 4^a)."""
    e = pauli_coeffs(rho_a, a).reshape(-1).real * 2 ** a
    return e[1:]  # drop identity


def qelm_readout(phi, n, a, rng, budget=131072, theta_max=0.2,
                 n_train=400, n_test=200):
    """Ridge regression theta-estimator on noisy all-Pauli features on A.

    Fixed TOTAL measurement budget per theta-sample, split uniformly over
    the 4^a - 1 feature settings (non-commuting Paulis cannot share shots).
    This is where operator spreading bites: a t=0 Clifford needle
    concentrates all sensitivity in one setting, while magic spreads it over
    exponentially many settings, each starved of shots.
    """
    n_feat = 4 ** a - 1
    shots_eff = max(budget / n_feat, 1.0)
    th_train = rng.uniform(-theta_max, theta_max, n_train)
    th_test = rng.uniform(-theta_max, theta_max, n_test)

    def feats(thetas):
        X = np.empty((len(thetas), n_feat))
        for i, th in enumerate(thetas):
            X[i] = all_pauli_features(reduced_density(phi(th), n, a), a)
        return X

    X_tr_clean, X_te_clean = feats(th_train), feats(th_test)
    kappa_sv = np.linalg.svd(
        X_tr_clean - X_tr_clean.mean(0), compute_uv=False)
    kappa = float(kappa_sv[0] / max(kappa_sv[min(len(kappa_sv), n_train) - 1],
                                    1e-300))

    # needle metric: sensitivity of the single best Pauli setting
    dth = 1e-3
    f_p = all_pauli_features(reduced_density(phi(dth), n, a), a)
    f_m = all_pauli_features(reduced_density(phi(-dth), n, a), a)
    s = (f_p - f_m) / (2 * dth)
    s_max_sq = float(np.max(s ** 2))
    s_sum_sq = float(np.sum(s ** 2))

    def add_noise(X):
        sig = np.sqrt(np.clip(1.0 - X ** 2, 0.0, 1.0) / shots_eff)
        return X + rng.normal(0.0, 1.0, X.shape) * sig

    X_tr, X_te = add_noise(X_tr_clean), add_noise(X_te_clean)
    # center
    mu, ybar = X_tr.mean(0), th_train.mean()
    Xc, yc = X_tr - mu, th_train - ybar
    # dual ridge with validation split for lambda
    n_val = n_train // 4
    Xv, yv, Xf, yf = Xc[:n_val], yc[:n_val], Xc[n_val:], yc[n_val:]
    Kf = Xf @ Xf.T
    scale = np.trace(Kf) / len(Kf)
    best = (np.inf, None)
    for lam in [1e-6, 1e-4, 1e-2, 1e-1, 1.0]:
        alpha = np.linalg.solve(Kf + lam * scale * np.eye(len(Kf)), yf)
        pred_v = (Xv @ Xf.T) @ alpha
        mse_v = float(np.mean((pred_v - yv) ** 2))
        if mse_v < best[0]:
            best = (mse_v, lam)
    lam = best[1]
    K = Xc @ Xc.T
    alpha = np.linalg.solve(K + lam * (np.trace(K) / len(K)) * np.eye(len(K)),
                            yc)
    pred = (X_te - mu) @ Xc.T @ alpha + ybar
    mse = float(np.mean((pred - th_test) ** 2))
    return dict(mse=mse, kappa=kappa, lam=lam,
                mse_prior=float(np.var(th_test)),
                s_max_sq=s_max_sq, s_sum_sq=s_sum_sq,
                shots_eff=shots_eff, budget=budget)


# ---------------------------------------------------------------- main sweep

def run_cell(n, t, a, seed, mode, layers, budget, generator):
    rng = np.random.default_rng([seed, t, a, 0 if mode == "oneshot" else 1])
    if generator == "x0":
        G_mat = op_on_qubit(PAULIS[1], 0, n)
    else:  # sumx
        G_mat = sum(op_on_qubit(PAULIS[1], q, n) for q in range(n))
    blocks = build_layers(n, t, layers if mode == "interleaved" else max(t, 1),
                          rng)
    phi, G_eff = make_state_fn(blocks, G_mat, mode, n)

    # QFI (finite difference on the state family)
    delta = 1e-4
    rho_p = reduced_density(phi(delta), n, a)
    rho_m = reduced_density(phi(-delta), n, a)
    rho_0 = reduced_density(phi(0.0), n, a)
    drho = (rho_p - rho_m) / (2 * delta)
    fqa = qfi_from_states(rho_0, drho)
    # full QFI = 4 Var (oneshot) / FD on global pure state (both modes)
    psi_p, psi_m = phi(delta), phi(-delta)
    rho_full0 = np.outer(phi(0.0), phi(0.0).conj())
    drho_full = (np.outer(psi_p, psi_p.conj())
                 - np.outer(psi_m, psi_m.conj())) / (2 * delta)
    fq_full = qfi_from_states(rho_full0, drho_full)

    spread = generator_spread_metrics(G_eff, n, a)
    qelm = qelm_readout(phi, n, a, rng, budget=budget)

    # CRB at full budget with the optimal (unconstrained) measurement on A
    crb = 1.0 / (budget * fqa) if fqa > 1e-9 else np.inf
    eta = float(crb / qelm["mse"]) if np.isfinite(crb) else 0.0
    # needle efficiency: Fisher of the single best Pauli setting vs F_Q^A
    eta_needle = (qelm["s_max_sq"] / fqa) if fqa > 1e-9 else 0.0
    return dict(n=n, t=t, a=a, seed=seed, mode=mode,
                fqa=fqa, fq_full=fq_full, eta=eta, eta_needle=eta_needle,
                crb=crb, **spread, **qelm)


def aggregate(records, key):
    """mean/std of `key` over seeds, per (t, a)."""
    out = {}
    for r in records:
        out.setdefault((r["t"], r["a"]), []).append(r[key])
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}


def make_figure(records, t_list, a_list, n, mode, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    grids = {}
    for key in ["fqa", "eta", "eta_needle"]:
        agg = aggregate(records, key)
        grids[key] = np.array([[agg[(t, a)][0] for t in t_list]
                               for a in a_list])

    fq_ref = np.array([[np.mean([r["fq_full"] for r in records
                                 if r["t"] == t and r["a"] == a])
                        for t in t_list] for a in a_list])

    def heat(ax, Z, title, log=False, vmin=None, vmax=None):
        M = np.log10(np.clip(Z, 1e-12, None)) if log else Z
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis",
                       vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(t_list)), t_list)
        ax.set_yticks(range(len(a_list)),
                      [f"{a} (f={a / n:.2f})" for a in a_list])
        ax.set_xlabel("t (T gates / magic)")
        ax.set_ylabel("|A| (access)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    heat(axes[0, 0], grids["fqa"] / np.clip(fq_ref, 1e-12, None),
         r"$F_Q^A / F_Q$  (info available in A)", vmin=0, vmax=1)
    heat(axes[0, 1], grids["eta"],
         r"$\log_{10}\,\eta$ = CRB/MSE (extractability)", log=True)
    heat(axes[1, 0], grids["eta_needle"],
         r"$\log_{10}\,\eta_{\rm needle}$ = best single-Pauli Fisher / $F_Q^A$",
         log=True)

    ax = axes[1, 1]
    ts = np.array([r["t"] for r in records])
    x = np.array([r["fqa"] / max(r["fq_full"], 1e-12) for r in records])
    y = np.array([max(r["eta_needle"], 1e-12) for r in records])
    sizes = np.array([15 + 25 * (r["a"] - min(a_list)) for r in records])
    sc = ax.scatter(x, y, c=ts, s=sizes, cmap="plasma", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel(r"$F_Q^A / F_Q$ (information present)")
    ax.set_ylabel(r"$\eta_{\rm needle}$ (extractable by best setting)")
    ax.set_title("Money plot: high-info / low-extractability = Region III")
    plt.colorbar(sc, ax=ax, label="t")

    fig.suptitle(f"mode={mode}, n={n}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"figure -> {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--mode", choices=["oneshot", "interleaved", "both"],
                   default="both")
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--budget", type=int, default=131072,
                   help="total measurement shots per theta-sample, split "
                        "across the 4^a - 1 Pauli settings")
    p.add_argument("--generator", choices=["x0", "sumx"], default="x0")
    p.add_argument("--t", type=int, nargs="*",
                   default=[0, 1, 2, 4, 6, 8, 12, 16, 24, 32])
    p.add_argument("--a", type=int, nargs="*", default=[2, 4, 5, 6])
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    if args.smoke:
        args.n, args.seeds = 6, 2
        args.t, args.a = [0, 4, 16], [2, 3]

    # --- unit checks -------------------------------------------------------
    rng = np.random.default_rng(0)
    # pauli_coeffs: G = X_0 on 2 qubits -> only coefficient 'IX' (idx 0*4+1)
    c = pauli_coeffs(op_on_qubit(PAULIS[1], 0, 2), 2).reshape(-1)
    assert abs(c[1] - 1.0) < 1e-12 and np.sum(np.abs(c) > 1e-12) == 1, \
        "pauli_coeffs failed IX test"
    c = pauli_coeffs(op_on_qubit(PAULIS[3], 1, 2), 2).reshape(-1)
    assert abs(c[12] - 1.0) < 1e-12, "pauli_coeffs failed ZI test"
    # oneshot full QFI for G=X_0 must be 4 Var = 4
    blocks = build_layers(4, 2, 2, rng)
    phi, G_eff = make_state_fn(blocks, op_on_qubit(PAULIS[1], 0, 4),
                               "oneshot", 4)
    d = 1e-4
    r0 = np.outer(phi(0.0), phi(0.0).conj())
    dr = (np.outer(phi(d), phi(d).conj())
          - np.outer(phi(-d), phi(-d).conj())) / (2 * d)
    assert abs(qfi_from_states(r0, dr) - 4.0) < 1e-3, "full QFI != 4"
    print("unit checks passed")

    modes = (["oneshot", "interleaved"] if args.mode == "both"
             else [args.mode])
    for mode in modes:
        records = []
        t0 = time.time()
        cells = list(iproduct(args.t, args.a, range(args.seeds)))
        for i, (t, a, seed) in enumerate(cells):
            r = run_cell(args.n, t, a, seed, mode, args.layers, args.budget,
                         args.generator)
            records.append(r)
            if (i + 1) % 20 == 0 or i == len(cells) - 1:
                print(f"[{mode}] {i + 1}/{len(cells)} cells, "
                      f"{time.time() - t0:.0f}s", flush=True)
        tag = f"{args.out}_{mode}_n{args.n}_{args.generator}"
        with open(f"{tag}.json", "w") as f:
            json.dump(records, f, indent=1)
        print(f"records -> {tag}.json")
        make_figure(records, args.t, args.a, args.n, mode, f"{tag}.png")

        # console summary: mean over seeds
        print(f"\n=== {mode}: mean F_Q^A/F_Q | log10(eta_needle) "
              f"per (a x t) ===")
        agg_f = aggregate(records, "fqa")
        agg_e = aggregate(records, "eta_needle")
        agg_ff = aggregate(records, "fq_full")
        header = "a\\t " + " ".join(f"{t:>12d}" for t in args.t)
        print(header)
        for a in args.a:
            row = [f"{agg_f[(t, a)][0] / max(agg_ff[(t, a)][0], 1e-12):.2f}"
                   f"/{np.log10(max(agg_e[(t, a)][0], 1e-12)):+.1f}"
                   for t in args.t]
            print(f"a={a}  " + " ".join(f"{s:>12s}" for s in row))


if __name__ == "__main__":
    main()
