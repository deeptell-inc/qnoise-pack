#!/usr/bin/env python
"""Doping-convention dependence of the collapse constant (open item 6).

Ensembles (n=8, dense statevector, G = X_0, one-shot):
  t1     : single T per doping event (baseline; alpha = ln(4/3)/2 predicted)
  tlayer4: 4 parallel T gates on distinct random qubits per event
           (t counts TOTAL T gates; prediction: same alpha PER T GATE)
  ccz    : one CCZ on a random qubit triple per event (t counts CCZ gates;
           branching factor differs -> computable new alpha)

Measures eta_needle(t) and sensitivity purity Pi(t); fits rates.
"""

import argparse
import json
import time

import numpy as np
from qiskit.quantum_info import Operator, random_clifford

from prototype_phase_diagram import pauli_coeffs, reduced_density, \
    qfi_from_states


def dope_phases(kind, n, rng):
    """Diagonal phase vector(s) for one doping event."""
    idx = np.arange(2 ** n)
    if kind == "t1":
        q = int(rng.integers(n))
        return np.exp(1j * np.pi / 4 * ((idx >> q) & 1))
    if kind == "tlayer4":
        qs = rng.choice(n, size=4, replace=False)
        ph = np.ones(2 ** n, dtype=complex)
        for q in qs:
            ph *= np.exp(1j * np.pi / 4 * ((idx >> int(q)) & 1))
        return ph
    if kind == "ccz":
        q1, q2, q3 = [int(q) for q in rng.choice(n, size=3, replace=False)]
        bits = ((idx >> q1) & 1) * ((idx >> q2) & 1) * ((idx >> q3) & 1)
        return np.where(bits == 1, -1.0 + 0j, 1.0 + 0j)
    raise ValueError(kind)


def run_cell(n, kind, events, seed, a):
    rng = np.random.default_rng([seed, events, a, hash(kind) % 2 ** 31])
    U = Operator(random_clifford(n, seed=int(rng.integers(2 ** 31)))).data
    for _ in range(events):
        U = dope_phases(kind, n, rng)[:, None] * U
        C = Operator(random_clifford(n, seed=int(rng.integers(2 ** 31)))).data
        U = C @ U
    psi0 = np.zeros(2 ** n, dtype=complex)
    psi0[0] = 1.0
    u0, u1 = U @ psi0, U @ np.roll(psi0, 1)  # X_0|0..0> = |0..1>

    def phi(th):
        return np.cos(th) * u0 - 1j * np.sin(th) * u1

    dth = 1e-3
    ep = pauli_coeffs(reduced_density(phi(dth), n, a), a).reshape(-1).real \
        * 2 ** a
    em = pauli_coeffs(reduced_density(phi(-dth), n, a), a).reshape(-1).real \
        * 2 ** a
    s2 = ((ep - em) / (2 * dth)) ** 2
    s2[0] = 0.0
    d = 1e-4
    r0 = reduced_density(phi(0.0), n, a)
    dr = (reduced_density(phi(d), n, a)
          - reduced_density(phi(-d), n, a)) / (2 * d)
    fqa = qfi_from_states(r0, dr)
    tot = s2.sum()
    pur = float(np.sum((s2 / tot) ** 2)) if tot > 1e-18 else 0.0
    eta = float(s2.max() / fqa) if fqa > 1e-9 else 0.0
    return dict(kind=kind, events=events, seed=seed, a=a,
                eta_needle=eta, purity=pur, fqa=fqa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--a", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--events", type=int, nargs="*",
                    default=[0, 1, 2, 3, 4, 6, 8, 12, 16, 24])
    ap.add_argument("--kinds", nargs="*", default=["t1", "tlayer4", "ccz"])
    args = ap.parse_args()

    recs, t0 = [], time.time()
    for kind in args.kinds:
        for ev in args.events:
            for seed in range(args.seeds):
                recs.append(run_cell(args.n, kind, ev, seed, args.a))
        print(f"{kind} done {time.time() - t0:.0f}s", flush=True)
    with open(f"ensemble_alpha_n{args.n}.json", "w") as f:
        json.dump(recs, f, indent=1)

    from scipy.optimize import curve_fit

    def model(t, al, c):
        return np.log((1 - c) * np.exp(-al * t) + c)

    print(f"\n=== n={args.n}, a={args.a}: rates per doping EVENT ===")
    for kind in args.kinds:
        evs = sorted({r["events"] for r in recs if r["kind"] == kind})
        etas = [np.mean([r["eta_needle"] for r in recs
                         if r["kind"] == kind and r["events"] == e])
                for e in evs]
        purs = [np.mean([r["purity"] for r in recs
                         if r["kind"] == kind and r["events"] == e])
                for e in evs]
        e0 = max(etas[0], 1e-12)
        y = np.log(np.clip(np.array(etas) / e0, 1e-12, None))
        po, pc = curve_fit(model, np.array(evs, float), y, p0=[0.2, 1e-2],
                           bounds=([0, 1e-9], [5, 1]), maxfev=20000)
        w = [e for e in evs if 1 <= e <= 12]
        pr = -np.polyfit(w, [np.log(purs[evs.index(e)]) for e in w], 1)[0]
        n_t = {"t1": 1, "tlayer4": 4, "ccz": None}[kind]
        per_t = f", per-T-gate alpha={po[0] / n_t:.3f}" if n_t else ""
        print(f"{kind:>8}: alpha_eta/event={po[0]:.3f}+/-"
              f"{np.sqrt(pc[0, 0]):.3f}, purity rate/event={pr:.3f}{per_t}")


if __name__ == "__main__":
    main()
