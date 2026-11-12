#!/usr/bin/env python
"""Figure for the scrambler-family experiments: figures/fig_zoo_n{N}.pdf."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42})
ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "figures"


def mean_trace(recs, xkey, val):
    xs = sorted({r[xkey] for r in recs})
    return np.array(xs), np.array([np.mean([val(r) for r in recs if r[xkey] == x]) for x in xs])


def main(n):
    ratio = lambda r: r["fqa"] / r["fq_full"]
    eta = lambda r: r["eta_needle"]
    B = json.load(open(DATA / f"zoo_brickwork_n{n}.json"))
    U = json.load(open(DATA / f"zoo_u1_n{n}.json"))
    M = json.load(open(DATA / f"zoo_matchgate_n{n}.json"))
    S = json.load(open(DATA / f"zoo_summary_n{n}.json"))
    a_list = sorted(int(a) for a in S["haar"])
    a_mid, a_hi = a_list[1], a_list[2]
    col = {a_list[0]: "tab:red", a_mid: "tab:blue", a_hi: "tab:purple"}
    fig, ax = plt.subplots(2, 3, figsize=(7.16, 4.5))

    # (a) information arrives at the light cone
    p = ax[0, 0]
    for a in (a_mid, a_hi):
        for place, ls in (("near", "-"), ("far", "--")):
            x, y = mean_trace([r for r in B if r["gate"] == "haar" and r["a"] == a and r["place"] == place], "depth", ratio)
            p.plot(x, y, ls, marker="o", ms=2.5, color=col[a], label=f"$f$={a/n:.1f}, {place}")
        p.axvline(n - a, color=col[a], lw=0.6, ls=":")
    p.set(xlabel="brickwork depth $D$", ylabel="$F_Q^A/F_Q$", title="(a) information vs depth")
    p.legend(loc="lower right")

    # (b) best-setting fraction vs depth
    p = ax[0, 1]
    for a in (a_mid, a_hi):
        x, y = mean_trace([r for r in B if r["gate"] == "haar" and r["a"] == a and r["place"] == "near"], "depth", eta)
        p.semilogy(x, y, "-o", ms=2.5, color=col[a], label=f"Haar gates, $f$={a/n:.1f}")
        p.axhline(S["haar"][str(a)]["eta"], color=col[a], lw=0.6, ls=":")
    x, y = mean_trace([r for r in B if r["gate"] == "clifford" and r["a"] == a_hi and r["place"] == "near"], "depth", eta)
    p.semilogy(x, y, "-s", ms=2.5, color="gray", label=f"Clifford gates, $f$={a_hi/n:.1f}")
    p.set(xlabel="brickwork depth $D$", ylabel=r"$\eta_{\rm needle}$", title="(b) best setting vs depth")
    p.legend(loc="lower left")

    # (c) U(1) symmetry
    p = ax[0, 2]
    for g, lab, c in (("haar", "generic", "tab:blue"), ("u1", "U(1)-symmetric", "tab:orange")):
        x, y = mean_trace([r for r in U if r["gate"] == g and r["a"] == a_mid and r["place"] == "near"], "depth", eta)
        p.semilogy(x, y, "-o", ms=2.5, color=c, label=lab)
    d = S["u1"][str(a_mid)]
    p.set(xlabel="brickwork depth $D$", ylabel=r"$\eta_{\rm needle}$",
          title=f"(c) U(1): plateau $\\times${d['eta_ratio']:.2f}")
    p.set_xscale("symlog", linthresh=4)
    p.legend(loc="upper right")

    # (d) matchgate: rank filling
    p = ax[1, 0]
    for a in a_list:
        x, y = mean_trace([r for r in M if r["a"] == a], "k", lambda r: r["gram_rank"] / r["gram_m"])
        p.plot(x, y, "-o", ms=2.5, color=col[a], label=f"$f$={a/n:.1f}")
        p.plot([0], [a / n], "x", color="k", ms=5)
    p.set(xlabel="number of CZ gates $k$", ylabel=r"rank$(\Gamma)/m$", xlim=(-0.5, 8.5),
          title="(d) rank at the free-fermion point")
    p.legend(loc="lower right")

    # (e) matchgate: best-setting fraction vs non-Gaussian gates
    p = ax[1, 1]
    for a in a_list:
        x, y = mean_trace([r for r in M if r["a"] == a], "k", eta)
        p.semilogy(x, y, "-o", ms=2.5, color=col[a], label=f"$f$={a/n:.1f}")
    p.set(xlabel="number of CZ gates $k$", ylabel=r"$\eta_{\rm needle}$",
          title="(e) best setting vs CZ count")
    p.legend(loc="upper right")

    # (f) decay rates with bootstrap intervals
    p = ax[1, 2]
    rows = [("doped Clifford\n(per $T$ gate)", 0.142, (0.130, 0.148), "k")]
    for a in a_list[1:]:
        d = S["matchgate"][str(a)]["eta_rate_per_cz"]
        rows.append((f"matchgate, $f$={a/n:.1f}\n(per CZ)", d["rate"], d["ci"], "tab:green"))
    for a in a_list[1:]:
        d = S["brickwork"][str(a)]["eta_rate_per_layer"]
        rows.append((f"Haar brickwork, $f$={a/n:.1f}\n(per layer)", d["rate"], d["ci"], "tab:blue"))
    d = S["u1"][str(a_mid)]["eta_rate_u1"]
    rows.append((f"U(1) brickwork, $f$={a_mid/n:.1f}\n(per layer)", d["rate"], d["ci"], "tab:orange"))
    for i, (lab, r, ci, c) in enumerate(rows):
        p.errorbar(r, -i, xerr=[[r - ci[0]], [ci[1] - r]], fmt="o", ms=3.5, color=c, capsize=2)
    p.set_yticks([-i for i in range(len(rows))])
    p.set_yticklabels([r[0] for r in rows], fontsize=6.5)
    p.axvline(np.log(4 / 3) / 2, color="k", lw=0.6, ls="--")
    p.set(xlabel="decay rate of $\\eta_{\\rm needle}$", title="(f) decay rates, 95% CI")

    fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.9)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"fig_zoo_n{n}.pdf")
    fig.savefig(OUT / f"fig_zoo_n{n}.png", dpi=200)
    print(OUT / f"fig_zoo_n{n}.pdf")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
