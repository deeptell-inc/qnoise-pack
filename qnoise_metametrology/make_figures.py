#!/usr/bin/env python
"""Regenerate the three manuscript figures from the shipped data.

Usage (from repo root):
  python qnoise_metametrology/make_figures.py --data data --out figures
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Journal-legible defaults: figures are generated at their final printed
# width (7.16 in, two-column) so fonts are NOT shrunk on inclusion; 8 pt
# matches IEEE/APS caption text. Type-42 fonts keep the PDF text
# selectable/accessible.
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
FIGSIZE = (7.16, 2.75)

from scipy.optimize import curve_fit


def logmodel(t, alpha, c):
    return np.log((1 - c) * np.exp(-alpha * t) + c)


def model(t, alpha, c):
    return (1 - c) * np.exp(-alpha * np.asarray(t)) + c


def fit_eta(recs, a):
    ts = sorted({r["t"] for r in recs})
    etas = [np.mean([r["eta_needle"] for r in recs
                     if r["t"] == t and r["a"] == a]) for t in ts]
    e0 = max(etas[0], 1e-12)
    y = np.log(np.clip(np.array(etas) / e0, 1e-12, None))
    po, pc = curve_fit(logmodel, np.array(ts, float), y, p0=[0.14, 1e-2],
                       bounds=([0, 1e-9], [3, 1]), maxfev=20000)
    return po[0], float(np.sqrt(pc[0, 0])), po[1]


def fig1(data, out):
    recs = json.load(open(data / "results_oneshot_n8_x0.json"))
    t_list = sorted({r["t"] for r in recs})
    a_list = sorted({r["a"] for r in recs})
    n = 8

    def grid(key):
        M = np.zeros((len(a_list), len(t_list)))
        S = np.zeros_like(M)
        for i, a in enumerate(a_list):
            for j, t in enumerate(t_list):
                v = [r[key] for r in recs if r["a"] == a and r["t"] == t]
                M[i, j], S[i, j] = np.mean(v), np.std(v)
        return M, S

    Ff, _ = grid("fqa")
    Fq, _ = grid("fq_full")
    E, Es = grid("eta_needle")
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    panels = [
        (axes[0], Ff / np.clip(Fq, 1e-12, None),
         r"(a) information present:  $F_Q^A/F_Q$", False),
        (axes[1], np.log10(np.clip(E, 1e-12, None)),
         r"(b) best-Pauli sensitivity proxy:  $\log_{10}\,\eta_{\rm needle}$",
         True)]
    for ax, Z, title, log in panels:
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis",
                       vmin=(0 if not log else -2), vmax=(1 if not log else 0))
        ax.set_xticks(range(len(t_list)), t_list)
        ax.set_yticks(range(len(a_list)), [f"{a / n:.2f}" for a in a_list])
        ax.set_xlabel("magic  $t$  (number of $T$ gates)")
        ax.set_ylabel("access fraction  $f=|A|/n$")
        ax.set_title(title)
        ax.axhline(0.5, color="crimson", lw=2, ls="--")
        plt.colorbar(im, ax=ax)
    axes[0].text(0.3, 0.18, "$f=1/2$ decoupling reference (one-shot)",
                 color="crimson")
    fig.suptitle(f"One-shot encoding, $n={n}$, 8 circuit realizations per "
                 f"cell (cell std $\\leq$ {Es.max():.2f})")
    fig.tight_layout()
    fig.savefig(out / "fig1_phase_diagram.pdf")
    fig.savefig(out / "fig1_phase_diagram.png", dpi=300)
    plt.close(fig)


def fig2(data, out):
    recs = json.load(open(data / "lean2_oneshot_n10.json"))
    ts = sorted({r["t"] for r in recs})
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    colors = {3: "tab:red", 5: "tab:blue", 8: "tab:purple"}
    for a in [3, 5, 8]:
        rank = [np.mean([r["gram_rank"] / r["gram_m"] for r in recs
                         if r["t"] == t and r["a"] == a]) for t in ts]
        axes[0].plot(ts, rank, "o-", color=colors[a], label=f"f={a / 10:.1f}")
    for a in [5, 8]:
        eta = [np.mean([r["eta_needle"] for r in recs
                        if r["t"] == t and r["a"] == a]) for t in ts]
        fq = [np.mean([r["fqa"] / max(r["fq_full"], 1e-12) for r in recs
                       if r["t"] == t and r["a"] == a]) for t in ts]
        axes[1].plot(ts, eta, "o-", color=colors[a],
                     label=f"$\\eta$, f={a / 10:.1f}")
        axes[1].plot(ts, fq, "s--", color=colors[a], alpha=0.4,
                     label=f"$F_Q^A/F_Q$, f={a / 10:.1f}")
    for ax in axes:
        ax.axvspan(6, 16, color="gold", alpha=0.18)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(-0.15, 70)
        ax.set_xlabel("t (T gates)")
    axes[0].set_ylabel("rank$(\\Gamma)/m$")
    axes[0].set_title("(a) Identifiability fills with magic")
    axes[0].legend()
    axes[1].set_yscale("log")
    axes[1].set_ylabel("$\\eta_{\\rm needle}$,  $F_Q^A/F_Q$")
    axes[1].set_title("(b) Pauli proxy drains; information persists "
                      "($f\\geq 0.5$)")
    axes[1].legend(ncol=2)
    fig.suptitle("n=10 one-shot, 16 circuit realizations per cell: "
                 "finite-size window (shaded, f=0.5)")
    fig.tight_layout()
    fig.savefig(out / "fig2_sweetspot.pdf")
    fig.savefig(out / "fig2_sweetspot.png", dpi=300)
    plt.close(fig)


def fig3(data, out):
    fits = json.load(open(data / "alpha_fits.json"))
    fam = {}
    for fn, alist, n in [("family_oneshot_n10.json", [6, 8], 10),
                         ("family_oneshot_n12.json", [8, 9], 12)]:
        recs = json.load(open(data / fn))
        for a in alist:
            al, er, _ = fit_eta([dict(r, eta_needle=r["eta_family"])
                                 for r in recs], a)
            fam[(n, a)] = (al, er)
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    ax = axes[0]
    cmap = plt.get_cmap("tab10")
    ci = 0
    for r in fits:
        if r["mode"] != "oneshot" or r["f"] <= 0.55:
            continue
        if abs(r["f"] - 0.75) > 0.08 and r["n"] not in (14, 16, 18):
            continue
        if r["n"] in (16, 18) and r["f"] < 0.65:
            continue
        color = cmap(ci % 10)
        ci += 1
        ts = np.array(r["ts"], float)
        ax.semilogy(ts, r["etas"], "o", ms=4, color=color,
                    label=f"n={r['n']}, f={r['f']:.2f}")
        tt = np.linspace(0, ts.max(), 300)
        ax.semilogy(tt, model(tt, r["alpha"], r["floor"]), "-",
                    color=color, alpha=0.6)
    tt = np.linspace(0, 64, 300)
    ax.semilogy(tt, np.exp(-np.log(4 / 3) / 2 * tt), "k--", lw=1.5,
                label=r"branching-bound rate $e^{-t\,\ln(4/3)/2}$")
    ax.set_xlabel("$t$ ($T$ gates)")
    ax.set_ylabel(r"$\eta_{\rm needle}(t)/\eta_{\rm needle}(0)$")
    ax.set_ylim(2e-4, 2)
    ax.set_title("(a) collapse vs branching-bound rate, $n = 8$–$18$")
    ax.legend(loc="lower left")
    ax = axes[1]
    for (n, mode) in sorted({(r["n"], r["mode"]) for r in fits}):
        if mode != "oneshot":
            continue
        sel = [r for r in fits if r["n"] == n and r["mode"] == mode
               and r["f"] > 0.5 + 1e-9 and np.isfinite(r["alpha"])]
        if not sel:
            continue
        ax.errorbar([r["f"] for r in sel], [r["alpha"] for r in sel],
                    yerr=[r["alpha_err"] for r in sel], marker="o", ls="",
                    capsize=3, label=f"needle n={n}")
    for (n, a), (al, er) in fam.items():
        ax.errorbar([a / n], [al], yerr=[er], marker="s", ls="", capsize=3,
                    mfc="none", color="gray",
                    label="family" if (n, a) == (10, 6) else None)
    ax.axhline(np.log(4 / 3) / 2, color="k", lw=1.5, ls="--")
    ax.text(0.56, 0.148, r"$\ln(4/3)/2 = 0.144$")
    ax.axhline(np.log(4 / 3) / 4, color="gray", lw=1, ls=":")
    ax.set_xlabel("$f = |A|/n$")
    ax.set_ylabel(r"$\alpha$ (per $T$ gate)")
    ax.set_title(r"(b) fitted $\alpha$: needle vs commuting family")
    ax.set_ylim(0, 0.25)
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    fig.savefig(out / "fig3_alpha.pdf")
    fig.savefig(out / "fig3_alpha.png", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(exist_ok=True)
    fig1(data, out)
    print("figures/fig1_phase_diagram.png")
    fig2(data, out)
    print("figures/fig2_sweetspot.png")
    fig3(data, out)
    print("figures/fig3_alpha.png")


if __name__ == "__main__":
    main()
