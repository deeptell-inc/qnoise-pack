#!/usr/bin/env python
"""Convert the protocol's shot budgets into wall-clock time at assumed
repetition rates (feasibility section of the manuscript).

Only arithmetic on quantities already in the paper; the repetition rates
are ASSUMPTIONS stated as such, not measured device numbers.
"""
import json
import sys

n, a, m = 10, 5, 20            # window operating point of the finite-shot study
S = 4096                       # shots per setting (same-budget experiment)
K_full = 4 ** a - 1            # 1023 Pauli settings on A
K_list = [16, 128, 512, K_full]
rates = {"1 kHz": 1e3, "10 kHz": 1e4, "100 kHz": 1e5}


def budget(K, S=S, m=m):
    return S * K * (2 * m + 2) + S     # calibration + signal + estimation


def fmt_time(sec):
    if sec < 60:
        return f"{sec:.0f} s"
    if sec < 3600:
        return f"{sec/60:.1f} min"
    if sec < 86400:
        return f"{sec/3600:.1f} h"
    return f"{sec/86400:.1f} d"


rows = []
for K in K_list:
    B = budget(K)
    rows.append({"K": K, "B_shots": B,
                 **{f"t_{r}": B / v for r, v in rates.items()}})

# gates on the manuscript's quoted numbers
assert budget(K_full) == 175_992_832          # Table in Appendix E
assert abs(2 * m * K_full * S / budget(K_full) - 0.95) < 0.005   # calibration share 95%
assert fmt_time(budget(K_full) / 1e4) == "4.9 h"

if __name__ == "__main__":
    print(f"n={n} a={a} m={m} S={S}  (budget B = S*K*(2m+2)+S)")
    print(f"{'K':>5} {'B [shots]':>14} " + " ".join(f"{r:>10}" for r in rates))
    for r in rows:
        print(f"{r['K']:>5} {r['B_shots']:>14,} " +
              " ".join(f"{fmt_time(r[f't_{k}']):>10}" for k in rates))
    out = sys.argv[1] if len(sys.argv) > 1 else None
    if out:
        json.dump(rows, open(out, "w"), indent=1)
        print("->", out)
