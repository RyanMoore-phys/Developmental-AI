"""Combine the Hopper ablation seeds (0-7) and report mean+/-sd + Welch t-tests."""
import re
import numpy as np

arms = {"organism": [], "curiosity": [], "vanilla": []}
for arm in arms:
    for s in range(8):
        try:
            t = open(f"mj_long/{arm}_s{s}.log").read()
            v = re.findall(r"Avg reward:\s+([0-9.]+)", t)
            arms[arm].append(float(v[-1]) if v else None)
        except Exception:
            arms[arm].append(None)


def arr(x):
    return np.array([v for v in x if v is not None], dtype=float)


def welch(a, b):
    a, b = arr(a), arr(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    na, nb = len(a), len(b)
    se = (va / na + vb / nb) ** 0.5
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return ma - mb, t, df, na, nb


print("=== COMBINED ABLATION (Hopper-v4, 300k steps) ===")
for a, v in arms.items():
    d = arr(v)
    print(f"{a:10s}", [None if x is None else round(x) for x in v],
          " mean", round(d.mean(), 1), " sd", round(d.std(ddof=1), 1), " n", len(d))
print("--- Welch t-tests ---")
for (x, y) in [("curiosity", "vanilla"), ("organism", "curiosity"), ("organism", "vanilla")]:
    diff, t, df, na, nb = welch(arms[x], arms[y])
    print(f"{x:10s} vs {y:10s}: diff={diff:7.1f}  t={t:5.2f}  df={df:4.1f}  (n={na},{nb})")
