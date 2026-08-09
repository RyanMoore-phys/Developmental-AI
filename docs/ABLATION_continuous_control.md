# Ablation — Developmental machinery on continuous control (Hopper-v4)

**Claim (one line):** the developmental components are *task-appropriate, not
universally beneficial* — on a continuous-control task, **intrinsic-motivation
curiosity helps a lot**, the **symbolic / knowledge-graph layer hurts**, and the two
**cancel** so the full "organism" lands on top of the vanilla baseline.

This is a deliberately *honest negative-with-structure* result: it tells us **where**
the machinery applies, and it motivated a concrete code change (auto-disabling the
symbolic layer on continuous action spaces).

---

## Pre-registration

- **Environment:** MuJoCo `Hopper-v4` (continuous 3-D action space), 300k env steps/run.
- **Question:** does the developmental stack improve a continuous-control agent, and
  which *part* of it is responsible?
- **Arms (each = the same PPO core, components toggled):**
  - `vanilla` — PPO only.
  - `curiosity` — PPO + intrinsic motivation (ICM / learning-progress).
  - `organism` — PPO + curiosity + symbolic/KG knowledge-conditioning (the "full" stack).
- **Metric:** final return (mean of last-window evaluation), higher is better.
- **Design:** 8 seeds per arm (5 initial + 3 confirmatory), **pre-registered before the
  extra 3 seeds were run**. Significance by **Welch's t-test** (unequal variance), two-tailed.
- **Falsification targets, stated up front:**
  - If `organism > curiosity`, the symbolic layer earns its place on continuous control.
  - If `organism ≈ vanilla` while `curiosity > vanilla`, the symbolic layer is *cancelling*
    a real gain and should not be on by default here.

---

## Results (n = 8 per arm)

| Arm | Mean return | SD | Median | n |
|---|---:|---:|---:|---:|
| `curiosity` | **889.8** | 160.3 | 955.0 | 8 |
| `organism` | 698.6 | 56.8 | 705.0 | 8 |
| `vanilla` | 679.1 | 162.1 | 657.0 | 8 |

Per-seed returns:
```
organism   [616, 726, 709, 700, 779, 617, 741, 701]
curiosity  [980, 1062, 930, 1037, 778, 990, 672, 669]
vanilla    [690, 490, 984, 685, 825, 514, 629, 616]
```

### Pairwise tests (Welch, two-tailed)

| Comparison | Δ mean | t | p | Cohen's d | Verdict |
|---|---:|---:|---:|---:|---|
| `curiosity` − `vanilla` | **+210.6** | +2.61 | **0.020** | +1.31 | curiosity **helps** (large) |
| `organism` − `curiosity` | **−191.1** | −3.18 | **0.012** | −1.59 | symbolic layer **hurts** (large) |
| `organism` − `vanilla` | +19.5 | +0.32 | 0.756 | +0.16 | full stack ≈ vanilla (null) |

---

## Interpretation

1. **Curiosity is the workhorse on continuous control.** Intrinsic motivation lifts
   return by ~+31% over vanilla (p = 0.020, d = 1.31). On a sparse-shaping locomotion
   task, better exploration is exactly the right inductive bias, and it shows.

2. **The symbolic / KG layer is actively harmful here.** Stacking it on top of curiosity
   *removes* the entire curiosity gain (p = 0.012, d = −1.59). The knowledge-graph layer
   was designed for **discrete, structured, object/relation** tasks; on a continuous
   torque-control problem its fact-extraction and knowledge-conditioning add noise and a
   compute tax with no signal to condition on. (It is *not* a wash — it is a real cost.)

3. **The two effects cancel.** `organism ≈ vanilla` (p = 0.76) is not "the stack does
   nothing." It is "+curiosity −symbolic ≈ 0." Reporting only the full-organism-vs-vanilla
   number would have **hidden** both a real benefit and a real harm. The decomposition is
   the result.

**Theme:** developmental machinery is *task-appropriate*. The same components that help
on discrete structured environments (see the rung ladder) do not transfer wholesale to
continuous control — and our framework is honest enough to measure which piece does what.

---

## Action taken

The symbolic layer now **auto-disables on continuous action spaces** (overridable via
`symbolic.force_on_continuous: true`), so the default continuous-control configuration is
`curiosity`-equivalent — the arm that actually wins — rather than the self-cancelling full
stack. Discrete environments are unaffected.

```
core/developmental_loop.py  — auto-disable when not self.is_discrete
configs/default.yaml        — symbolic.force_on_continuous: false
```

---

## Threats to validity (honest caveats)

- **One environment, one task family.** This is Hopper locomotion. The claim is "on this
  continuous-control task," not "on all continuous control." Hopper is a standard probe,
  but a second env (e.g. Walker2d / HalfCheetah) would strengthen generality. *Not yet run.*
- **n = 8.** Adequate to detect the large effects here (d > 1.3) at p < 0.05, but the
  curiosity-vs-vanilla gap, while significant, has wide spread (SD ≈ 160); more seeds would
  tighten the interval. The symbolic-harm effect is the most robust of the three.
- **Single PPO core / hyperparameters.** Components were toggled on a fixed backbone; we did
  not re-tune per arm, so this measures the *marginal* contribution of each component under a
  shared config, not each component's best-case ceiling.
- **"Curiosity helps" is calibrated to p = 0.020**, i.e. significant at α = 0.05 but not
  beyond — stated as "helps," not "dramatically/robustly helps."

## Reproduce

```
_mj_longrun.sh    # 5 seeds × 3 arms, 300k steps (initial)
_mj_extra.sh      # seeds 5,6,7 (confirmatory)
_mj_analyze.py    # combined n=8 mean ± sd + Welch t-tests
```
