"""Smoke test for the value-gated / naturally-annealing / sharpened distillation.

Confirms: (1) augment build is correct, (2) the distill path runs without error,
(3) the value GATE fires with a non-degenerate fraction (not ~0, not ~1), and
(4) the effective distill weight ANNEALS downward as the real policy starts
solving (natural anneal). PPO stays in control the whole time.
"""
import copy, yaml
from developmental_ai.core.developmental_loop import DevelopmentalAI

base = yaml.safe_load(open("configs/minigrid_doorkey.yaml"))
cfg = copy.deepcopy(base)
cfg.setdefault("environment", {})["name"] = "MiniGrid-DoorKey-5x5-v0"
cfg["environment"]["max_episode_steps"] = 250
cfg["seed"] = 42
cfg.setdefault("symbolic", {})["enabled"] = False
cfg.setdefault("llm", {})["enabled"] = False
cfg["llm"].setdefault("reward_shaping", {})["enabled"] = False
cfg.setdefault("loop", {})["curriculum_enabled"] = False
dt = cfg.setdefault("dream_training", {})
dt["enabled"] = True
dt["control"] = False               # AUGMENT mode (real policy keeps control)
dt["activate_after_episodes"] = 20
dt["distill_weight"] = 1.0
# new levers (explicit so the smoke pins them; these are also the defaults)
dt["value_gate"] = True
dt["natural_anneal"] = True
dt["teacher_temp"] = 0.5
cfg.setdefault("skill_bank", {})["storage_dir"] = "/tmp/dream_fix_smoke_sb"

a = DevelopmentalAI(config=cfg)
print(f"[build] dream_augment={a.dream_augment} dream_control={a.dream_control} "
      f"value_gate={a.dream_distill_value_gate} anneal={a.dream_distill_anneal} "
      f"temp={a.dream_distill_temp} distill_opt={a._distill_opt is not None}", flush=True)
a.run(total_timesteps=45000, log_interval=40, verbose=0)

dl = list(a.training_metrics["dream_distill_loss"])
gf = list(a.training_metrics["dream_distill_gate_frac"])
ew = list(a.training_metrics["dream_distill_eff_weight"])
frac = sum(1 for r in a.training_metrics['episode_reward'] if r > 0.1) / max(1, len(a.training_metrics['episode_reward']))
print(f"[end] dream_training_active(should be False)={a.dream_training_active} "
      f"dream_augment_active={a.dream_augment_active} solve_ema={a._dream_solve_ema:.3f}", flush=True)
print(f"[end] distill updates n={len(dl)}  loss first={dl[0] if dl else 0:.4f} last={dl[-1] if dl else 0:.4f}", flush=True)
if gf:
    print(f"[end] gate_frac: first={gf[0]:.3f} last={gf[-1]:.3f} mean={sum(gf)/len(gf):.3f} "
          f"min={min(gf):.3f} max={max(gf):.3f}", flush=True)
if ew:
    print(f"[end] eff_weight: first={ew[0]:.3f} last={ew[-1]:.3f} (base=1.0; should anneal DOWN as solving)", flush=True)
print(f"[end] PPO-controlled frac_solved={frac:.3f} episodes={len(a.training_metrics['episode_reward'])}", flush=True)

# acceptance: distill ran, gate is non-degenerate on average, weight annealed if solving
gate_ok = bool(gf) and (0.02 < (sum(gf)/len(gf)) < 0.98)
anneal_ok = (not ew) or (a._dream_solve_ema < 0.05) or (ew[-1] < ew[0] + 1e-6)
ok = (len(dl) > 0) and (not a.dream_training_active) and gate_ok and anneal_ok
print("\nDREAM FIX SMOKE OK" if ok else "\nDREAM FIX SMOKE PROBLEM"
      f" (distill_n={len(dl)} gate_ok={gate_ok} anneal_ok={anneal_ok})", flush=True)
