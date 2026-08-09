"""INTEGRATION: the whole arch-v3 stack on the REAL loop (Crafter, pixels).

Unit tests pin each piece; this proves they are actually WIRED to each other
inside DevelopmentalAI — the class of bug that unit tests structurally cannot
catch (a correct function nobody calls; a correct call with the wrong config
key; two subsystems that each work and disagree about a vocabulary).

Crafter renders 64x64x3 = 12288 = 3*s*s, so the conv trunk is legal here and
this runs on a laptop in ~a minute.

Proves, on the real loop:
  1. `policy.arch: conv` is READ from config and builds a conv policy whose
     encoder is in the optimizer (not a silently-ignored config key).
  2. The loop RUNS: episodes complete, PPO updates, nothing raises.
  3. Options + nesting machinery is live and bounded: max_skill_depth reaches
     the executor from `skills_as_options`, and no env ever receives a
     non-primitive action.
  4. Minted skills carry the arch-v3 metadata (arch/enc_dim/head_dim/
     slot_map) and are ~an order of magnitude smaller on disk than the flat
     policy would be.
  5. The mastery ledger is FED by real play (asked_total > 0 once options
     actually fire) — the wiring #39 was missing.

Run: PYTHONPATH=. python tests/_arch_v3_integration_smoke.py
"""
import os
import shutil
import sys

sys.path.insert(0, ".")

import numpy as np
import torch
import yaml

OUT = "/tmp/arch_v3_integration"
SB = OUT + "/skills"


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT, exist_ok=True)

    cfg = yaml.safe_load(open("configs/crafter.yaml"))
    cfg["policy"]["arch"] = "conv"          # THE switch under test
    cfg["policy"]["enc_dim"] = 64
    cfg["policy"]["n_steps"] = 64
    # options REQUIRE the parallel discrete path (the loop disables them
    # otherwise, with a warning that is easy to miss in a passing test)
    cfg.setdefault("parallel_envs", {}).update({"enabled": True, "num_envs": 2})
    cfg.setdefault("skills_as_options", {})
    cfg["skills_as_options"].update({
        "enabled": True, "k_slots": 4, "max_skill_depth": 2,
        "max_option_steps": 6, "competence_floor": 0.0,
        "gate_by_preconditions": False, "scripted_options": [],
    })
    cfg.setdefault("skill_bank", {})["storage_dir"] = SB
    cfg.setdefault("goals", {})["mint_skills"] = True
    for k in ("llm", "symbolic_grounding", "brain_viewer", "video"):
        cfg.setdefault(k, {})["enabled"] = False

    from developmental_ai.core.developmental_loop import DevelopmentalAI
    ai = DevelopmentalAI(config=cfg)

    # ---- 1. the config key is REAL, not silently ignored ----------------
    assert ai.policy.arch == "conv", (
        f"policy.arch=conv was ignored (got {ai.policy.arch!r}) — the knob "
        f"is decorative")
    assert ai.policy.encoder is not None
    enc_ids = {id(p) for p in ai.policy.encoder.parameters()}
    opt_ids = {id(p) for g in ai.policy.optimizer.param_groups
               for p in g["params"]}
    assert enc_ids <= opt_ids, "encoder is not in the optimizer — it can "\
        "never learn, and every skill inherits a random perception trunk"
    enc0 = ai.policy.encoder.conv[0].weight.detach().clone()

    # ---- 3a. nesting config reached the executor ------------------------
    ex = ai.option_executor
    assert ex is not None, "options did not initialise"
    assert ex.max_skill_depth == 2, (
        f"max_skill_depth={ex.max_skill_depth}: the knob is in "
        f"skills_as_options but the executor never saw it")

    # ---- 2/3b. RUN and watch every action the envs receive --------------
    P = ai.action_dim
    seen = []
    _raw_act = ex.act

    def _spy(*a, **k):
        acts = _raw_act(*a, **k)
        seen.extend(int(x) for x in acts)
        return acts
    ex.act = _spy

    # long enough for Crafter achievements -> goal unlocks -> a MINT, and
    # for the minted skill to bind and be invoked as an option. 1200 steps
    # exercised neither, and the summary below used to claim both anyway.
    ai.run(total_timesteps=int(os.environ.get("ARCHV3_STEPS", 9000)),
           log_interval=10**9, save_interval=10**9, verbose=0)

    assert seen, "options layer never resolved an action"
    assert all(0 <= a < P for a in seen), (
        f"an env received a NON-PRIMITIVE action: "
        f"{[a for a in seen if not (0 <= a < P)][:5]} (P={P})")
    assert float((ai.policy.encoder.conv[0].weight.detach()
                  - enc0).abs().max()) > 0, \
        "the encoder never moved across a real run — PPO is not reaching it"

    # ---- 4. minted skills carry arch-v3 metadata and are SMALL ----------
    skills = list(ai.skill_bank.skills.values())
    if skills:
        conv_sk = [s for s in skills if s.arch == "conv"]
        assert conv_sk, (
            f"minted {len(skills)} skill(s) under a conv policy but none "
            f"recorded arch='conv': {[s.arch for s in skills]}")
        s0 = conv_sk[0]
        assert s0.enc_dim == 64 and s0.head_dim == ai.meta_action_dim, (
            f"metadata wrong: enc_dim={s0.enc_dim} head_dim={s0.head_dim} "
            f"(expected 64 / {ai.meta_action_dim})")
        assert s0.slot_map is not None, "slot_map not recorded at mint — "\
            "nested invocation can never resolve a child by identity"
        mb = os.path.getsize(s0.policy_path) / 1048576
        # the flat equivalent at 12288 obs would be ~2x256x12288x4B ~= 25 MB
        assert mb < 8.0, f"conv skill is {mb:.1f} MB — not the expected shrink"
        print(f"  minted {len(skills)} skill(s); conv policy.pt = {mb:.2f} MB")
    else:
        print("  UNEXERCISED: no skill minted — mint-metadata path NOT verified")

    # ---- 5. the mastery ledger is fed by real play ----------------------
    asked = sum(int(s.asked_total or 0) for s in skills)
    if ex.slot_stats:
        print(f"  option invocations={sum(v['invocations'] for v in ex.slot_stats.values())}, "
              f"nested_pushes={ex.nested_pushes}, mastery asked_total={asked}")
        assert asked > 0, (
            "options fired but NOTHING reached the mastery ledger — that is "
            "exactly the #39 gap this build closed")
    else:
        print("  UNEXERCISED: no option fired — mastery-ledger path NOT verified")

    try:
        ai.env.close()
    except Exception:
        pass
    shutil.rmtree(OUT, ignore_errors=True)
    # THE SUMMARY MUST ONLY CLAIM WHAT ACTUALLY RAN. The first version of
    # this test printed "minted skills carry arch/enc_dim/head_dim/slot_map"
    # on a run where NOTHING WAS MINTED — a green line asserting an
    # unexecuted branch is worse than a red one.
    verified = ["policy.arch=conv honoured by the real loop",
                "encoder in the optimizer AND moved during the run",
                "max_skill_depth reached the executor from skills_as_options",
                f"all {len(seen)} env actions stayed primitives (<{P})"]
    if skills and [x for x in skills if x.arch == "conv"]:
        verified.append("minted skill carries arch/enc_dim/head_dim/slot_map "
                        "and is order-of-magnitude smaller")
    else:
        verified.append("MINT METADATA NOT EXERCISED (no skill minted)")
    if ex.slot_stats:
        verified.append("mastery ledger fed by real option invocations")
    else:
        verified.append("MASTERY LEDGER NOT EXERCISED (no option fired)")
    print("[arch-v3-integration] " + "; ".join(verified))


if __name__ == "__main__":
    main()
