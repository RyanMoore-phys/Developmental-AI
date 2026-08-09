"""Smoke: the competence gate is a GATE, not a one-way latch.

Measured live 2026-07-25 (139k steps, 2227 option decisions): ALL 16 skills sat
at invocations=0 and every single option invocation was the scripted bootstrap.
Cause: `self_model` froze the shared competence bias to stop failure-dominated
drift, but `achievement_goals._load` restored the bias FROM THE STATE FILE — a
value already poisoned by that very drift (-5.68 => competence 0.0034 for every
task, vs competence_floor 0.25). Frozen + poisoned = permanent, and re-applied
on every restart. A task needed w >= +4.58 just to be offered.

Three contracts here, each pinned to a way the gate can silently re-latch:
  A. a restored bias NEVER re-poisons the frozen parameter;
  B. the offered-set is non-empty for a fresh skill at the configured floor;
  C. competence lookup survives slot RENAME (id keeps its mint-time name).

Run: PYTHONPATH=. python tests/_competence_gate_smoke.py
"""
import math
import sys

sys.path.insert(0, ".")

import numpy as np
import torch
import yaml

from developmental_ai.core.self_model import CompetencePredictor
from developmental_ai.policy.options import _slot_index_of


def main() -> None:
    # ---- A. THE LATCH ITSELF ------------------------------------------
    # The bias is frozen, so whatever load leaves there is PERMANENT.
    cp = CompetencePredictor(n_tasks=8)
    assert cp.head.bias.requires_grad is False, \
        "bias is no longer frozen — the original drift bug is back"

    src = open("developmental_ai/core/achievement_goals.py").read()
    assert 'self.competence.head.bias.fill_(0.0)' in src, \
        ("the restore path must pin the frozen bias to 0. Restoring a stored "
         "bias into a frozen parameter is what latched the gate shut.")
    assert 'bias.fill_(\n                        float(d.get("competence_bias"' \
        not in src, "the poisoned restore is still present"

    # the poison, reproduced: bias -5.68 puts EVERY task under any sane floor
    poisoned = -5.68
    assert 1 / (1 + math.exp(-poisoned)) < 0.01
    with torch.no_grad():
        cp.head.bias.fill_(poisoned)
    assert float(cp.predict_all().max()) < 0.01, \
        "sanity: a poisoned bias should flatten competence to ~0"

    # ---- B. GATE OPEN AT THE CONFIGURED FLOOR -------------------------
    cfg = yaml.safe_load(open("configs/minecraft_skybot.yaml"))
    floor = float(cfg["skills_as_options"]["competence_floor"])
    fresh = CompetencePredictor(n_tasks=8)          # zeros init, bias 0
    c0 = float(fresh.predict_all()[0])
    assert abs(c0 - 0.5) < 1e-6, f"fresh competence should be 0.5, got {c0}"
    assert c0 >= floor, (
        f"a NEVER-TRIED skill starts below competence_floor {floor} — it can "
        f"never be offered, so it can never earn competence: a latch")

    # and a genuinely failing task must still close the gate (it is a GATE)
    for _ in range(60):
        fresh.update(0, 0.0)
    assert float(fresh.predict_all()[0]) < floor, \
        "repeated failure no longer closes the gate — it is not a gate at all"
    # ...while an untouched task stays open (bias freeze = decoupled tasks)
    assert float(fresh.predict_all()[1]) >= floor, \
        ("failing task 0 dragged task 1 down — the shared-bias coupling bug "
         "has returned")

    # ---- C. NAME DRIFT MUST NOT ORPHAN A SKILL ------------------------
    # live: 3/16 skills were orphaned when slots 0/1/2 were re-keyed to
    # `discovered_N` while the ids kept their mint-time names.
    assert _slot_index_of("ach_01_break_oak_leaves") == 1
    assert _slot_index_of("ach_46_discovered_38") == 46
    assert _slot_index_of("scripted_chop") == -1
    opt_src = open("developmental_ai/policy/options.py").read()
    assert "_slot_index_of(b[\"skill_id\"])" in opt_src, \
        "the competence gate does not fall back to the stable slot index"
    loop_src = open("developmental_ai/core/developmental_loop.py").read()
    assert "_comp.update({i: float(_cv[i])" in loop_src, \
        "competence dict is not keyed by slot index — the fallback finds nothing"
    assert loop_src.count("_comp.update({i: float(_cv[i])") == 2, \
        ("only ONE of the two _comp build sites was fixed. Fixing a latch in "
         "one gate and not its twin has happened THREE times in this project.")

    print(f"[competence-gate-smoke] ALL PASS: bias pinned to 0 on restore "
          f"(frozen), fresh competence 0.500 >= floor {floor} so an untried "
          f"skill is offered, repeated failure still closes the gate, tasks "
          f"stay decoupled, and slot-index fallback survives rename "
          f"(both build sites patched)")


if __name__ == "__main__":
    main()
