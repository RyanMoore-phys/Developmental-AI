"""READ-ONLY bridge: existing skill_bank registry rows -> contracts.Skill.

WHAT IS CLAIMED
    1. Reads `<storage_dir>/registry.json` (falling back to `.bak` exactly
       like SkillBank._load_registry) with plain `open(..., "r")`. It never
       constructs a SkillBank (whose __init__ runs os.makedirs), never
       writes, never loads policy.pt. tests check the directory is
       byte-identical afterwards.
    2. REFUSES any path inside `skill_bank_mc_curiosity/` — the live agent's
       accumulated developmental memory (CLAUDE.md §3). Copy the registry
       elsewhere first if you need to inspect it.
    3. Identity is the bank's `skill_id`; the display `name` travels in
       payload["display_name"] and is marked non-identifying (CLAUDE.md
       §4.6). Two rows with the same name stay two skills; two rows with
       different names are not thereby different behaviours — that is
       skills.find_duplicates' job, on probe states, with the weights loaded.
    4. Competence: ONLY `asked_log` (0/1 over counted real invocations) is
       converted, to a Beta(1 + k, 1 + n - k) in payload["legacy_competence"].
       `success_rate` is NOT used: per skill_bank.py's own comment it is
       "an online BCE logit with no denominator", and `successful_episodes`
       is derived from it.
    5. Initiation/termination: the legacy bank has no learned sets. Grounded
       `preconditions` become initiation {"kind": "legacy_preconditions"};
       otherwise {"kind": "unlearned"}. Termination is the options runtime's
       rule {"kind": "legacy_option_runtime"} — declared, not learned, and
       labelled so.

Offline only.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

from ..contracts import ContractError, Skill

FORBIDDEN_COMPONENT = "skill_bank_mc_curiosity"


def _guard(storage_dir: str) -> str:
    p = os.path.realpath(os.path.abspath(storage_dir))
    if FORBIDDEN_COMPONENT in p.split(os.sep):
        raise ContractError(
            f"refusing to read {storage_dir!r}: {FORBIDDEN_COMPONENT}/ is the live "
            f"agent's developmental memory (CLAUDE.md §3). Copy registry.json "
            f"elsewhere and point the bridge at the copy.")
    return p


def read_registry(storage_dir: str) -> Tuple[List[Dict[str, Any]], str]:
    """-> (raw skill rows, the file they came from). Missing registry -> ([], "")."""
    p = _guard(storage_dir)
    path = os.path.join(p, "registry.json")
    for cand in (path, path + ".bak"):
        if not os.path.exists(cand):
            continue
        try:
            with open(cand, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        rows = data.get("skills", []) if isinstance(data, dict) else []
        return [r for r in rows if isinstance(r, dict) and r.get("skill_id")], cand
    return [], ""


def legacy_to_skill(row: Dict[str, Any], source: str = "") -> Skill:
    sid = str(row["skill_id"])
    log = [int(x) for x in (row.get("asked_log") or []) if int(x) in (0, 1)]
    k, n = sum(log), len(log)
    pre = row.get("preconditions")
    if isinstance(pre, dict) and pre:
        initiation = {"kind": "legacy_preconditions",
                      "predicates": {str(a): bool(b) for a, b in pre.items()}}
    else:
        initiation = {"kind": "unlearned"}
    payload = {
        "display_name": str(row.get("name", "")),
        "display_name_is_identity": False,
        "legacy_source": source,
        "legacy_arch": str(row.get("arch", "flat")),
        "legacy_competence": {"a": 1.0 + k, "b": 1.0 + n - k, "n": n,
                              "source": "asked_log (counted real invocations)"},
        "legacy_ignored": ["success_rate", "successful_episodes", "is_mastered",
                           "mastery_level"],
    }
    for key in ("obs_dim", "action_dim", "head_dim", "enc_dim"):
        if isinstance(row.get(key), int):
            payload[f"legacy_{key}"] = int(row[key])
    return Skill(skill_id=sid, version=1,
                 controller_ref=f"skill_bank:{sid}:policy.pt",
                 initiation=initiation,
                 termination={"kind": "legacy_option_runtime",
                              "learned": False},
                 competence_evidence=(), payload=payload)


def read_skill_bank(storage_dir: str) -> List[Skill]:
    rows, src = read_registry(storage_dir)
    out, seen = [], set()
    for r in rows:
        s = legacy_to_skill(r, src)
        if s.skill_id in seen:
            raise ContractError(f"registry {src} lists skill_id {s.skill_id!r} twice")
        seen.add(s.skill_id)
        out.append(s)
    return out
