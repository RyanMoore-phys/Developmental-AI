"""Skill notes — every skill is a READABLE DOCUMENT, not just weights.

The Obsidian move (user's vision, July 2026): a skill directory becomes a
note —

    ach_03_discovered_3/
      policy.pt          the behaviour (an attachment, not the identity)
      versions/          superseded policies (provenance, recoverable)
      discovery.png      what the world looked like at first unlock
      note.md            THE SKILL: name, what it does, when it applies,
                         what it achieves, its history, its [[links]]

note.md is the unit the brain viewer renders, the thing a human reads to
know what the agent knows, and the place [[wikilinks]] live. Links are
EARNED only: grounded effects ("this skill breaks [[oak_log]]"), observed
prerequisites, composition — never embedding-similarity decoration.

VLM NAMING: llava looks at the discovery frame (plus any ground-truth
effect hint) and names the skill — "chop_oak_trunk", not "discovered_3".
The name is display metadata; skill_id stays slot-stable. Naming is
best-effort: no Ollama -> deterministic fallback, never a crash.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_NAME_PROMPT = (
    "You name a newly learned skill for a Minecraft-playing agent, from the "
    "agent's first-person view at the moment the skill first worked.{hint}\n"
    "Return ONLY a JSON object: {{\"name\": \"verb_noun_phrase\", "
    "\"description\": \"one sentence: what the agent does and what it "
    "achieves\"}}\n"
    "The name must be 2-4 lowercase words joined by underscores, starting "
    "with a verb (e.g. \"chop_oak_trunk\", \"clear_tall_grass\", "
    "\"approach_tree_line\"). Be literal about what is visible."
)


def name_skill_via_vlm(frame_png: bytes,
                       effect_hint: Optional[str] = None,
                       model: str = "llava:7b",
                       ) -> Optional[Tuple[str, str]]:
    """Ask the local VLM to name a skill from its discovery frame.

    Returns (name, description) or None. Synchronous (~1s) but only runs at
    mint time — a handful of calls per run, never in the step loop.
    """
    try:
        from developmental_ai.llm.llm_module import (
            _get_ollama_client, _parse_json_object)
        client = _get_ollama_client()
        if client is None:
            return None
        hint = (f" Ground truth: the skill caused '{effect_hint}'."
                if effect_hint else "")
        resp = client.generate(
            model=model, prompt=_NAME_PROMPT.format(hint=hint),
            images=[frame_png], options={"temperature": 0.2},
            keep_alive=-1)
        obj = _parse_json_object(resp.get("response", ""))
        if not isinstance(obj, dict):
            return None
        name = str(obj.get("name", "")).strip().lower()
        name = re.sub(r"[^a-z0-9_]+", "_", name).strip("_")[:48]
        desc = str(obj.get("description", "")).strip()[:300]
        if not name or "_" not in name:
            return None
        return name, desc or f"Learned behaviour: {name}"
    except Exception as e:  # naming is a nicety — never kill a run for it
        logger.debug("VLM skill naming failed: %s", e)
        return None


def encode_png(frame) -> Optional[bytes]:
    """Frame -> PNG bytes (for the VLM naming call)."""
    if frame is None:
        return None
    try:
        import io
        import imageio.v2 as imageio
        f = np.asarray(frame)
        if f.dtype != np.uint8:
            f = np.clip(f * 255 if f.max() <= 1.0 else f,
                        0, 255).astype(np.uint8)
        buf = io.BytesIO()
        imageio.imwrite(buf, f, format="png")
        return buf.getvalue()
    except Exception:
        return None


def save_discovery_frame(frame, skill_dir: str) -> Optional[str]:
    """Persist the unlock-moment frame as discovery.png (provenance)."""
    if frame is None:
        return None
    try:
        import imageio.v2 as imageio
        f = np.asarray(frame)
        if f.dtype != np.uint8:
            f = np.clip(f * 255 if f.max() <= 1.0 else f,
                        0, 255).astype(np.uint8)
        path = os.path.join(skill_dir, "discovery.png")
        imageio.imwrite(path, f)
        return path
    except Exception as e:
        logger.debug("discovery frame save failed: %s", e)
        return None


def _wikilink(x: str) -> str:
    return f"[[{x}]]"


def write_note(skill, skill_dir: str,
               effects: Optional[List[str]] = None,
               preconditions: Optional[Dict[str, bool]] = None,
               prerequisites: Optional[List[str]] = None,
               provenance: Optional[Dict] = None,
               competence_history: Optional[List[Tuple[float, float]]] = None,
               ) -> str:
    """Render note.md for a skill. Overwrites; the note is a VIEW of the
    skill's current state (history lives in the frontmatter lists).

    effects        block/achievement names this skill GROUNDEDLY causes
    preconditions  predicate -> bool at unlock (phase-2 vocabulary)
    prerequisites  skill_ids observed achieved-before (earned DAG evidence)
    provenance     {"run": ..., "episode": ..., "stream": ..., "timestep": ...}
    competence_history  [(unix_time, success_rate), ...]
    """
    effects = effects or []
    preconditions = preconditions or {}
    prerequisites = prerequisites or []
    provenance = provenance or {}
    competence_history = competence_history or []

    front = {
        "id": skill.skill_id,
        "name": skill.name,
        "competence": round(float(skill.success_rate), 3),
        "milestone_reached": bool(skill.is_mastered),
        "practice_episodes": int(skill.total_episodes),
        "created": time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(skill.created_at or time.time())),
        "updated": time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(skill.updated_at or time.time())),
        "effects": effects,
        "preconditions": {k: bool(v) for k, v in preconditions.items()},
        "prerequisites": prerequisites,
        "provenance": provenance,
        "competence_history": [
            [round(t, 1), round(c, 3)] for t, c in competence_history[-50:]],
    }

    lines = ["---", json.dumps(front, indent=1, default=str), "---", ""]
    lines.append(f"# {skill.name}")
    lines.append("")
    if skill.description:
        lines.append(skill.description)
        lines.append("")
    if effects:
        lines.append("## Achieves")
        for e in effects:
            lines.append(f"- {_wikilink(e)}")
        lines.append("")
    if preconditions:
        lines.append("## When it applies (grounded predicates at unlock)")
        for k, v in sorted(preconditions.items()):
            lines.append(f"- {k}: {'yes' if v else 'no'}")
        lines.append("")
    if prerequisites:
        lines.append("## Came after (observed, not assumed)")
        for p in prerequisites:
            lines.append(f"- {_wikilink(p)}")
        lines.append("")
    if os.path.exists(os.path.join(skill_dir, "discovery.png")):
        lines.append("## First success")
        lines.append("![discovery](discovery.png)")
        lines.append("")
    versions_dir = os.path.join(skill_dir, "versions")
    if os.path.isdir(versions_dir):
        vs = sorted(f for f in os.listdir(versions_dir) if f.endswith(".pt"))
        if vs:
            lines.append("## Policy history")
            for v in vs:
                lines.append(f"- versions/{v}")
            lines.append("")

    path = os.path.join(skill_dir, "note.md")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines))
    os.replace(tmp, path)
    return path


_TIER = {}  # block -> rank for "break_" naming (log > solid > plant)


def _block_rank(b: str) -> int:
    if "log" in b:
        return 2
    if b in ("grass", "tall_grass", "fern", "vine", "seagrass",
             "dead_bush", "poppy", "dandelion"):
        return 0
    return 1


def reground_names(skill_bank) -> int:
    """One-time cleanup of stale VLM-HALLUCINATED skill names (run-7 left
    "fish_blue_water", "collect_heart_items", ...). Replace any non-grounded,
    non-placeholder name with a grounded one derived from the skill's note
    effects ("break_<highest-tier-block>"), or a neutral slot name if the
    note has no effects. Grounded names and placeholders are left alone
    (placeholders get proper names at their next unlock). Returns count.

    Runs at startup so the brain shows HONEST names immediately, before any
    re-unlock. Later re-unlocks refine these with THIS run's ground truth.
    """
    from developmental_ai.skill_bank.skill_bank import (
        is_grounded_name, is_placeholder_name)
    import re as _re
    changed = 0
    for sid, sk in skill_bank.skills.items():
        if is_grounded_name(sk.name) or is_placeholder_name(sk.name):
            continue  # already grounded, or a placeholder (fixed on unlock)
        # a hallucination — reground from note effects if any
        front = read_note_frontmatter(
            os.path.join(skill_bank.storage_dir, sid)) or {}
        effects = [e for e in front.get("effects", []) if e]
        if effects:
            best = max(effects, key=_block_rank)
            new_name = f"break_{best}"
            new_desc = (f"Breaks {best} (regrounded from prior-run evidence; "
                        f"refines at next unlock).")
        else:
            m = _re.match(r"ach_(\d+)_", sid)
            slot = int(m.group(1)) if m else 0
            new_name = f"skill_slot_{slot:02d}"
            new_desc = "Discovered behaviour (name pending next unlock)."
        logger.info("regrounded skill name '%s' -> '%s'", sk.name, new_name)
        sk.name = new_name
        sk.description = new_desc
        changed += 1
    if changed:
        skill_bank._save_registry()
        logger.info("Regrounded %d hallucinated skill name(s)", changed)
    return changed


def backfill_notes(skill_bank, broadcaster=None,
                   knowledge_graph=None) -> int:
    """Write note.md for skills that predate the notes system.

    Naming/framing cannot be recovered retroactively (no discovery frame
    was kept), so a backfilled note is deliberately THIN: it records what
    is actually known — competence, practice, observed precedence from the
    broadcaster's unlock log — and nothing invented. The skill's real name
    and preconditions arrive automatically at its next unlock, via the
    metadata-backfill path in save_skill.

    Returns the number of notes written.
    """
    written = 0
    slot_names = list(getattr(broadcaster, "slot_names", []) or [])
    unlock_log = list(getattr(broadcaster, "unlock_log", []) or [])
    # slot -> the achieved-before mask of that slot's FIRST unlock
    first_mask = {}
    for ev in unlock_log:
        first_mask.setdefault(int(ev["slot"]), ev.get("achieved_before"))

    for sid, sk in skill_bank.skills.items():
        d = os.path.join(skill_bank.storage_dir, sid)
        if os.path.exists(os.path.join(d, "note.md")):
            continue
        prereqs = []
        m = re.match(r"ach_(\d+)_", sid)
        if m and int(m.group(1)) in first_mask:
            mask = np.asarray(first_mask[int(m.group(1))])
            prereqs = [f"ach_{i:02d}_{slot_names[i]}"[:64]
                       for i in np.nonzero(mask > 0)[0]
                       if i < len(slot_names)]
        try:
            os.makedirs(d, exist_ok=True)
            write_note(sk, d, effects=[],
                       preconditions=sk.preconditions or {},
                       prerequisites=prereqs,
                       provenance={"backfilled": True,
                                   "note": "minted before the notes system; "
                                           "name/preconditions fill in at "
                                           "its next unlock"},
                       competence_history=[(sk.updated_at or time.time(),
                                            float(sk.success_rate))])
            written += 1
        except Exception as e:
            logger.warning("note backfill failed for %s: %s", sid, e)
    if written:
        logger.info("Backfilled %d skill notes", written)
    return written


def read_note_frontmatter(skill_dir: str) -> Optional[Dict]:
    """Parse note.md frontmatter (the JSON block) — the viewer's node data."""
    path = os.path.join(skill_dir, "note.md")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            text = f.read()
        m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
        return json.loads(m.group(1)) if m else None
    except Exception:
        return None
