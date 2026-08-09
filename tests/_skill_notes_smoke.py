"""Skill notes smoke (Mac-ok, stubbed VLM).

Contracts:
  1. write_note renders note.md with parseable frontmatter and earned
     [[wikilinks]] only (effects + observed prerequisites).
  2. read_note_frontmatter round-trips what write_note stored.
  3. VLM naming sanitizes into verb_noun snake_case and falls back to None
     (never raises) when Ollama is absent or the answer is garbage.
  4. encode_png produces decodable PNG bytes from a uint8 frame.
"""
import json
import os
import shutil

import numpy as np

SB = "/tmp/skill_notes_smoke_sb"


def test_note_roundtrip():
    from developmental_ai.skill_bank.skill_bank import SkillBank
    from developmental_ai.skill_bank import skill_notes
    shutil.rmtree(SB, ignore_errors=True)
    bank = SkillBank(storage_dir=SB)
    sk = bank.save_skill(
        "ach_00_discovered_0", "chop_oak_trunk",
        {"w": __import__("torch").zeros(2)},
        description="Swing at a trunk until it breaks.",
        success_rate=0.62, total_episodes=9, dedup=True)
    d = os.path.join(SB, sk.skill_id)
    path = skill_notes.write_note(
        sk, d,
        effects=["oak_log", "birch_log"],
        preconditions={"tree_visible": True, "breakable_in_reach": True},
        prerequisites=["ach_01_discovered_1"],
        provenance={"episode": 12, "timestep": 48000, "env": "Treechop"})
    text = open(path).read()
    assert "[[oak_log]]" in text and "[[ach_01_discovered_1]]" in text
    assert "tree_visible: yes" in text
    front = skill_notes.read_note_frontmatter(d)
    assert front is not None and front["id"] == sk.skill_id
    assert front["competence"] == 0.62
    assert front["effects"] == ["oak_log", "birch_log"]
    assert front["prerequisites"] == ["ach_01_discovered_1"]
    assert front["provenance"]["timestep"] == 48000
    print("  1/2. note round-trip ok (wikilinks + frontmatter)")


def test_vlm_naming():
    from developmental_ai.skill_bank import skill_notes
    import developmental_ai.llm.llm_module as L

    class _Client:
        def __init__(self, answer):
            self.answer = answer

        def generate(self, **kw):
            return {"response": self.answer}

    orig = L._get_ollama_client
    try:
        L._get_ollama_client = lambda: _Client(
            json.dumps({"name": "Chop Oak-Trunk!!", "description": "d"}))
        got = skill_notes.name_skill_via_vlm(b"png", "oak_log")
        assert got is not None and got[0] == "chop_oak_trunk", got

        L._get_ollama_client = lambda: _Client("not json at all")
        assert skill_notes.name_skill_via_vlm(b"png") is None

        L._get_ollama_client = lambda: None
        assert skill_notes.name_skill_via_vlm(b"png") is None
    finally:
        L._get_ollama_client = orig
    print("  3. VLM naming sanitizes + fails safe ok")


def test_encode_png():
    from developmental_ai.skill_bank import skill_notes
    png = skill_notes.encode_png(
        np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8))
    assert png is not None and png[:4] == b"\x89PNG"
    assert skill_notes.encode_png(None) is None
    print("  4. encode_png ok")


if __name__ == "__main__":
    for fn in (test_note_roundtrip, test_vlm_naming, test_encode_png):
        print(f"[skill-notes-smoke] {fn.__name__}")
        fn()
    print("[skill-notes-smoke] ALL PASS")
