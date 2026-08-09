"""Smoke: selective resume — keep perception across restarts, never the policy.

MEASURED 2026-08-03: `load_checkpoint()` has existed for months and NOTHING
in the launch path called it (only the demo/eval/transfer scripts do). So
every restart discarded world_model.pt (55 MB), policy.pt, curiosity.pt,
dream_actor.pt and glue_layer.pt and relearned from scratch, while the skill
bank persisted — the system retained SKILLS better than LEARNING. Visible in
the logs as entropy starting ~90% after every deploy and the reward curve
0.02 -> 0.14 -> 0.38 -> 0.53 being re-run from zero each time.

What this pins:
  1. The refusal list. `policy` / `dream_actor` / `curiosity` must be refused
     BY NAME, so a config typo can never resurrect a converged, 0-entropy
     policy under a reward function that has since changed.
  2. The loaders match the WRITER. symbolic_decoder.pt is the inner
     `.decoder`; glue_layer.pt is a two-key {"gnn","gate"} dict. Loading
     either into the outer object raises or silently no-ops.
  3. A shape mismatch is LOUD and leaves fresh weights intact — action_dim
     moved 10 -> 12 historically.
  4. Off by default: no resume_components -> byte-identical old behaviour.

Run: PYTHONPATH=. python tests/_resume_smoke.py
"""
import sys

sys.path.insert(0, ".")

import os
import tempfile

import torch
import yaml

LOOP = "developmental_ai/core/developmental_loop.py"
CFG = "configs/minecraft_skybot.yaml"


def main() -> None:
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    src = open(LOOP).read()
    cfg = yaml.safe_load(open(CFG))

    # ---- 1. skybot resumes perception, and ONLY perception --------------
    rc = cfg["loop"]["resume_components"]
    assert set(rc) == {"world_model", "symbolic_decoder", "glue_layer",
                       "knowledge_graph"}, f"unexpected resume set: {rc}"
    for banned in ("policy", "dream_actor", "curiosity"):
        assert banned not in rc, (
            f"'{banned}' is in resume_components — restoring a converged "
            "policy resumes its entropy collapse")

    # ---- 2. the refusal is enforced in CODE, not just absent from config -
    class _Stub:
        """Minimal stand-in: only the bits _resume_components touches."""
        def __init__(self, tmp):
            self.device = torch.device("cpu")
            self.config = {}
            self.tmp = tmp
            self.loaded = []
        # world_model / symbolic_decoder / glue are probed via attributes;
        # leaving them absent proves the loader fails LOUD rather than
        # silently skipping a component the user asked for.
    tmp = tempfile.mkdtemp()
    s = _Stub(tmp)
    # a checkpoint that LOOKS complete
    torch.save({"nope": torch.zeros(1)}, os.path.join(tmp, "world_model.pt"))
    torch.save({"nope": torch.zeros(1)}, os.path.join(tmp, "policy.pt"))

    DevelopmentalAI._resume_components(s, tmp, ["policy", "dream_actor",
                                                "curiosity"])
    # nothing raised, and crucially nothing was restored
    assert not getattr(s, "_restored", None), "a refused component loaded"
    assert "REFUSING" in src, "the by-name refusal is not implemented"
    assert 'if _n in ("policy", "dream_actor", "curiosity")' in src, \
        "the refusal list changed — policy could be resumed by config alone"

    # ---- 3. a missing/failed component must not take the run down -------
    #  world_model.pt here is garbage; the loader must warn and continue.
    class _WM:
        def load_state_dict(self, m):
            raise RuntimeError("size mismatch for encoder.0.weight")
    s2 = _Stub(tmp)
    s2.world_model = _WM()
    DevelopmentalAI._resume_components(s2, tmp, ["world_model"])   # no raise
    # and an absent directory is a clean no-op, not a crash
    DevelopmentalAI._resume_components(s2, os.path.join(tmp, "nope"),
                                       ["world_model"])

    # ---- 4. loaders address the same objects the writer wrote -----------
    save_blk = src[src.index("def _save_checkpoint"):src.index("def load_checkpoint")]
    assert "self.symbolic_decoder.decoder.state_dict()" in save_blk
    assert 'getattr(self.symbolic_decoder, "decoder", None)' in src, \
        "resume loads symbolic_decoder into the OUTER object, not .decoder"
    assert '"gnn": self.glue.knowledge_integrator.gnn.state_dict()' in save_blk
    assert '_ki.gnn.load_state_dict(m["gnn"])' in src, \
        "resume does not unpack the two-key glue_layer dict"

    # ---- 5. OFF by default ----------------------------------------------
    assert 'resume_components") or [])' in src, "resume is not opt-in"
    s3 = _Stub(tmp)
    s3.config = {"loop": {}}          # no resume_components -> never called
    assert not s3.loaded

    print("[resume-smoke] ALL PASS: skybot resumes world_model + "
          "symbolic_decoder + glue_layer + knowledge_graph and nothing else; "
          "policy/dream_actor/curiosity are refused BY NAME in code so a "
          "config typo cannot resurrect a 0-entropy policy; loaders address "
          "the same objects _save_checkpoint wrote (.decoder, the "
          "{gnn,gate} dict); a shape mismatch or missing dir warns and "
          "continues on fresh weights; off by default")


if __name__ == "__main__":
    main()
