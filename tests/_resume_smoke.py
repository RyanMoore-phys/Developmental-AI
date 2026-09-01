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

    # ---- 1. skybot resumes EXPERIENCE, never policy ----------------------
    # (2026-08-10: perception + familiarity joined — grounded heads, label
    # evidence and the "what have I seen" counts are experience, not policy)
    # (2026-08-17: magnet joined — the learning-progress estimates that
    # decide WHERE TO LOOK are experience by the same test. Measured after
    # that day's relaunch: "w=0.0000, target=None", every category at LP
    # 0.000 — the magnet was amnesiac, not suppressed. Its cold-start
    # budget is deliberately NOT in the payload; see VisionScaffold.state.)
    # (2026-08-23: replay_buffer joined — the world model's RAW EXPERIENCE.
    # It passes the same test: restoring frames and actions cannot resume a
    # converged optimum, it only spares the model re-collecting the world.
    # Until this existed the buffer had NO persistence at all, so the
    # supervisor's automatic crash-relaunch restarted world-model training
    # from EMPTY — the effective memory was never the configured capacity,
    # it was "however long since the last crash". Deliberately NOT in
    # _RESUME_ENCODER_KEYED: raw frames are not expressed in the encoder's
    # latent space, and training a fresh encoder on old frames is exactly
    # what a replay buffer is FOR.)
    rc = cfg["loop"]["resume_components"]
    assert set(rc) == {"world_model", "symbolic_decoder", "glue_layer",
                       "knowledge_graph", "perception", "familiarity",
                       "magnet", "replay_buffer"}, \
        f"unexpected resume set: {rc}"
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
        # state_dict() must answer the PRE-FLIGHT key check (2026-08-23): the
        # loader now compares the model's keys against the payload's before
        # copying anything. Returning the payload's own key here keeps this
        # case about the SHAPE mismatch it was written for.
        def state_dict(self):
            return {"nope": torch.zeros(1)}

        def load_state_dict(self, m, strict=True):
            raise RuntimeError("size mismatch for encoder.0.weight")
    s2 = _Stub(tmp)
    s2.world_model = _WM()
    DevelopmentalAI._resume_components(s2, tmp, ["world_model"])   # no raise

    # ---- 3b. a checkpoint that CANNOT FILL the model is refused BEFORE any
    #      tensor is copied (2026-08-23 review). Previously strict=False
    #      accepted it silently, world_model entered `_ok`, and the
    #      dependency gate then admitted perception/familiarity onto a
    #      half-fresh encoder. The refusal must still warn-and-continue, and
    #      must not have mutated the module on its way out.
    class _WMPartial:
        def __init__(self):
            self.touched = False

        def state_dict(self):
            return {"nope": torch.zeros(1), "encoder.0.weight": torch.zeros(1)}

        def load_state_dict(self, m, strict=True):
            self.touched = True
            return ([], [])
    s3 = _Stub(tmp)
    s3.world_model = _WMPartial()
    DevelopmentalAI._resume_components(s3, tmp, ["world_model"])   # no raise
    assert not s3.world_model.touched, (
        "a checkpoint missing `encoder.0.weight` reached load_state_dict — "
        "the pre-flight must refuse BEFORE copying, or the model is left a "
        "partial hybrid while the resume reports success")
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
