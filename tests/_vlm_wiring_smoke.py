"""VLM wiring smoke (2026-09-20).

THE LIVE FAILURE THIS ENCODES
    `ollama list` showed `qwen2.5vl:3b` present on the training host. Every run logged

        VLM MODEL NOT FOUND: 'qwen2.5vl:3b' is not on the Ollama server.
        Available: ["model='qwen2.5vl", "model='qwen2.5vl:3b' modified_at=..."]

    and proceeded with the labeller DISABLED — while Ollama held 4.2 GB of
    VRAM resident for a model nothing ever called. Zero labels, full cost.

    CAUSE: newer `ollama` python clients return pydantic objects. `list()`
    gives a ListResponse whose `.models` holds Model objects carrying a
    `.model` attribute. The probe tested `isinstance(m, dict)`, fell through
    to `str(m)`, and compared the REPR — which never equals the tag.

    This is the project's signature failure (machinery that runs and quietly
    does nothing) in the ONE subsystem whose job is to bootstrap meaning, and
    it was invisible except as an absence of facts.

    WHAT IS *NOT* BROKEN, checked rather than assumed: GenerateResponse DOES
    implement `.get()`, so the four `resp.get("response")` call sites are
    fine. The first instinct was to "fix" them too; measuring first avoided
    four pointless edits to a working path.

Contracts:
    A. Name extraction handles EVERY shape a client has returned: dict, bare
       string, and pydantic object. Shape-specific parsing is what broke.
    B. probe_ollama_model resolves a model that `ollama list` reports, and
       rejects one it does not — against the LIVE server when present.
    C. A missing/unreachable server DEGRADES, never raises. The run must
       survive a broken VLM exactly as it survives a slow one.
    D. GenerateResponse is dict-accessible, so the label path's
       `resp.get("response")` is correct.
    E. END TO END: a real generate call through the symbolizer's own query
       path returns parseable JSON naming declared predicates.

Run: PYTHONPATH=. python tests/_vlm_wiring_smoke.py
"""
import json
import os
import sys

sys.path.insert(0, ".")

SKIP_LIVE = os.environ.get("VLM_SMOKE_OFFLINE") == "1"


def _names_from(models):
    """The extraction under test, mirroring llm_module's probe."""
    names = set()
    for m in models:
        if isinstance(m, dict):
            n = m.get("model") or m.get("name")
        elif isinstance(m, str):
            n = m
        else:
            n = getattr(m, "model", None) or getattr(m, "name", None)
        if n:
            names.add(str(n))
            names.add(str(n).split(":")[0])
    return names


class _FakeModel:
    """Stands in for ollama's pydantic Model: a `.model` attribute and a
    repr that CONTAINS the tag — which is exactly why comparing reprs
    looked like it might work and did not."""
    def __init__(self, tag):
        self.model = tag

    def __repr__(self):
        return f"model='{self.model}' modified_at=datetime.datetime(2026,9,19)"


def test_name_extraction_handles_every_shape():
    want = "qwen2.5vl:3b"
    for label, entry in (
            ("dict/model", {"model": want}),
            ("dict/name", {"name": want}),
            ("bare string", want),
            ("pydantic object", _FakeModel(want))):
        names = _names_from([entry])
        assert want in names, f"{label}: {want} not resolved from {names}"
        assert "qwen2.5vl" in names, f"{label}: bare family not resolved"
    # THE REGRESSION ITSELF: the old code produced the repr, which must NOT
    # be mistaken for a tag.
    bad = _names_from([_FakeModel(want)])
    assert not any("modified_at" in n for n in bad), (
        "a repr leaked into the resolved names — this is the 4.2 GB bug")
    print(f"  A. dict/model, dict/name, str and pydantic all resolve "
          f"{want!r}; no repr leaks")


def test_probe_resolves_against_the_live_server():
    if SKIP_LIVE:
        print("  B. skipped (VLM_SMOKE_OFFLINE=1)")
        return
    try:
        import ollama
    except ImportError:
        print("  B. skipped (ollama not installed)")
        return
    from developmental_ai.llm.llm_module import probe_ollama_model
    r = ollama.list()
    models = getattr(r, "models", None) or (
        r.get("models") if isinstance(r, dict) else [])
    if not models:
        print("  B. skipped (server has no models)")
        return
    tags = sorted(_names_from(models))
    real = next((t for t in tags if ":" in t), tags[0])

    assert probe_ollama_model(real, deep=False) is True, (
        f"{real} is on the server and `ollama list` reports it, but the "
        f"probe says it is missing — this is the exact 4.2 GB failure")
    assert probe_ollama_model("definitely-not-a-model:v0", deep=False) is False, (
        "the probe accepted a model that does not exist; a wrong model "
        "answering is worse than none, because the predicates look real")
    print(f"  B. probe resolves {real!r} and rejects a nonexistent tag "
          f"({len(tags)} names visible)")


def test_absent_server_degrades_and_never_raises():
    """C. min_labels keeps a silent VLM safe, but only if nothing throws."""
    from developmental_ai.llm import llm_module as L
    saved = L.OLLAMA_AVAILABLE
    try:
        L.OLLAMA_AVAILABLE = False
        assert L.probe_ollama_model("anything:v1", deep=True) is True, (
            "with no client the probe must return True (unverifiable, not "
            "unhealthy) — returning False would disable a working VLM on a "
            "box where the library merely is not importable")
    finally:
        L.OLLAMA_AVAILABLE = saved
    print("  C. no client -> unverifiable-not-unhealthy, no exception")


def test_generate_response_is_dict_accessible():
    """D. The label path uses resp.get("response"). Prove that is correct.

    Checked rather than assumed: on finding the list() bug the instinct was
    that every `.get` on an ollama response must be broken too. It is not —
    GenerateResponse implements `.get`. Four edits avoided by measuring.
    """
    if SKIP_LIVE:
        print("  D. skipped (VLM_SMOKE_OFFLINE=1)")
        return
    try:
        import ollama
    except ImportError:
        print("  D. skipped (ollama not installed)")
        return
    r = ollama.list()
    models = getattr(r, "models", None) or []
    if not models:
        print("  D. skipped (no models)")
        return
    tag = _names_from(models[:1])
    tag = next(t for t in sorted(tag) if ":" in t)
    resp = ollama.generate(model=tag, prompt='Return {"ok": true} as JSON.',
                           format="json",
                           options={"num_predict": 16, "temperature": 0.0})
    assert hasattr(resp, "get"), (
        "GenerateResponse lost .get — every resp.get(\"response\") call site "
        "in llm_module and vlm_symbolizer is now broken")
    txt = (resp.get("response") or "").strip()
    json.loads(txt)
    assert txt == (getattr(resp, "response", "") or "").strip(), (
        "dict access and attribute access disagree")
    print(f"  D. GenerateResponse.get works and agrees with .response")


def test_end_to_end_scene_query():
    """E. A real call through the symbolizer's OWN path returns usable JSON.

    The probe only proves a tag exists. This proves the thing the agent
    actually calls returns something the head can be trained on — the
    difference between "the model is installed" and "the sensor works".
    """
    if SKIP_LIVE:
        print("  E. skipped (VLM_SMOKE_OFFLINE=1)")
        return
    try:
        import ollama  # noqa: F401
        import numpy as np
        from developmental_ai.llm.vlm_symbolizer import (
            VLMSymbolizer, PREDICATES)
    except ImportError as e:
        print(f"  E. skipped ({e})")
        return
    import yaml
    cfg = yaml.safe_load(open(os.path.join("configs",
                                           "minecraft_skybot.yaml")))
    model = (cfg.get("symbolic_grounding") or {}).get("model", "qwen2.5vl:3b")

    sym = VLMSymbolizer(latent_dim=32, model=model, device="cpu")
    if not getattr(sym, "_available", False):
        print("  E. skipped (symbolizer reports the VLM unavailable)")
        return
    # A frame with sky over ground — enough for the model to say something.
    frame = np.zeros((128, 128, 3), np.uint8)
    frame[:64] = (120, 165, 255)
    frame[64:] = (90, 130, 70)
    png = sym._encode_png(frame)
    assert png, "the frame could not be encoded for the VLM"
    txt = sym._ollama_scene_query(png)
    assert txt, "the scene query returned nothing"
    data = json.loads(txt)
    assert isinstance(data, dict) and data, f"unparseable/empty: {txt[:120]}"
    known = set(PREDICATES)
    overlap = known & set(data)
    assert overlap, (
        f"the VLM answered {sorted(data)[:8]} — none of which are declared "
        f"predicates. The head can only be trained on the declared "
        f"vocabulary, so a reply with no overlap yields zero labels.")
    print(f"  E. scene query returned {len(data)} keys, "
          f"{len(overlap)} in the declared vocabulary "
          f"(e.g. {sorted(overlap)[:4]})")


if __name__ == "__main__":
    test_name_extraction_handles_every_shape()
    test_probe_resolves_against_the_live_server()
    test_absent_server_degrades_and_never_raises()
    test_generate_response_is_dict_accessible()
    test_end_to_end_scene_query()
    print("[vlm-wiring] ALL PASS")
