"""Config-key smoke (2026-09-20).

THE FAILURE THIS ENCODES, MEASURED THE SAME DAY

    `self.config.get("X", default)` on a key that does not exist returns the
    DEFAULT. Silently. Forever. The subsystem runs, logs nothing unusual, and
    quietly uses a value nobody chose.

    An AST audit of every config.get() literal in developmental_loop.py
    against the live YAML found SIX such keys. Three were written this month:

        symbol_grounding.fovea_frac      -> lives at symbolic_grounding
        vision_scaffold.forward_actions  -> lives at llm.vision   (x2)

    Every one of them fell back to a default that HAPPENED to equal the
    configured value, so nothing misbehaved — which is exactly why nobody
    noticed. The trap is the next edit: change `llm.vision.forward_actions`
    and the flow probe keeps aiming at the old action while walkability
    labels the wrong bearing, with no error anywhere.

    This is the same species as the VLM bug found hours earlier, where a
    model-name comparison against a pydantic repr meant Ollama held 4.2 GB of
    VRAM for a labeller that produced zero labels. Code that runs and quietly
    does nothing is this project's signature failure, and a key typo is its
    cheapest form.

WHY A SOURCE CONTRACT AND NOT A RUNTIME CHECK
    A runtime check only fires on the branch that reads the key, which may be
    an env or a mode nobody exercised. Reading the source catches every
    literal whether or not that line ever runs, needs no torch, no gymnasium
    and no MineRL, and takes milliseconds.

Contracts:
    A. Every top-level `self.config.get("<literal>")` key in the loop either
       exists in the live config or is on OPTIONAL with a stated reason.
    B. OPTIONAL is honest: an entry that IS present in the config is stale
       and must be removed, so the allowlist cannot quietly become a
       dumping ground.
    C. The audit actually works — a deliberately bogus key is caught.

Run: PYTHONPATH=. python tests/_config_keys_smoke.py
"""
import ast
import os
import sys

sys.path.insert(0, ".")

import yaml

LOOP = os.path.join("developmental_ai", "core", "developmental_loop.py")
CFG = os.path.join("configs", "minecraft_skybot.yaml")

# Keys the loop reads that are DELIBERATELY absent from the live config.
# Each needs a reason. Anything not here and not in the YAML is a typo.
OPTIONAL = {
    "glue": "glue-layer block; absent means the defaults, and the live run "
            "has never configured it",
    "seed": "absent means a random seed, which is what a lifelong run wants",
    "symbolic_decoder": "the pixel-gated decoder is off in this config; the "
                        "block is only read when it is on",
}
# NOTE the five entries this list started with — async_replay, async_wm,
# sensors, spatial, slots — were removed within a minute of writing it,
# because contract B pointed out they ARE in the live config. Writing an
# allowlist from memory is how one stops being true; the contract is what
# keeps it honest.


def _config_get_keys(path):
    """Every literal key in `self.config.get("...")`, with line numbers."""
    out = {}
    for n in ast.walk(ast.parse(open(path).read())):
        if not (isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "get" and n.args
                and isinstance(n.args[0], ast.Constant)
                and isinstance(n.args[0].value, str)):
            continue
        f = n.func.value
        # `self.config.get(...)` only — not nested block reads, which are
        # scoped by whatever produced the block and are checked by their own
        # presence in it.
        if isinstance(f, ast.Attribute) and f.attr == "config":
            out.setdefault(n.args[0].value, []).append(n.lineno)
    return out


def test_every_key_exists_or_is_declared_optional():
    used = _config_get_keys(LOOP)
    cfg = yaml.safe_load(open(CFG)) or {}
    unknown = {k: v for k, v in used.items()
               if k not in cfg and k not in OPTIONAL}
    assert not unknown, (
        "config keys read by the loop that do not exist and are not declared "
        "OPTIONAL — each silently returns its default:\n"
        + "\n".join(f"    {k!r} at line(s) {v}" for k, v in sorted(unknown.items()))
        + "\n  Fix the key, or add it to OPTIONAL with the reason it is absent.")
    print(f"  A. {len(used)} top-level keys read; all present in the config "
          f"or declared optional ({len(OPTIONAL)} allowlisted)")


def test_optional_list_is_not_stale():
    cfg = yaml.safe_load(open(CFG)) or {}
    stale = sorted(k for k in OPTIONAL if k in cfg)
    assert not stale, (
        f"OPTIONAL claims {stale} are absent from the config, but they are "
        f"present. An allowlist that is not true stops being read — remove "
        f"these entries so the list keeps meaning something.")
    used = _config_get_keys(LOOP)
    unread = sorted(k for k in OPTIONAL if k not in used)
    assert not unread, (
        f"OPTIONAL lists {unread}, which the loop never reads. Dead entries "
        f"make the next real one easy to miss.")
    print(f"  B. allowlist is live: none of the {len(OPTIONAL)} entries are "
          f"in the config, and all are actually read")


def test_the_audit_catches_a_bogus_key():
    """C. Prove the instrument works, or A is a green light that never turns red."""
    import tempfile
    src = ('class X:\n'
           '    def f(self):\n'
           '        return self.config.get("definitely_not_a_key", 1)\n')
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        tmp = fh.name
    try:
        found = _config_get_keys(tmp)
        assert "definitely_not_a_key" in found, "the AST walk missed a key"
        cfg = yaml.safe_load(open(CFG)) or {}
        assert "definitely_not_a_key" not in cfg and \
               "definitely_not_a_key" not in OPTIONAL
        print(f"  C. a bogus key is detected at line "
              f"{found['definitely_not_a_key'][0]}")
    finally:
        os.unlink(tmp)


def test_the_three_keys_this_found_are_fixed():
    """The regression itself, named so it cannot come back quietly."""
    src = open(LOOP).read()
    for bad, good in (('config.get("symbol_grounding"', "symbolic_grounding"),
                       ('config.get("vision_scaffold"', "llm.vision")):
        assert bad not in src, (
            f'{bad}...) is back — that key does not exist; the value lives '
            f'at {good} and the read silently returns its default')
    assert 'get("llm", {}) or {}).get("vision"' in src, (
        "the forward_actions read no longer goes through llm.vision")
    print("  D. symbol_grounding and vision_scaffold reads are gone; "
          "forward_actions resolves through llm.vision")


if __name__ == "__main__":
    test_every_key_exists_or_is_declared_optional()
    test_optional_list_is_not_stale()
    test_the_audit_catches_a_bogus_key()
    test_the_three_keys_this_found_are_fixed()
    print("[config-keys] ALL PASS")
