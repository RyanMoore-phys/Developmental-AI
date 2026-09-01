"""Config-echo smoke: the configured-vs-in-effect gap made visible.

Contracts:
  1. Nested get/[] tracking records section + leaf dotted paths on ONE
     shared recorder; sibling views over the same child see each other.
  2. Unread detection: read leaves excluded; an ancestor-section read does
     NOT mark its leaves; empty sections contribute no unread entries.
  3. mark_all_read clears a subtree (root-relative and view-relative);
     unknown prefixes are contained; "" clears everything.
  4. dict(tc) round-trips ==, holds reference-equal (never re-copied,
     never view-wrapped) sub-objects, and yaml.safe_dump/safe_load works;
     C-level access (len/iter/in) is dict-exact and untracked.
  5. .get default path still recorded (root and nested); absent keys never
     appear in the unread report; [] on a missing key still KeyErrors.
  6. List values come back unwrapped and reference-equal; the list is one
     leaf and its interior dicts generate no paths.
  7. effective_summary counts read/unread leaves, truncates at max_unread
     with a visible (+n more), and reports all-consumed after suppression.
  8. Monitoring never raises: garbage input surfaces as error-as-data.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from developmental_ai.infra.config_echo import (
    TrackedConfig, effective_summary, unread_keys)


def _mk():
    data = {
        "llm": {
            "vision": {"weight": 20.0, "interval": 5},
            "enabled": True,
        },
        "seed": 7,
        "stages": [1, 2, {"x": 9}],
        "empty_section": {},
    }
    return data, TrackedConfig(data)


def test_nested_get_tracking():
    data, tc = _mk()
    v = tc.get("llm")
    assert isinstance(v, TrackedConfig), type(v)
    assert v.get("vision")["weight"] == 20.0
    assert {"llm", "llm.vision", "llm.vision.weight"} <= tc.accessed_paths, \
        tc.accessed_paths
    # a second view over the same child is a NEW object on the SAME recorder
    v2 = tc["llm"]
    assert v2 is not v
    assert v2.get("enabled") is True
    assert "llm.enabled" in tc.accessed_paths
    assert "llm.enabled" in v.accessed_paths          # tree-wide, any view
    # accessed_paths is a snapshot, not a live alias
    snap = tc.accessed_paths
    tc.get("seed")
    assert "seed" not in snap and "seed" in tc.accessed_paths


def test_unread_detection():
    data, tc = _mk()
    tc.get("llm").get("vision")["weight"]             # leaf read
    tc["seed"]                                        # leaf read
    u = unread_keys(tc)
    # "llm" and "llm.vision" were read as sections: their OWN paths are
    # recorded but their leaves stay unread; empty_section has no leaves.
    assert u == ["llm.enabled", "llm.vision.interval", "stages"], u


def test_mark_all_read():
    data, tc = _mk()
    tc.mark_all_read("llm.vision")
    u = unread_keys(tc)
    assert "llm.vision.interval" not in u and "llm.vision.weight" not in u, u
    assert "llm.enabled" in u and "seed" in u, u
    tc.mark_all_read("no.such.section")               # contained, no raise
    tc.mark_all_read("seed")                          # prefix naming a leaf
    assert "seed" not in unread_keys(tc)
    tc.mark_all_read()                                # "" = everything
    assert unread_keys(tc) == [], unread_keys(tc)
    # view-relative prefix resolves against the view's own position
    d2, t2 = _mk()
    t2.get("llm").mark_all_read("vision")
    u2 = unread_keys(t2)
    assert "llm.vision.weight" not in u2 and "llm.enabled" in u2, u2


def test_dict_roundtrip_and_yaml():
    data, tc = _mk()
    tc.get("llm").get("vision").get("weight")
    plain = dict(tc)
    assert plain == data
    assert type(plain["llm"]) is dict                 # storage never holds views
    assert plain["llm"] is data["llm"]                # reference-equal, no copy
    v = tc["llm"]
    assert dict.__getitem__(v, "vision") is data["llm"]["vision"]
    dumped = yaml.safe_dump(plain)
    assert yaml.safe_load(dumped) == data
    # C-level access: dict-exact behaviour, recorded nothing
    assert len(tc) == len(data) and set(tc) == set(data) and "seed" in tc
    assert "stages" not in tc.accessed_paths          # documented blind spot


def test_get_default_recorded():
    data, tc = _mk()
    assert tc.get("missing", 3) == 3
    assert "missing" in tc.accessed_paths
    v = tc.get("llm")
    assert v.get("nope") is None
    assert "llm.nope" in tc.accessed_paths
    assert all("missing" not in p and "nope" not in p for p in unread_keys(tc))
    try:
        tc["absent"]
        raise AssertionError("KeyError expected")
    except KeyError:
        pass


def test_lists_unwrapped():
    data, tc = _mk()
    s = tc["stages"]
    assert type(s) is list and s is data["stages"], type(s)
    assert s[2] == {"x": 9} and type(s[2]) is dict    # interior dict untouched
    u = unread_keys(tc)
    assert "stages" not in u                          # the list leaf was read
    assert not any(p.startswith("stages.") for p in u), u


def test_effective_summary():
    data, tc = _mk()
    tc.get("seed")
    tc.get("llm").get("vision")["weight"]
    s = effective_summary(tc)
    assert s.startswith("config: 2 keys read, 3 never read: "), s
    assert "llm.enabled" in s and "llm.vision.interval" in s and "stages" in s
    s2 = effective_summary(tc, max_unread=2)
    assert s2.endswith("(+1 more)"), s2
    assert "stages" not in s2                         # truncated, marker shown
    tc.mark_all_read()
    s3 = effective_summary(tc)
    assert s3 == "config: 5 keys read, all keys consumed", s3


def test_monitoring_never_raises():
    s = effective_summary(None)                       # type: ignore[arg-type]
    assert s.startswith("config: echo unavailable ("), s
    u = unread_keys(None)                             # type: ignore[arg-type]
    assert len(u) == 1 and u[0].startswith("<unread-scan failed:"), u
    # self-referential config must not hang or blow the stack
    loop = {"a": 1}
    loop["self"] = loop
    tl = TrackedConfig(loop)
    assert unread_keys(tl) == ["a", "self.a"], unread_keys(tl)


if __name__ == "__main__":
    tests = [
        (test_nested_get_tracking,
         "nested get/[] records section+leaf paths on one shared recorder"),
        (test_unread_detection,
         "unread = present leaves never read; ancestor reads don't count"),
        (test_mark_all_read,
         "mark_all_read clears subtrees, contains bad prefixes, '' clears all"),
        (test_dict_roundtrip_and_yaml,
         "dict(tc) round-trips by reference; yaml works; C-level untracked"),
        (test_get_default_recorded,
         ".get default path recorded; absent keys never reported; [] raises"),
        (test_lists_unwrapped,
         "lists pass through unwrapped and count as one leaf"),
        (test_effective_summary,
         "summary counts leaves, truncates visibly, reports all-consumed"),
        (test_monitoring_never_raises,
         "garbage/cyclic input becomes error-as-data, never an exception"),
    ]
    for i, (fn, contract) in enumerate(tests, 1):
        fn()
        print(f"[infra-config-echo] {i}. {contract}")
    print("[infra-config-echo] ALL PASS")
