"""Teacher-probe smoke (no network; fake canned responses only).

Contracts:
  1. A constant-true teacher scores discrimination exactly 0.0 -> FAIL,
     regardless of the positive/negative balance (the incident case:
     accuracy would have scored it 60%).
  2. A perfect teacher scores +1.0 -> PASS; a perfectly INVERTED teacher
     scores -1.0 -> FAIL (sign carries the pathology).
  3. Partial discrimination lands where the mean formula says: 2/3 vs 0
     PASSes at 0.34, 1/3 vs 0 FAILs just under it.
  4. Lenient parsing: question-key alias, "true"/"false" strings, bare
     bools, code fences, prose wrappers — all resolve; garbage does not.
  5. Failures are data, never exceptions: throwing query fns, missing
     canned entries and unparseable text land in "errors", cost the item
     its vote, and an empty side leaves discrimination None -> FAIL.
  6. run_manifest carries the exit-code contract: any FAIL -> 1, all
     PASS -> 0, malformed manifest contained -> 1.
  7. The shipped example manifest is structurally valid and PASSes end
     to end under a perfect fake teacher keyed by its host-side paths.
  8. The CLI in --fake mode exits with run_manifest's code and never
     touches the network (host points at a dead port).
"""
import json
import os
import subprocess
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from tools.teacher_probe import (DEFAULT_THRESHOLD, format_table,
                                 make_fake_query_fn, parse_answer,
                                 probe_question, run_manifest)

_PROMPT = 'Is it there? Strict JSON: {"answer": true} or {"answer": false}.'


def _question(key="thing_visible", n_pos=3, n_neg=2):
    # paths are namespaced by key: the fake query map is keyed by image
    # path, so two questions sharing paths would share canned answers
    items = [{"image": f"/x/{key}/pos_{i}.png", "expect": True}
             for i in range(n_pos)]
    items += [{"image": f"/x/{key}/neg_{i}.png", "expect": False}
              for i in range(n_neg)]
    return {"key": key, "prompt": _PROMPT, "items": items}


def _canned(question, answer_for):
    """Build a canned-response map from item path -> bool|raw-string."""
    out = {}
    for it in question["items"]:
        v = answer_for(it)
        out[it["image"]] = v if isinstance(v, str) \
            else json.dumps({"answer": v})
    return out


def test_constant_true_fails():
    q = _question()
    fake = make_fake_query_fn(_canned(q, lambda it: True))
    r = probe_question(q, fake)
    assert r["n_pos"] == 3 and r["n_neg"] == 2, r
    assert r["discrimination"] == 0.0, r["discrimination"]
    assert r["verdict"] == "FAIL", r
    # the trap this tool exists to spring: accuracy on the same battery
    # would have read 3/5 = 60% and looked half-alive
    acc = sum(1 for it in r["items"] if it["answer"] == it["expect"]) / 5
    assert acc == 0.6, acc
    print(f"  1. constant-true: discrimination {r['discrimination']} FAIL "
          f"(accuracy would have said {acc:.0%})")


def test_perfect_and_inverted():
    q = _question()
    perfect = make_fake_query_fn(_canned(q, lambda it: it["expect"]))
    r = probe_question(q, perfect)
    assert r["discrimination"] == 1.0 and r["verdict"] == "PASS", r
    inverted = make_fake_query_fn(_canned(q, lambda it: not it["expect"]))
    ri = probe_question(q, inverted)
    assert ri["discrimination"] == -1.0 and ri["verdict"] == "FAIL", ri
    print(f"  2. perfect {r['discrimination']:+.1f} PASS, "
          f"inverted {ri['discrimination']:+.1f} FAIL")


def test_partial_discrimination_threshold():
    q = _question()                       # pos_0..2, neg_0..1
    # yes on 2 of 3 positives, no on all negatives -> 0.667 PASS
    two_of_three = make_fake_query_fn(_canned(
        q, lambda it: it["expect"] and not it["image"].endswith("pos_2.png")))
    r = probe_question(q, two_of_three)
    assert abs(r["discrimination"] - 2 / 3) < 1e-9, r["discrimination"]
    assert r["verdict"] == "PASS", r
    # yes on only 1 of 3 positives -> 0.333, just under the 0.34 bar
    one_of_three = make_fake_query_fn(_canned(
        q, lambda it: it["image"].endswith("pos_0.png")))
    r1 = probe_question(q, one_of_three)
    assert abs(r1["discrimination"] - 1 / 3) < 1e-9, r1["discrimination"]
    assert r1["verdict"] == "FAIL", r1
    print(f"  3. partial: 2/3 -> {r['discrimination']:+.3f} PASS, "
          f"1/3 -> {r1['discrimination']:+.3f} FAIL at {DEFAULT_THRESHOLD}")


def test_lenient_parsing():
    cases = [
        ('{"answer": true}', True),
        ('{"answer": "true"}', True),
        ('{"answer": "False"}', False),           # capitalised string
        ('{"answer": "yes"}', True),
        ('{"answer": 0}', False),                 # 0/1 numeric
        ('{"thing_visible": true}', True),        # question-key alias
        ('{"Thing_Visible": "no"}', False),       # alias, case-insensitive
        ('```json\n{"answer": false}\n```', False),
        ('Sure! Here you go: {"answer": true} Hope that helps.', True),
        ('true', True),                           # bare token
        ('"false"', False),
        ('{"answer": true, "confidence": 0.9}', True),
        ('{"confidence": 0.9}', None),            # no answer anywhere
        ('the trunk is clearly visible', None),   # prose, not an answer
        ('', None),
        (None, None),
    ]
    for raw, want in cases:
        got = parse_answer(raw, "thing_visible")
        assert got is want, (raw, got, want)
    print(f"  4. lenient parsing: {len(cases)} variants ok "
          f"(alias, strings, fences, prose, garbage->None)")


def test_failures_are_data():
    q = _question(n_pos=2, n_neg=2)
    # negatives: one canned garbage, one missing from the map entirely;
    # positives answer honestly
    canned = {"/x/thing_visible/pos_0.png": '{"answer": true}',
              "/x/thing_visible/pos_1.png": '{"answer": true}',
              "/x/thing_visible/neg_0.png": "%%% not json or a bool %%%"}
    r = probe_question(q, make_fake_query_fn(canned))
    assert r["n_pos"] == 2 and r["n_neg"] == 0, r
    assert r["discrimination"] is None and r["verdict"] == "FAIL", r
    kinds = sorted(e["kind"] for e in r["errors"])
    assert kinds == ["parse_failure", "query_failure"], kinds
    # a query fn that RAISES is contained per item, not propagated
    def bomb(image, prompt):
        raise ConnectionError("host unreachable")
    rb = probe_question(q, bomb)
    assert rb["verdict"] == "FAIL" and len(rb["errors"]) == 4, rb
    assert all(e["kind"] == "query_failure" for e in rb["errors"])
    # one-sided battery (no negatives at all): undefined, FAIL as data
    q1 = _question(n_pos=3, n_neg=0)
    r1 = probe_question(q1, make_fake_query_fn(_canned(q1, lambda i: True)))
    assert r1["discrimination"] is None and r1["verdict"] == "FAIL", r1
    # and format_table renders every shape above without raising
    txt = format_table({"results": [r, rb, r1], "n_fail": 3, "errors": []})
    assert "n/a" in txt and "FAIL" in txt
    print("  5. failures are data: garbage+missing+raise+one-sided all "
          "FAIL with errors listed, table renders")


def test_run_manifest_exit_code():
    good = _question(key="alive")
    bad = _question(key="stuck")
    canned = {}
    canned.update(_canned(good, lambda it: it["expect"]))
    # constant-true on the second question -> its FAIL must set exit 1
    for it in bad["items"]:
        canned[it["image"]] = '{"answer": true}'
    rep = run_manifest({"questions": [good, bad]},
                       make_fake_query_fn(canned))
    assert rep["n_fail"] == 1 and rep["exit_code"] == 1, rep
    assert not rep["all_pass"]
    rep2 = run_manifest({"questions": [good]},
                        make_fake_query_fn(canned))
    assert rep2["all_pass"] and rep2["exit_code"] == 0, rep2
    # malformed manifests are contained, and an EMPTY check cannot pass
    for bad_m in ({}, {"questions": []}, {"questions": "x"}, "not a dict"):
        r = run_manifest(bad_m, make_fake_query_fn({}))
        assert r["exit_code"] == 1 and r["errors"], (bad_m, r)
    print("  6. exit-code contract: 1 fail -> 1, all pass -> 0, "
          "malformed/empty -> 1 with errors")


def test_example_manifest():
    path = os.path.join(_ROOT, "tools", "probes",
                        "skybot_vision_manifest.example.json")
    with open(path) as f:
        manifest = json.load(f)
    q = manifest["questions"][0]
    assert q["key"] == "trunk_visible"
    assert '{"answer"' in q["prompt"], "prompt must demand strict JSON"
    pos = [i for i in q["items"] if i["expect"]]
    neg = [i for i in q["items"] if not i["expect"]]
    assert len(pos) == 3 and len(neg) == 2, (pos, neg)
    assert all(i["image"].startswith("/tmp/t_") for i in q["items"])
    # a perfect fake teacher keyed by the training host paths PASSes end to end
    canned = {i["image"]: json.dumps({"answer": i["expect"]})
              for i in q["items"]}
    rep = run_manifest(manifest, make_fake_query_fn(canned))
    assert rep["all_pass"] and rep["results"][0]["discrimination"] == 1.0
    print(f"  7. example manifest: {len(pos)}+/{len(neg)}- host-side items, "
          f"perfect fake -> PASS")


def test_cli_fake_mode():
    q = _question(key="cli_check")
    with tempfile.TemporaryDirectory() as d:
        man = os.path.join(d, "m.json")
        fak = os.path.join(d, "f.json")
        with open(man, "w") as f:
            json.dump({"questions": [q]}, f)
        # constant-true canned map -> expect exit 1; host is a DEAD port,
        # so any network attempt in fake mode would error differently
        with open(fak, "w") as f:
            json.dump({it["image"]: '{"answer": true}'
                       for it in q["items"]}, f)
        tool = os.path.join(_ROOT, "tools", "teacher_probe.py")
        p = subprocess.run(
            [sys.executable, tool, man, "--fake", fak,
             "--host", "http://127.0.0.1:1"],
            capture_output=True, text=True, timeout=60)
        assert p.returncode == 1, (p.returncode, p.stdout, p.stderr)
        assert "cli_check" in p.stdout and "FAIL" in p.stdout, p.stdout
        assert "no network" in p.stdout, p.stdout
        # flip to a perfect map -> exit 0
        with open(fak, "w") as f:
            json.dump({it["image"]: json.dumps({"answer": it["expect"]})
                       for it in q["items"]}, f)
        p2 = subprocess.run(
            [sys.executable, tool, man, "--fake", fak,
             "--host", "http://127.0.0.1:1"],
            capture_output=True, text=True, timeout=60)
        assert p2.returncode == 0, (p2.returncode, p2.stdout, p2.stderr)
        assert "PASS" in p2.stdout
        # unreadable manifest -> usage error 2, still no crash
        p3 = subprocess.run(
            [sys.executable, tool, os.path.join(d, "absent.json"),
             "--fake", fak], capture_output=True, text=True, timeout=60)
        assert p3.returncode == 2, (p3.returncode, p3.stdout)
    print("  8. CLI --fake: constant-true exits 1, perfect exits 0, "
          "bad manifest exits 2 — dead host untouched")


if __name__ == "__main__":
    for i, fn in enumerate(
            (test_constant_true_fails, test_perfect_and_inverted,
             test_partial_discrimination_threshold, test_lenient_parsing,
             test_failures_are_data, test_run_manifest_exit_code,
             test_example_manifest, test_cli_fake_mode), 1):
        print(f"[infra-teacher-probe] {i}. {fn.__name__}")
        fn()
    print("[infra-teacher-probe] ALL PASS")
