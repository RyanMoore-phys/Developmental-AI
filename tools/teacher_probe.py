#!/usr/bin/env python
"""Contrastive health check for a vision/LLM "teacher" dependency.

WHY THIS EXISTS
    This project once trusted a vision teacher that answered "true" to
    literally every probe it was shown.  Every downstream label, fact and
    reward built on those answers was fiction, and the failure was only
    discovered via an ad-hoc manual image battery run mid-incident.  A
    teacher is a dependency like any other, and dependencies get health
    checks — automated, repeatable, and runnable before a multi-day run
    is bet on the answers.

WHY DISCRIMINATION, NOT ACCURACY
    The metric here is DISCRIMINATION: does the answer actually differ
    between contrastive items?  Accuracy is the wrong health metric
    because a constant answerer scores high accuracy on any unbalanced
    set (a "true"-forever teacher gets 60% on a 3-positive/2-negative
    battery and looks half-alive).  Discrimination for a question is

        mean(P_yes | expect true)  -  mean(P_yes | expect false)

    which is exactly 0.0 for ANY constant answerer regardless of the
    positive/negative balance, 1.0 for a perfect one, and negative for
    an inverted one.  P_yes for a single item is taken as 1.0/0.0 from
    the single parsed bool: queries run at temperature 0, so one sample
    IS the estimate — repeated sampling would just re-read the argmax.

DEFENSIVENESS
    Everything a broken teacher, a broken manifest, or a dead network
    can throw is caught and surfaced as data (per-item "errors" lists,
    verdict FAIL) rather than as an exception: a health check that can
    itself crash a run is worse than no health check.  Items whose
    answer cannot be parsed are EXCLUDED from the means instead of being
    coerced to False — coercion would gift a garbage-spewing teacher
    free discrimination on the negative side.  A side with no scorable
    items leaves discrimination undefined (None) and the verdict FAIL:
    "could not demonstrate discrimination" is a failure, not a pass.

USAGE
    ./venv/bin/python tools/teacher_probe.py <manifest.json> \
        [--model qwen2.5vl:7b] [--host http://localhost:11434] \
        [--fake responses.json]

    Manifest schema:
        {"questions": [{"key": "<name>",
                        "prompt": "<full prompt asking for strict JSON
                                   {\"answer\": bool}>",
                        "items": [{"image": "<path>", "expect": true},
                                  ...]}]}

    Exit code 1 if any question FAILs (0 all pass, 2 usage/load error).
    With --fake, canned response strings are read from a JSON map
    {image_path: response_text} and NO network code is ever constructed
    — the fake path and the ollama path are separate factories, chosen
    before either could open a socket.

Library use (no side effects on import):
    from tools.teacher_probe import probe_question, run_manifest
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from typing import Any, Callable, Dict, List, Optional

# (image_path, prompt) -> raw response text, or None for a query failure.
QueryFn = Callable[[str, str], Optional[str]]

# PASS bar for discrimination.  0.34 sits just above one-flip-in-three
# (1/3 ≈ 0.333): a teacher that separates fewer than roughly a third of
# its contrastive pairs is indistinguishable from noise for our use,
# while any teacher doing honest work on a 3-5 item battery clears it
# with room to spare.
DEFAULT_THRESHOLD: float = 0.34

_TRUE_STRINGS = {"true", "yes", "y", "1"}
_FALSE_STRINGS = {"false", "no", "n", "0"}


# ---- lenient answer parsing -----------------------------------------------

def _coerce_bool(value: Any) -> Optional[bool]:
    """Map the value shapes real teachers actually emit onto a bool.

    Observed in the wild: proper bools, "true"/"false" strings (any
    case), "yes"/"no", and 0/1 numerics.  Anything else is None —
    unparseable, never guessed."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().strip('."\'').lower()
        if s in _TRUE_STRINGS:
            return True
        if s in _FALSE_STRINGS:
            return False
    return None


def _extract_json_object(text: str) -> Optional[Any]:
    """Pull the first JSON value out of possibly-decorated model output.

    Models wrap JSON in code fences and prose despite "strict JSON"
    prompts; refusing to look past the wrapper would misclassify a
    perfectly discriminating teacher as unparseable."""
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        lines = lines[1:]                       # drop ```json / ``` opener
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    lo, hi = s.find("{"), s.rfind("}")
    if 0 <= lo < hi:
        try:
            return json.loads(s[lo:hi + 1])
        except Exception:
            pass
    return None


def parse_answer(text: Optional[str], key: str) -> Optional[bool]:
    """Leniently parse one teacher response into a bool, or None.

    Accepted, in order of preference:
      * {"answer": <bool-ish>}          — the requested strict form
      * {"<question key>": <bool-ish>}  — models often echo the concept
        name from the prompt instead of the literal word "answer"
      * a bare JSON/text bool ("true", "false", "yes", ...)

    <bool-ish> covers bools, "true"/"false"-style strings and 0/1 (see
    `_coerce_bool`).  None means unparseable; the caller EXCLUDES such
    items rather than defaulting them, because a default answer would
    manufacture discrimination a broken teacher does not have."""
    if not text or not isinstance(text, str):
        return None
    obj = _extract_json_object(text)
    if isinstance(obj, dict):
        want = ("answer", key.strip().lower())
        for target in want:
            for k, v in obj.items():
                if isinstance(k, str) and k.strip().lower() == target:
                    got = _coerce_bool(v)
                    if got is not None:
                        return got
        return None
    if obj is not None:                          # bare JSON scalar
        return _coerce_bool(obj)
    return _coerce_bool(text)                    # bare non-JSON token


# ---- query-fn factories ----------------------------------------------------
# Network code lives ONLY inside the closure built by
# `make_ollama_query_fn`; `make_fake_query_fn` shares nothing with it,
# so a --fake run cannot touch the network even by accident.

def make_fake_query_fn(responses: Dict[str, str]) -> QueryFn:
    """Query fn serving canned response strings from {image_path: text}.

    Lookup tries the manifest's literal path first, then the basename —
    manifests are written with host-side paths and replayed on machines
    where only the filename survives.  A miss returns None (a query
    failure, surfaced as data), never a fabricated answer."""
    by_base = {}
    for k, v in responses.items():
        by_base.setdefault(str(k).rsplit("/", 1)[-1], v)

    def query(image_path: str, prompt: str) -> Optional[str]:
        if image_path in responses:
            return responses[image_path]
        return by_base.get(str(image_path).rsplit("/", 1)[-1])

    return query


def make_ollama_query_fn(host: str, model: str,
                         timeout: float = 120.0) -> QueryFn:
    """Query fn hitting ollama's /api/generate with a base64 image.

    format="json" + temperature 0 mirrors the production symbolizer's
    call shape — the probe must measure the teacher AS DEPLOYED, not a
    friendlier configuration of it.  Exceptions propagate to
    `probe_question`, which records them per item; a dead host FAILs
    the battery instead of crashing the caller."""
    import urllib.request                        # stdlib; imported here so

    # the fake path never even loads network machinery.
    url = host.rstrip("/") + "/api/generate"

    def query(image_path: str, prompt: str) -> Optional[str]:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        payload = json.dumps({
            "model": model, "prompt": prompt, "images": [b64],
            "format": "json", "stream": False,
            "options": {"temperature": 0.0, "num_predict": 128},
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return (body.get("response") or "").strip()

    return query


# ---- core measurement ------------------------------------------------------

def probe_question(question: Dict[str, Any], query_fn: QueryFn,
                   threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """Measure one question's discrimination over its contrastive items.

    Returns a plain JSON-able dict (monitoring output is data, not
    objects that can only be inspected in a debugger):

        {"key", "n_pos", "n_neg",          # items that PARSED, per side
         "p_yes_pos", "p_yes_neg",         # mean P_yes per side (or None)
         "discrimination",                 # pos - neg, None if a side empty
         "verdict",                        # "PASS" / "FAIL"
         "items": [{"image","expect","answer","raw"}...],
         "errors": [{"image","kind","detail"}...]}

    Never raises: malformed items, throwing query fns and unparseable
    responses all land in "errors" and cost the item its vote."""
    key = str(question.get("key", "?"))
    result: Dict[str, Any] = {"key": key, "n_pos": 0, "n_neg": 0,
                              "p_yes_pos": None, "p_yes_neg": None,
                              "discrimination": None, "verdict": "FAIL",
                              "items": [], "errors": []}
    prompt = question.get("prompt")
    items = question.get("items")
    if not isinstance(prompt, str) or not isinstance(items, list):
        result["errors"].append({"image": None, "kind": "bad_question",
                                 "detail": "missing prompt or items list"})
        return result

    yes: Dict[bool, List[float]] = {True: [], False: []}
    for it in items:
        image = it.get("image") if isinstance(it, dict) else None
        expect = it.get("expect") if isinstance(it, dict) else None
        rec = {"image": image, "expect": expect, "answer": None, "raw": None}
        result["items"].append(rec)
        if not isinstance(image, str) or not isinstance(expect, bool):
            result["errors"].append({"image": image, "kind": "bad_item",
                                     "detail": "item needs a string 'image' "
                                               "and a bool 'expect'"})
            continue
        try:
            raw = query_fn(image, prompt)
        except Exception as e:                   # dead host, missing file...
            result["errors"].append({"image": image, "kind": "query_failure",
                                     "detail": f"{type(e).__name__}: {e}"})
            continue
        if raw is None:
            result["errors"].append({"image": image, "kind": "query_failure",
                                     "detail": "query returned no response"})
            continue
        rec["raw"] = raw[:200]                   # keep reports bounded
        ans = parse_answer(raw, key)
        if ans is None:
            result["errors"].append({"image": image, "kind": "parse_failure",
                                     "detail": f"unparseable: {raw[:80]!r}"})
            continue
        rec["answer"] = ans
        yes[expect].append(1.0 if ans else 0.0)

    result["n_pos"], result["n_neg"] = len(yes[True]), len(yes[False])
    if yes[True]:
        result["p_yes_pos"] = sum(yes[True]) / len(yes[True])
    if yes[False]:
        result["p_yes_neg"] = sum(yes[False]) / len(yes[False])
    if yes[True] and yes[False]:
        result["discrimination"] = result["p_yes_pos"] - result["p_yes_neg"]
        # tiny epsilon so a threshold met exactly is not lost to float dust
        if result["discrimination"] >= threshold - 1e-9:
            result["verdict"] = "PASS"
    return result


def run_manifest(manifest: Dict[str, Any], query_fn: QueryFn,
                 threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """Probe every question in a manifest.

    Returns {"results": [...], "all_pass": bool, "n_fail": int,
    "exit_code": 0|1, "errors": [...]}.  `exit_code` IS the process
    exit-code contract (1 if any question FAILs) so callers — the CLI
    here, CI wrappers elsewhere — share one definition of failure
    instead of each re-deriving it.  A manifest with no scorable
    questions fails: an empty health check proves nothing.  Never
    raises."""
    out: Dict[str, Any] = {"results": [], "all_pass": False, "n_fail": 0,
                           "exit_code": 1, "errors": []}
    questions = manifest.get("questions") if isinstance(manifest, dict) \
        else None
    if not isinstance(questions, list) or not questions:
        out["errors"].append({"kind": "bad_manifest",
                              "detail": "manifest needs a non-empty "
                                        "'questions' list"})
        return out
    for q in questions:
        try:
            res = probe_question(q if isinstance(q, dict) else {},
                                 query_fn, threshold=threshold)
        except Exception as e:                   # belt and braces: contain
            res = {"key": "?", "n_pos": 0, "n_neg": 0, "p_yes_pos": None,
                   "p_yes_neg": None, "discrimination": None,
                   "verdict": "FAIL", "items": [],
                   "errors": [{"image": None, "kind": "internal",
                               "detail": f"{type(e).__name__}: {e}"}]}
        out["results"].append(res)
    out["n_fail"] = sum(1 for r in out["results"] if r["verdict"] != "PASS")
    out["all_pass"] = out["n_fail"] == 0
    out["exit_code"] = 0 if out["all_pass"] else 1
    return out


# ---- reporting -------------------------------------------------------------

def format_table(report: Dict[str, Any],
                 threshold: float = DEFAULT_THRESHOLD) -> str:
    """Render a run_manifest report as the human table the CLI prints."""
    rows = report.get("results", [])
    kw = max([len("question")] + [len(str(r.get("key"))) for r in rows])
    lines = [f"{'question':<{kw}}  n_pos  n_neg  discrimination  verdict"]
    for r in rows:
        d = r.get("discrimination")
        d_s = f"{d:+.3f}" if d is not None else "  n/a"
        lines.append(f"{str(r.get('key')):<{kw}}  {r.get('n_pos', 0):>5}  "
                     f"{r.get('n_neg', 0):>5}  {d_s:>14}  "
                     f"{r.get('verdict', 'FAIL')}")
        for e in r.get("errors", []):
            lines.append(f"{'':<{kw}}    ! {e.get('kind')}: "
                         f"{e.get('image') or ''} {e.get('detail')}")
    for e in report.get("errors", []):
        lines.append(f"! {e.get('kind')}: {e.get('detail')}")
    n = len(rows)
    n_fail = report.get("n_fail", n)
    lines.append(f"[teacher-probe] {n - n_fail}/{n} questions PASS "
                 f"(discrimination threshold {threshold})")
    return "\n".join(lines)


# ---- CLI -------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="teacher_probe",
        description="Contrastive discrimination health check for a vision "
                    "teacher (PASS requires answers that DIFFER between "
                    "contrastive items, not mere accuracy).")
    ap.add_argument("manifest", help="probe manifest json")
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--fake", default=None, metavar="responses.json",
                    help="canned {image_path: response_text} map; when set, "
                         "no network code is constructed at all")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = ap.parse_args(argv)

    try:
        with open(args.manifest, "r") as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"[teacher-probe] cannot load manifest {args.manifest}: {e}")
        return 2
    if args.fake is not None:
        try:
            with open(args.fake, "r") as f:
                canned = json.load(f)
        except Exception as e:
            print(f"[teacher-probe] cannot load --fake {args.fake}: {e}")
            return 2
        query_fn = make_fake_query_fn(canned)
        print(f"[teacher-probe] FAKE mode: {args.fake} (no network)")
    else:
        query_fn = make_ollama_query_fn(args.host, args.model)
        print(f"[teacher-probe] model={args.model} host={args.host}")

    report = run_manifest(manifest, query_fn, threshold=args.threshold)
    print(format_table(report, threshold=args.threshold))
    return int(report["exit_code"])


if __name__ == "__main__":
    sys.exit(main())
