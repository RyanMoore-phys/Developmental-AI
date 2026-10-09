"""Evidence-store recovery must not rebuild the whole history as fat dicts.

LIVE INCIDENT (2026-10-09, main): the trainer held 11.8 GB (9.3 RSS + 2.4
swap) on a 16 GB box. One 3.58 GB anonymous block — the largest single item —
was read page-by-page out of the live process via /proc/PID/mem:

  - 218,496 of 218,504 16 KB pages were pymalloc pools, ~25% full
    (~0.9 GB live objects pinning ~3.58 GB);
  - ~8.7M live str objects; ~490k copies EACH of 'content_hash', 'env',
    'episode', 'stream', 'oseq', 't_wall', 'stream-1', 'MineRLTreechop-v0'
    and the episode id '20261008T093108-157972-e0' (pid 157972 = the
    PREVIOUS run, i.e. reloaded from runlogs/foundation_shadow at boot).

Cause: EvidenceStore._recover decoded every line of every chunk, BODY
included, and held them all at once before ingesting only seq/kind/meta/hash.
The freed bodies left arenas pinned by the surviving meta dicts, and
json.loads gave every meta its own copy of every key.

Measured on a 1/12 copy of the live store (47,848 records): retained
474 MB -> 80 MB, peak 513 MB -> 119 MB, identical ingest digest.

Contracts:
  1. _slim keeps exactly seq/kind/meta/hash (no body).
  2. Keys and short values of separately-decoded lines become ONE shared
     object; 64-char content hashes are left alone.
  3. Meta compares equal before and after (no semantic change).
  4. Every recovery decode site goes through _slim (exactly 4).

    PYTHONPATH=. python tests/_store_recovery_memory_smoke.py
"""
import json
import os

from developmental_ai.foundation.experience import store as S

LINE = ('{"seq": %d, "kind": "observation", "part": "dev", "hash": "%s", '
        '"meta": {"env": "MineRLTreechop-v0", "stream": "stream-1", '
        '"episode": "20261008T093108-157972-e0", "oseq": %d, '
        '"channel": "sensor", "content_hash": "%s"}, '
        '"body": {"value": [0.1, 0.2, 0.3]}}')


def _decoded(seq):
    h = ("%064x" % (seq * 7919))
    return json.loads(LINE % (seq, h, seq, h))


a_raw, b_raw = _decoded(1), _decoded(2)
a, b = S._slim(a_raw), S._slim(b_raw)

# 1
assert set(a) == {"seq", "kind", "meta", "hash"}, set(a)
print("  1. _slim drops the body: keys", sorted(a))

# 2
ka = {k: k for k in a["meta"]}
kb = {k: k for k in b["meta"]}
assert all(ka[k] is kb[k] for k in ka), "meta keys not shared"
for k in ("env", "stream", "episode", "channel"):
    assert a["meta"][k] is b["meta"][k], k
assert a["kind"] is b["kind"]
assert not (a_raw["meta"]["env"] is b_raw["meta"]["env"]), \
    "precondition: json.loads should NOT share values (else this test proves nothing)"
assert len(a["meta"]["content_hash"]) == 64
print("  2. keys + short labels shared across lines; json.loads alone does not share them")

# 3
assert a["meta"] == a_raw["meta"] and a["seq"] == 1 and a["hash"] == a_raw["hash"]
print("  3. meta/seq/hash unchanged in value")

# 4
src = open(os.path.join(os.path.dirname(S.__file__), "store.py")).read()
n = src.count("_slim(d)") + src.count("_slim(ck.decode_line(ln)[0])")
assert n == 4, n
print("  4. all 4 recovery decode sites call _slim")

print("[store_recovery_memory_smoke] ALL PASS")
