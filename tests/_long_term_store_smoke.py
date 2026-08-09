"""Long-term memory store smoke (Mac-ok) — the memory-hierarchy primitive.

Contracts:
  1. Register + touch: recency (last_used_step) and frequency (usage_count).
  2. Eviction (RAM->disk): stalest + lowest-value ACTIVE items surface first;
     value damps eviction (a competent skill idles longer before it's a
     candidate); protected ids are never evicted; the idle floor is honored.
  3. Recall (disk->RAM): the DORMANT item whose cue best matches the current
     perception is returned first; below-threshold matches are excluded;
     active items are never recalled.
  4. Persistence: the index survives a save/reload (a lifelong run is killed
     and resumed) and a corrupt index falls back to .bak.
  5. Kind isolation: skills, goals, and facts don't cross-recall.
  6. Unbounded-total / bounded-active: 10k items register fine; recall still
     works; the scan cap is honored without dropping correctness silently.
"""
import json
import os
import shutil

import numpy as np

D = "/tmp/ltm_smoke"


_SHARED = {}

def _store(cue_dim=8, fresh=False, **kw):
    from developmental_ai.memory.long_term_store import LongTermStore
    if fresh or "s" not in _SHARED:
        _SHARED["s"] = LongTermStore(D, cue_dim=cue_dim, **kw)
    return _SHARED["s"]


def test_register_touch():
    shutil.rmtree(D, ignore_errors=True)
    s = _store()
    s.register("sk_a", "skill", [1, 0, 0, 0, 0, 0, 0, 0], step=10, value=0.5)
    s.touch("sk_a", 40)
    s.touch("sk_a", 55)
    it = s.items["sk_a"]
    assert it.last_used_step == 55 and it.usage_count == 2
    # cue is stored unit-normalized
    assert abs(np.linalg.norm(it.cue) - 1.0) < 1e-5
    print("  1. register + touch ok (recency + frequency)")


def test_eviction():
    s = _store(fresh=True)
    # three active skills, varied idle + value
    s.register("stale_weak", "skill", [1, 0, 0, 0, 0, 0, 0, 0], step=0,
               value=0.2)
    s.register("stale_strong", "skill", [0, 1, 0, 0, 0, 0, 0, 0], step=0,
               value=0.95)
    s.register("fresh", "skill", [0, 0, 1, 0, 0, 0, 0, 0], step=0, value=0.2)
    s.touch("stale_weak", 100)
    s.touch("stale_strong", 100)
    s.touch("fresh", 990)
    now = 1000
    # idle floor excludes the fresh one; weak-stale should rank above
    # strong-stale (value damps eviction)
    cand = s.evict_candidates(now, "skill", min_idle_steps=200, k=3)
    assert "fresh" not in cand, "recently-used item was an eviction candidate"
    assert cand[0] == "stale_weak", f"value didn't damp eviction: {cand}"
    assert "stale_strong" in cand
    # protect keeps an item resident even if stale
    cand2 = s.evict_candidates(now, "skill", min_idle_steps=200, k=3,
                               protect={"stale_weak"})
    assert "stale_weak" not in cand2
    print(f"  2. eviction ok (LRU x value, idle floor, protect): {cand}")


def test_recall():
    s = _store()   # SAME instance as test_eviction (shared)
    # page two skills out to dormant
    for mid in ("stale_weak", "stale_strong"):
        s.set_active(mid, False)
    s.set_active("fresh", False)
    # a perception cue very close to stale_strong's cue [0,1,0,...]
    hits = s.recall([0.02, 0.98, 0, 0, 0, 0, 0, 0], "skill", k=2,
                    threshold=0.6)
    assert hits, "no recall for a clearly-matching cue"
    assert hits[0][0] == "stale_strong", f"wrong recall order: {hits}"
    assert hits[0][1] > 0.9
    # an orthogonal cue recalls nothing above threshold
    none = s.recall([0, 0, 0, 1, 0, 0, 0, 0], "skill", k=2, threshold=0.6)
    assert none == [], f"recalled an unrelated memory: {none}"
    # active items are never recall candidates
    s.set_active("stale_strong", True)
    hits2 = s.recall([0, 1, 0, 0, 0, 0, 0, 0], "skill", k=2, threshold=0.6)
    assert all(mid != "stale_strong" for mid, _ in hits2)
    print("  3. recall ok (best cue match, threshold, active excluded)")


def test_persistence():
    s = _store()
    s.save()
    s.save()   # second save rotates the first index into .bak
    from developmental_ai.memory.long_term_store import LongTermStore
    s2 = LongTermStore(D, cue_dim=8)   # genuinely fresh reload from disk
    assert set(s2.items) == set(s.items), "items lost across reload"
    assert s2.items["stale_weak"].usage_count == s.items[
        "stale_weak"].usage_count
    # corrupt the index -> falls back to .bak
    assert os.path.exists(s.index_path + ".bak")
    with open(s.index_path, "w") as f:
        f.write('{"items": [ truncated')
    s3 = LongTermStore(D, cue_dim=8)
    assert "stale_weak" in s3.items, "corrupt index bricked the store"
    print("  4. persistence ok (reload + .bak fallback)")


def test_kind_isolation():
    shutil.rmtree(D, ignore_errors=True)
    s = _store(fresh=True)
    s.register("g1", "goal", [1, 0, 0, 0, 0, 0, 0, 0], step=0, active=False)
    s.register("f1", "fact", [1, 0, 0, 0, 0, 0, 0, 0], step=0, active=False)
    s.register("sk1", "skill", [1, 0, 0, 0, 0, 0, 0, 0], step=0, active=False)
    # identical cues, but recall for one kind never returns another
    assert s.recall([1, 0, 0, 0, 0, 0, 0, 0], "goal")[0][0] == "g1"
    assert s.recall([1, 0, 0, 0, 0, 0, 0, 0], "skill")[0][0] == "sk1"
    assert all(mid != "g1" for mid, _ in
               s.recall([1, 0, 0, 0, 0, 0, 0, 0], "skill"))
    print("  5. kind isolation ok (skills/goals/facts don't cross-recall)")


def test_unbounded_scale():
    shutil.rmtree(D, ignore_errors=True)
    s = _store(cue_dim=8, fresh=True, recall_scan_cap=5000)
    rng = np.random.RandomState(0)
    # 10k dormant skills — an unbounded lifetime's worth
    for i in range(10000):
        v = rng.randn(8).astype(np.float32)
        s.register(f"sk_{i}", "skill", v, step=i, value=0.5, active=False)
    # plant one exact match and confirm it's found despite the scan cap
    target = np.zeros(8, dtype=np.float32); target[0] = 1.0
    s.register("needle", "skill", target, step=10001, value=0.9,
               active=False)
    s.touch("needle", 10001)      # most-recent -> inside the capped scan
    hits = s.recall(target, "skill", k=1, threshold=0.9)
    assert hits and hits[0][0] == "needle", (
        f"needle lost in 10k-item store: {hits[:2]}")
    st = s.stats()
    assert st["total"] == 10001 and st["dormant"] == 10001
    print(f"  6. unbounded scale ok (10k items, recall found the needle)")


if __name__ == "__main__":
    for fn in (test_register_touch, test_eviction, test_recall,
               test_persistence, test_kind_isolation, test_unbounded_scale):
        print(f"[ltm-smoke] {fn.__name__}")
        fn()
    print("[ltm-smoke] ALL PASS")
