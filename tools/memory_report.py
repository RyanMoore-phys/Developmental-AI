"""Pretty-print runlogs/memory_census.jsonl (written by infra/memory_census.py).

    PYTHONPATH=. python tools/memory_report.py [path] [--trend N] [--all-modules]

Shows the LATEST census in full (process, replay per stream per column with
budget-vs-actual, modules/optimizers by device, loose arrays, CUDA, residual)
and, with --trend, one line per census for the last N records so growth over
a run is visible. Read-only; no torch import.
"""
import argparse
import json
import os
import sys


def _gb(n):
    if n is None:
        return "    n/a"
    return f"{float(n) / 1e9:7.3f}"


def _mb(n):
    if n is None:
        return "n/a"
    return f"{float(n) / 1e6:,.1f} MB"


def load(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue                    # a torn final line is not fatal
    return out


def show(rec, all_modules=False):
    p = rec.get("process") or {}
    st = p.get("proc_status") or {}
    sysm = p.get("system") or {}
    print(f"== memory census  t={rec.get('wall_time')}  "
          f"steps={rec.get('total_timesteps')}  trigger={rec.get('trigger')}  "
          f"took {rec.get('census_seconds')}s ==")
    print(f"process (GB): rss {_gb(p.get('rss'))}  uss {_gb(p.get('uss'))}  "
          f"pss {_gb(p.get('pss'))}  swap {_gb(p.get('swap', st.get('VmSwap')))}"
          f"  anon {_gb(st.get('RssAnon'))}  file {_gb(st.get('RssFile'))}  "
          f"peak {_gb(st.get('VmHWM'))}")
    if sysm:
        print(f"system (GB):  total {_gb(sysm.get('total'))}  available "
              f"{_gb(sysm.get('available'))}  swap used "
              f"{_gb(sysm.get('swap_used'))} of {_gb(sysm.get('swap_total'))}")
    for c in p.get("children") or []:
        print(f"  child pid {c['pid']:>7} {c['name'][:24]:<24} rss {_gb(c['rss'])}")

    rp = rec.get("replay") or {}
    if rp.get("present"):
        print(f"\nREPLAY ({rp.get('type')}, {rp.get('num_streams')} streams): "
              f"{_gb(rp.get('total_bytes'))} GB actual")
        for s in rp.get("streams") or []:
            print(f"  stream {s['stream']}: size {s['size']:,} / capacity "
                  f"{s['capacity']:,}  blocks of {s['block_rows']:,}  grows "
                  f"{s['grow_events']}  capped={s['growth_capped']}  "
                  f"{s['bytes_per_row']:,} B/row  total {_gb(s['total_bytes'])} GB")
            for name, ci in sorted((s.get("columns") or {}).items(),
                                   key=lambda kv: -kv[1].get("bytes", 0)):
                print(f"      {name:<13} {_gb(ci.get('bytes'))} GB  "
                      f"{ci.get('dtype', ''):<8} row {ci.get('row_shape')} "
                      f"blocks {ci.get('n_blocks', '-')}")
            if s.get("starts_cache_bytes"):
                print(f"      starts-cache {_mb(s['starts_cache_bytes'])}")
        b, a = rp.get("budget") or {}, rp.get("accounting") or {}
        if b:
            print(f"  BUDGET: limit {_gb(b.get('limit_bytes'))} GB, reserved "
                  f"{_gb(b.get('reserved_bytes'))} GB ({b.get('source')})")
        if a and a.get("undercount_bytes") is not None:
            e = a.get("explained") or {}
            print(f"  ACTUAL vs BUDGET: actual {_gb(a.get('actual_bytes'))} GB, "
                  f"undercount {_gb(a.get('undercount_bytes'))} GB = initial "
                  f"blocks {_gb(e.get('initial_block_per_stream_never_requested'))}"
                  f" + proprio-in-grows "
                  f"{_gb(e.get('proprio_column_of_grown_blocks_not_in_request'))}"
                  f" (unexplained {a.get('unexplained_bytes')} B); "
                  f"actual/limit = {a.get('actual_over_limit')}")

    sm = rec.get("sampler") or {}
    if sm.get("present"):
        print(f"\nSAMPLER queue: {sm.get('results_depth')}/{sm.get('max_prefetch')}"
              f" ready, {sm.get('jobs_depth')} pending, bytes "
              f"{sm.get('queued_batch_bytes_by_device')}")

    ob = rec.get("objects") or {}
    mods = ob.get("modules") or []
    print(f"\nMODULES ({ob.get('modules_found')} found; unique bytes, "
          f"storage-deduplicated):")
    for m in (mods if all_modules else mods[:20]):
        dev = ",".join(sorted(m.get("by_device") or {})) or "-"
        print(f"  {_mb(m['unique_bytes']):>12}  {dev:<8} {m['path'][:60]:<60} "
              f"{m['type']}  params {m['n_params']:,}")
    print("OPTIMIZERS (state bytes):")
    for o in ob.get("optimizers") or []:
        dev = ",".join(sorted(o.get("by_device") or {})) or "-"
        print(f"  {_mb(o['state_bytes']):>12}  {dev:<8} {o['path'][:60]}")
    if ob.get("loose_array_groups"):
        print("LOOSE ARRAYS (not module params/buffers; [*] = collapsed index):")
        for g in ob["loose_array_groups"][:20]:
            print(f"  {_mb(g['bytes']):>12}  {g['kind']:<12} x{g['count']:<6} "
                  f"{g['path'][:70]}")
    if ob.get("top_attributes"):
        print("BY AGENT ATTRIBUTE (py bytes approximate):")
        for t in ob["top_attributes"][:15]:
            print(f"  {_mb(t['total']):>12}  {t['attr']:<28} mod "
                  f"{_mb(t['module_bytes'])}  opt {_mb(t['optimizer_bytes'])}  "
                  f"np {_mb(t['numpy_bytes'])}  py~{_mb(t['py_approx_bytes'])}")
    if ob.get("largest_containers"):
        print("LARGEST CONTAINERS (len):")
        for c in ob["largest_containers"][:10]:
            print(f"  {c['len']:>10,}  {c['kind']:<6} {c['path'][:70]}")
    w = ob.get("walk") or {}
    if w.get("truncated"):
        print(f"  WALK TRUNCATED: {w['truncated']}")

    cu = rec.get("cuda") or {}
    print("\nCUDA:", "unavailable" if not cu.get("available") else (
        "not initialised" if not cu.get("initialized") else ""))
    for d in cu.get("devices") or []:
        print(f"  [{d.get('index')}] {d.get('name')}: allocated "
              f"{_gb(d.get('allocated'))}  reserved {_gb(d.get('reserved'))}  "
              f"max_alloc {_gb(d.get('max_allocated'))}  free "
              f"{_gb(d.get('free'))} / {_gb(d.get('total'))}  outside-torch "
              f"{_gb(d.get('used_outside_torch_allocator'))}  ooms "
              f"{d.get('num_ooms')}")

    sm = rec.get("summary") or {}
    if sm:
        print("\nSUMMARY (GB):")
        for k, v in (sm.get("cpu_accounted_bytes") or {}).items():
            print(f"  cpu  {k:<24} {_gb(v)}")
        print(f"  cpu  {'= accounted':<24} {_gb(sm.get('cpu_accounted_total'))}"
              f"   rss {_gb(sm.get('rss'))}  residual "
              f"{_gb(sm.get('residual_rss_minus_accounted'))}")
        for k, v in (sm.get("gpu_accounted_bytes") or {}).items():
            print(f"  gpu  {k:<24} {_gb(v)}")
        if "gpu_allocated_minus_accounted" in sm:
            print(f"  gpu  {'allocated - accounted':<24} "
                  f"{_gb(sm['gpu_allocated_minus_accounted'])}  "
                  f"(activations / transient)")
    dp = rec.get("deep_tensors")
    if dp:
        print(f"\nDEEP: {dp.get('tensors')} tensors, by device "
              f"{dp.get('by_device')}")
        for t in (dp.get("top_shapes") or [])[:10]:
            print(f"  {_mb(t['bytes']):>12} x{t['count']:<5} {t['shape']}")
    if rec.get("errors"):
        print("ERRORS:", rec["errors"])


def trend(recs):
    print(f"\n{'steps':>10} {'rss':>7} {'swap':>7} {'replay':>7} {'budget':>7} "
          f"{'resid':>7} {'cuda_al':>7} {'cuda_rs':>7}  trigger")
    for r in recs:
        p, rp, sm = (r.get("process") or {}), (r.get("replay") or {}), \
            (r.get("summary") or {})
        d = ((r.get("cuda") or {}).get("devices") or [{}])[0]
        print(f"{r.get('total_timesteps', 0):>10} {_gb(p.get('rss'))} "
              f"{_gb(p.get('swap'))} {_gb(rp.get('total_bytes'))} "
              f"{_gb((rp.get('budget') or {}).get('reserved_bytes'))} "
              f"{_gb(sm.get('residual_rss_minus_accounted'))} "
              f"{_gb(d.get('allocated'))} {_gb(d.get('reserved'))}  "
              f"{r.get('trigger')}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", nargs="?", default="runlogs/memory_census.jsonl")
    ap.add_argument("--trend", type=int, default=0,
                    help="also print one line per census for the last N")
    ap.add_argument("--all-modules", action="store_true")
    a = ap.parse_args(argv)
    if not os.path.isfile(a.path):
        print(f"no census file at {a.path} (touch runlogs/MEMCENSUS to "
              f"request one at the next segment)")
        return 1
    recs = load(a.path)
    if not recs:
        print(f"{a.path} has no readable records")
        return 1
    show(recs[-1], all_modules=a.all_modules)
    if a.trend:
        trend(recs[-a.trend:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
