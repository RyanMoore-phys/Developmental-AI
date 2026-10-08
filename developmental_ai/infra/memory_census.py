"""Memory census: what actually holds the RAM and VRAM of a running agent.

WHY THIS EXISTS (2026-10-07)
    The training host is a 16 GB box. Measured live: the python agent at
    7.57 GB RSS, two MineRL java clients at 1.29 + 1.19 GB, 4.2 GB available
    of 14.6, swap 2-2.5 GB in use. The log line says the replay buffer is
    "25000 transitions per stream (2.0 GB budget)". To decide what can be
    offloaded to the two CPU-only LAN nodes, the 7.57 GB has to be broken down
    into things that are actually movable — and the budget figure is known to
    be wrong (CLAUDE.md §6: "a 2 GB ceiling measured 3.70 GB").

    A budget is a claim; `.nbytes` is a measurement. This module reports both,
    side by side, and the difference.

WHAT IT IS NOT
    MEASUREMENT ONLY. It never writes to the agent, never calls a method that
    mutates state, never touches an RNG, and never initialises CUDA (creating
    a context to ask how much memory is free would itself cost ~300 MB of
    VRAM). Object attributes are read with `object.__getattribute__` on
    `__dict__`, so a proxy class's `__getattr__` (gym wrappers, RPC clients)
    is never triggered. Every section is wrapped: a failure becomes an entry
    in `errors`, never an exception into the training loop.

ACCURACY LABELS
    * replay / modules / optimizers / loose tensors / numpy: EXACT (`nbytes`,
      storage-deduplicated, so a tensor shared by two modules counts once).
    * python_objects: APPROXIMATE (`sys.getsizeof`, first-path attribution,
      node- and time-capped). Good for ranking, not for arithmetic.
    * residual = RSS - accounted: everything this cannot see — the
      interpreter, shared libraries (torch's .so files are ~1 GB on their
      own), the CUDA host-side context, allocator fragmentation, C buffers.

Usage:
    from developmental_ai.infra.memory_census import census
    rec = census(agent)            # fast mode, seconds at most
    rec = census(agent, deep=True) # + every live torch tensor via gc (slow)

The loop calls MemoryCensusHook.tick() from its single metrics emission site
(`_emit_metrics`); see configs/minecraft_skybot.yaml `diagnostics:`.
"""

import collections
import gc
import json
import logging
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = "memory_census/1"

# Walk limits. The walk is attribution, not a guarantee; when a cap trips the
# record says so (`walk.truncated`) instead of silently under-reporting.
_MAX_DEPTH = 10
_MAX_NODES = 3_000_000
_MAX_ITEMS_PER_CONTAINER = 200_000
_MAX_WALK_S = 30.0

# Types whose instance attributes are descended into. Anything else is sized
# shallowly (getsizeof) and not entered: threads, locks, sockets, gym envs,
# HTTP clients. Entering foreign objects is where a read-only walk would stop
# being read-only (properties, __getattr__ forwarding).
_DESCEND_MODULE_PREFIXES = ("developmental_ai",)

# Agent attributes accounted by their own exact sections, not by the walk.
_SPECIAL_ATTRS = ("replay_buffer", "_bg_sampler")


def _torch():
    try:
        import torch
        return torch
    except Exception:          # pragma: no cover - torch is a hard dep here
        return None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _obj_dict(obj) -> Optional[dict]:
    """An object's instance __dict__ without invoking __getattr__."""
    try:
        d = object.__getattribute__(obj, "__dict__")
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _slot_items(obj) -> List[Tuple[str, Any]]:
    out = []
    try:
        for klass in type(obj).__mro__:
            for s in klass.__dict__.get("__slots__", ()) or ():
                if isinstance(s, str) and s not in ("__dict__", "__weakref__"):
                    try:
                        out.append((s, object.__getattribute__(obj, s)))
                    except Exception:
                        pass
    except Exception:
        pass
    return out


def _snapshot(seq_fn, retries: int = 2):
    """list(...) of a container another thread may be mutating."""
    for _ in range(retries + 1):
        try:
            return seq_fn()
        except RuntimeError:       # "changed size during iteration"
            continue
    return None


def _np_owner(a: np.ndarray) -> np.ndarray:
    o = a
    try:
        while isinstance(o.base, np.ndarray):
            o = o.base
    except Exception:
        pass
    return o


def _storage_key(t) -> Tuple[Tuple[str, int], int]:
    """(dedupe key, bytes) for a tensor's underlying storage."""
    try:
        st = t.untyped_storage()
        return (str(t.device), int(st.data_ptr())), int(st.nbytes())
    except Exception:
        try:
            return ((str(t.device), int(t.data_ptr())),
                    int(t.numel() * t.element_size()))
        except Exception:
            return (("?", id(t)), 0)


def _dev_add(d: Dict[str, int], dev: str, n: int) -> None:
    d[dev] = d.get(dev, 0) + int(n)


def _gb(n) -> Optional[float]:
    try:
        return round(float(n) / 1e9, 4)
    except Exception:
        return None


# --------------------------------------------------------------------------
# process / system
# --------------------------------------------------------------------------
def _proc_status() -> Dict[str, int]:
    """Selected /proc/self/status fields, in bytes (Linux only)."""
    out = {}
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                if k in ("VmRSS", "VmHWM", "VmSwap", "VmData", "VmSize",
                         "RssAnon", "RssFile", "RssShmem", "Threads"):
                    parts = v.split()
                    if not parts:
                        continue
                    if k == "Threads":
                        out[k] = int(parts[0])
                    else:
                        out[k] = int(parts[0]) * 1024     # kB -> bytes
    except Exception:
        pass
    return out


def _process_section() -> Dict[str, Any]:
    sec: Dict[str, Any] = {"pid": os.getpid()}
    try:
        import psutil
        p = psutil.Process()
        try:
            mi = p.memory_full_info()      # uss/pss/swap via smaps on Linux
        except Exception:
            mi = p.memory_info()
        for f in ("rss", "vms", "uss", "pss", "swap", "shared", "data"):
            v = getattr(mi, f, None)
            if v is not None:
                sec[f] = int(v)
        try:
            sec["num_threads"] = int(p.num_threads())
        except Exception:
            pass
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        sec["system"] = {"total": int(vm.total),
                         "available": int(vm.available),
                         "used": int(vm.used),
                         "swap_total": int(sm.total),
                         "swap_used": int(sm.used)}
        kids = []
        try:
            for c in p.children(recursive=True):
                try:
                    cm = c.memory_info()
                    kids.append({"pid": c.pid, "name": c.name(),
                                 "rss": int(cm.rss)})
                except Exception:
                    continue
        except Exception:
            pass
        if kids:
            kids.sort(key=lambda k: -k["rss"])
            sec["children"] = kids[:20]
    except Exception as exc:
        sec["psutil_error"] = repr(exc)
    st = _proc_status()
    if st:
        sec["proc_status"] = st
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"):
        try:
            with open(path) as fh:
                sec["cgroup_" + os.path.basename(path)] = fh.read().strip()
        except Exception:
            pass
    return sec


# --------------------------------------------------------------------------
# replay buffer — exact, per column per stream, and budget vs actual
# --------------------------------------------------------------------------
_COLUMNS = ("observations", "proprio", "actions", "rewards", "dones",
            "restarts", "priorities")


def _column_info(col) -> Dict[str, Any]:
    from developmental_ai.world_model.replay_buffer import BlockArray
    if isinstance(col, BlockArray):
        blocks = list(col._blocks)
        return {"kind": "BlockArray", "bytes": int(sum(b.nbytes for b in blocks)),
                "n_blocks": len(blocks), "block_rows": int(col.block),
                "first_block_bytes": int(blocks[0].nbytes) if blocks else 0,
                "dtype": str(np.dtype(col.dtype)),
                "row_shape": list(col.row_shape)}
    if isinstance(col, np.ndarray):
        # A fixed np.zeros buffer: nbytes is VIRTUAL until rows are written
        # (calloc maps zero pages lazily) — labelled so it is not misread.
        return {"kind": "ndarray(virtual-until-touched)",
                "bytes": int(col.nbytes), "dtype": str(col.dtype),
                "row_shape": list(col.shape[1:])}
    return {"kind": type(col).__name__, "bytes": 0}


def _arrays_in_cache(cache) -> int:
    n = 0
    items = _snapshot(lambda: list((cache or {}).values())) or []
    for v in items:
        for x in (v if isinstance(v, (tuple, list)) else (v,)):
            if isinstance(x, np.ndarray):
                n += int(x.nbytes)
    return n


def _stream_info(s, idx: int) -> Dict[str, Any]:
    d = _obj_dict(s) or {}
    cols = {}
    for name in _COLUMNS:
        c = d.get(name)
        if c is None:
            continue
        cols[name] = _column_info(c)
    growable = bool(d.get("_growable", False))
    block = int(d.get("_block", 0) or 0)
    obs_b = 1 if d.get("_obs_uint8") else 4
    # EXACTLY what maybe_grow() passes to MEMORY_BUDGET.request() per grow
    # (replay_buffer.py maybe_grow: obs + actions + 13 B of rew/done/pri/restart).
    need = (block * int(d.get("obs_dim", 0)) * obs_b
            + block * int(d.get("action_dim", 0)) * 4
            + block * (4 + 4 + 4 + 1)) if growable else 0
    proprio_block = (block * int(d.get("proprio_dim", 0) or 0) * 4
                     if growable else 0)
    grows = int(d.get("_grow_events", 0) or 0)
    initial = (sum(ci.get("first_block_bytes", 0) for ci in cols.values())
               if growable else 0)
    total = int(sum(ci.get("bytes", 0) for ci in cols.values()))
    cache_b = _arrays_in_cache(d.get("_starts_cache"))
    return {
        "stream": idx,
        "size": int(d.get("size", 0) or 0),
        "capacity": int(d.get("capacity", 0) or 0),
        "growable": growable,
        "block_rows": block,
        "grow_events": grows,
        "growth_capped": bool(d.get("_growth_capped", False)),
        "columns": cols,
        "total_bytes": total,
        "bytes_per_row": (int(total // max(1, int(d.get("capacity", 0) or 1)))),
        "starts_cache_bytes": cache_b,
        "episode_starts_len": len(d.get("episode_starts") or []),
        # code-derived budget accounting for THIS stream
        "budget_request_per_grow": need,
        "reserved_from_code": grows * need,
        "unbudgeted_initial_bytes": initial,
        "unbudgeted_proprio_in_grows": grows * proprio_block,
    }


def _replay_section(agent) -> Dict[str, Any]:
    ad = _obj_dict(agent) or {}
    buf = ad.get("replay_buffer")
    if buf is None:
        return {"present": False}
    bd = _obj_dict(buf) or {}
    streams = bd.get("streams")
    if not isinstance(streams, list):
        streams = [buf]
    sinfo = [_stream_info(s, i) for i, s in enumerate(streams)]
    actual = int(sum(s["total_bytes"] for s in sinfo))
    cache = int(sum(s["starts_cache_bytes"] for s in sinfo))
    sec: Dict[str, Any] = {"present": True, "type": type(buf).__name__,
                           "num_streams": len(sinfo), "streams": sinfo,
                           "total_bytes": actual,
                           "starts_cache_bytes": cache}
    try:
        from developmental_ai.world_model.replay_buffer import MEMORY_BUDGET
        lim, res = int(MEMORY_BUDGET.limit), int(MEMORY_BUDGET.reserved)
        unb_init = int(sum(s["unbudgeted_initial_bytes"] for s in sinfo))
        unb_prop = int(sum(s["unbudgeted_proprio_in_grows"] for s in sinfo))
        res_code = int(sum(s["reserved_from_code"] for s in sinfo))
        sec["budget"] = {
            "limit_bytes": lim, "reserved_bytes": res,
            "source": str(MEMORY_BUDGET.source),
            # The budget is PROCESS-GLOBAL and never released: a reservation
            # made by a buffer that no longer exists stays counted.
            "reserved_by_live_streams_from_code": res_code,
            "reserved_not_by_live_streams": res - res_code,
        }
        growable = any(s["growable"] for s in sinfo)
        sec["accounting"] = {
            "actual_bytes": actual,
            "budget_reserved_bytes": res if growable else None,
            "undercount_bytes": (actual - res_code) if growable else None,
            "explained": {
                "initial_block_per_stream_never_requested": unb_init,
                "proprio_column_of_grown_blocks_not_in_request": unb_prop,
                "total": unb_init + unb_prop,
            },
            "unexplained_bytes": ((actual - res_code) - (unb_init + unb_prop))
            if growable else None,
            "actual_over_limit": (round(actual / lim, 3) if lim else None),
            "note": ("actual = sum of .nbytes over every column block; the "
                     "budget only ever sees maybe_grow() requests"),
        }
    except Exception as exc:
        sec["budget_error"] = repr(exc)
    return sec


# --------------------------------------------------------------------------
# background sampler queue
# --------------------------------------------------------------------------
def _batch_bytes(item, by_dev: Dict[str, int], seen: set) -> None:
    torch = _torch()
    vals = item.values() if isinstance(item, dict) else (
        item if isinstance(item, (list, tuple)) else [item])
    for v in vals:
        if torch is not None and torch.is_tensor(v):
            k, n = _storage_key(v)
            if k not in seen:
                seen.add(k)
                _dev_add(by_dev, str(v.device), n)
        elif isinstance(v, np.ndarray):
            _dev_add(by_dev, "cpu", int(v.nbytes))


def _sampler_section(agent) -> Dict[str, Any]:
    ad = _obj_dict(agent) or {}
    smp = ad.get("_bg_sampler")
    if smp is None:
        return {"present": False}
    sd = _obj_dict(smp) or {}
    sec: Dict[str, Any] = {"present": True}
    by_dev: Dict[str, int] = {}
    seen: set = set()
    for qname in ("_results", "_jobs"):
        q = sd.get(qname)
        if q is None:
            continue
        try:
            with q.mutex:                 # a read of the deque, nothing more
                items = list(q.queue)
        except Exception:
            items = []
        sec[qname.strip("_") + "_depth"] = len(items)
        if qname == "_results":
            sec["max_prefetch"] = int(getattr(q, "maxsize", 0) or 0)
            for it in items:
                if isinstance(it, Exception):
                    continue
                _batch_bytes(it, by_dev, seen)
    sec["queued_batch_bytes_by_device"] = by_dev
    sec["batch_size"] = sd.get("batch_size")
    sec["seq_len"] = sd.get("seq_len")
    sec["cpu_assembly"] = sd.get("_cpu_assembly")
    return sec


# --------------------------------------------------------------------------
# the object walk: modules, optimizers, loose tensors, numpy, python objects
# --------------------------------------------------------------------------
class _Walk:
    def __init__(self, deadline: float):
        torch = _torch()
        self.torch = torch
        self.nn_internal = set()
        if torch is not None:
            try:
                self.nn_internal = set(vars(torch.nn.Module()).keys())
            except Exception:
                pass
        self.deadline = deadline
        self.seen_ids: set = set()
        self.seen_storage: set = set()
        self.seen_np: set = set()
        self.nodes = 0
        self.truncated: List[str] = []
        self.modules: List[Dict[str, Any]] = []
        self.optimizers: List[Dict[str, Any]] = []
        self.loose: Dict[str, Dict[str, Any]] = {}      # top -> {dev: bytes}
        self.numpy: Dict[str, Dict[str, int]] = {}      # top -> {bytes, n}
        self.py: Dict[str, Dict[str, int]] = {}         # top -> {bytes, n}
        self.big: List[Tuple[int, str, str]] = []       # (len, path, type)
        self.groups: Dict[Tuple[str, str], List[int]] = {}  # loose arrays
        self.errors = 0

    # ---- leaf accounting --------------------------------------------------
    def _group(self, path: str, dev: str, n: int) -> None:
        # numeric indices collapsed so 500 skill copies are one row
        g = re.sub(r"\[\d+\]", "[*]", path)
        e = self.groups.setdefault((g, dev), [0, 0])
        e[0] += int(n)
        e[1] += 1

    def _tensor(self, t, top: str, path: str = "") -> None:
        k, n = _storage_key(t)
        if k in self.seen_storage:
            return
        self.seen_storage.add(k)
        self._group(path, "torch:" + str(t.device), n)
        d = self.loose.setdefault(top, {"by_device": {}, "count": 0})
        _dev_add(d["by_device"], str(t.device), n)
        d["count"] += 1

    def _ndarray(self, a: np.ndarray, top: str, path: str = "") -> None:
        o = _np_owner(a)
        if id(o) in self.seen_np:
            return
        self.seen_np.add(id(o))
        self._group(path, "numpy", o.nbytes)
        d = self.numpy.setdefault(top, {"bytes": 0, "count": 0})
        d["bytes"] += int(o.nbytes)
        d["count"] += 1

    def _pyobj(self, obj, top: str) -> None:
        try:
            n = sys.getsizeof(obj)
        except Exception:
            n = 0
        d = self.py.setdefault(top, {"approx_bytes": 0, "objects": 0})
        d["approx_bytes"] += int(n)
        d["objects"] += 1

    def _module(self, m, path: str, top: str, stack: list, depth: int) -> None:
        rec: Dict[str, Any] = {"path": path, "type": type(m).__name__}
        devs: Dict[str, int] = {}
        p_bytes = b_bytes = g_bytes = uniq = 0
        n_params = 0
        try:
            for p in m.parameters():
                n_params += int(p.numel())
                k, n = _storage_key(p)
                p_bytes += n
                _dev_add(devs, str(p.device), n)
                if k not in self.seen_storage:
                    self.seen_storage.add(k)
                    uniq += n
                g = p.grad
                if g is not None:
                    gk, gn = _storage_key(g)
                    g_bytes += gn
                    _dev_add(devs, str(g.device), gn)
                    if gk not in self.seen_storage:
                        self.seen_storage.add(gk)
                        uniq += gn
            for b in m.buffers():
                k, n = _storage_key(b)
                b_bytes += n
                _dev_add(devs, str(b.device), n)
                if k not in self.seen_storage:
                    self.seen_storage.add(k)
                    uniq += n
        except Exception:
            self.errors += 1
        rec.update({"n_params": n_params, "param_bytes": p_bytes,
                    "buffer_bytes": b_bytes, "grad_bytes": g_bytes,
                    "unique_bytes": uniq, "shared_bytes":
                    max(0, p_bytes + b_bytes + g_bytes - uniq),
                    "by_device": devs, "top": top})
        self.modules.append(rec)
        # Submodules are part of this record; mark them so a second path to
        # one does not produce a duplicate record. Their NON-torch attributes
        # (plain tensors, numpy stats, python lists) are still walked.
        try:
            subs = list(m.modules())
        except Exception:
            subs = [m]
        for sm in subs:
            self.seen_ids.add(id(sm))
            d = _obj_dict(sm) or {}
            for k, v in list(d.items()):
                if k in self.nn_internal:
                    continue
                stack.append((path + "." + k, v, depth + 1, top))

    def _optimizer(self, o, path: str, top: str) -> None:
        devs: Dict[str, int] = {}
        uniq = 0
        n_states = 0
        st = _snapshot(lambda: list(o.state.values())) or []
        for s in st:
            n_states += 1
            vals = _snapshot(lambda: list(s.values())) if isinstance(s, dict) \
                else []
            for v in vals or []:
                if self.torch is not None and self.torch.is_tensor(v):
                    k, n = _storage_key(v)
                    if k not in self.seen_storage:
                        self.seen_storage.add(k)
                        uniq += n
                        _dev_add(devs, str(v.device), n)
        n_params = 0
        try:
            for g in o.param_groups:
                n_params += len(g.get("params", []))
        except Exception:
            pass
        self.optimizers.append({"path": path, "type": type(o).__name__,
                                "state_bytes": uniq, "by_device": devs,
                                "params_with_state": n_states,
                                "params": n_params, "top": top})

    # ---- the walk ---------------------------------------------------------
    def run(self, roots: List[Tuple[str, Any]]) -> None:
        torch = self.torch
        Module = torch.nn.Module if torch is not None else ()
        Optim = torch.optim.Optimizer if torch is not None else ()
        stack = [(name, obj, 0, name) for name, obj in reversed(roots)]
        while stack:
            path, obj, depth, top = stack.pop()
            if obj is None or isinstance(obj, bool):
                continue                       # singletons
            oid = id(obj)
            if oid in self.seen_ids:
                continue
            self.seen_ids.add(oid)
            self.nodes += 1
            if self.nodes > _MAX_NODES:
                self.truncated.append(f"node cap {_MAX_NODES} at {path}")
                return
            if (self.nodes & 0x3FFF) == 0 and time.time() > self.deadline:
                self.truncated.append(f"time cap {_MAX_WALK_S}s at {path}")
                return
            try:
                # issubclass(type(obj)) rather than isinstance: isinstance
                # falls back to obj.__class__, which a proxy can intercept.
                t = type(obj)
                if issubclass(t, np.ndarray):
                    self._ndarray(obj, top, path)
                    continue
                if torch is not None and issubclass(t, torch.Tensor):
                    self._tensor(obj, top, path)
                    continue
                if Module and issubclass(t, Module):
                    self._module(obj, path, top, stack, depth)
                    continue
                if Optim and issubclass(t, Optim):
                    self._optimizer(obj, path, top)
                    continue
                if isinstance(obj, (str, bytes, bytearray, int, float,
                                    complex, np.generic)):
                    self._pyobj(obj, top)
                    continue
                if depth >= _MAX_DEPTH:
                    self._pyobj(obj, top)
                    continue
                if isinstance(obj, dict):
                    self._pyobj(obj, top)
                    items = _snapshot(lambda: list(obj.items()))
                    if items is None:
                        self.errors += 1
                        continue
                    self._note_big(len(items), path, "dict")
                    if len(items) > _MAX_ITEMS_PER_CONTAINER:
                        self.truncated.append(f"items cap at {path}")
                        items = items[:_MAX_ITEMS_PER_CONTAINER]
                    for k, v in items:
                        if isinstance(k, str):
                            if id(k) not in self.seen_ids:
                                self.seen_ids.add(id(k))
                                self._pyobj(k, top)
                            stack.append((path + "[" + k[:40] + "]", v,
                                          depth + 1, top))
                        else:
                            stack.append((path + "[k]", k, depth + 1, top))
                            stack.append((path + "[v]", v, depth + 1, top))
                    continue
                if isinstance(obj, (list, tuple, set, frozenset,
                                    collections.deque)):
                    self._pyobj(obj, top)
                    items = _snapshot(lambda: list(obj))
                    if items is None:
                        self.errors += 1
                        continue
                    self._note_big(len(items), path, type(obj).__name__)
                    if len(items) > _MAX_ITEMS_PER_CONTAINER:
                        self.truncated.append(f"items cap at {path}")
                        items = items[:_MAX_ITEMS_PER_CONTAINER]
                    for i, v in enumerate(items):
                        stack.append((path + "[" + str(i) + "]", v,
                                      depth + 1, top))
                    continue
                self._pyobj(obj, top)
                mod = getattr(type(obj), "__module__", "") or ""
                if not mod.startswith(_DESCEND_MODULE_PREFIXES):
                    continue
                d = _obj_dict(obj)
                kv = (_snapshot(lambda: list(d.items())) or []) if d else []
                if d is not None:
                    self._pyobj(d, top)
                kv += _slot_items(obj)
                for k, v in kv:
                    stack.append((path + "." + str(k), v, depth + 1, top))
            except Exception:
                self.errors += 1

    def _note_big(self, n: int, path: str, kind: str) -> None:
        if n < 1000:
            return
        self.big.append((n, path, kind))
        if len(self.big) > 200:
            self.big.sort(key=lambda x: -x[0])
            del self.big[50:]


def _seed_replay_seen(w: _Walk, agent) -> None:
    """Mark replay/sampler objects and their arrays as already accounted, so
    another reference to the buffer (the async trainer holds one) is not
    counted twice by the walk."""
    ad = _obj_dict(agent) or {}
    for name in _SPECIAL_ATTRS:
        o = ad.get(name)
        if o is None:
            continue
        w.seen_ids.add(id(o))
        od = _obj_dict(o) or {}
        streams = od.get("streams") if isinstance(od.get("streams"), list) \
            else [o]
        for s in streams:
            w.seen_ids.add(id(s))
            sd = _obj_dict(s) or {}
            for c in _COLUMNS:
                col = sd.get(c)
                if col is None:
                    continue
                w.seen_ids.add(id(col))
                blocks = getattr(col, "_blocks", None) \
                    if not isinstance(col, np.ndarray) else [col]
                for b in blocks or []:
                    w.seen_ids.add(id(b))
                    w.seen_np.add(id(_np_owner(b)))
            for c in ("_starts_cache", "episode_starts"):
                if sd.get(c) is not None:
                    w.seen_ids.add(id(sd.get(c)))


def _walk_section(agent, deadline: float) -> Dict[str, Any]:
    w = _Walk(deadline)
    _seed_replay_seen(w, agent)
    ad = _obj_dict(agent) or {}
    roots = [(k, v) for k, v in list(ad.items()) if k not in _SPECIAL_ATTRS]
    t0 = time.time()
    w.run(roots)
    mods = sorted(w.modules, key=lambda r: -r["unique_bytes"])
    opts = sorted(w.optimizers, key=lambda r: -r["state_bytes"])
    by_dev = {"modules": {}, "optimizers": {}, "loose_tensors": {}}
    # unique module bytes per device: recompute from records' unique share,
    # attributed to the module's dominant device
    for r in mods:
        dev = max(r["by_device"], key=r["by_device"].get) \
            if r["by_device"] else "none"
        _dev_add(by_dev["modules"], dev, r["unique_bytes"])
    for r in opts:
        for d, n in r["by_device"].items():
            _dev_add(by_dev["optimizers"], d, n)
    for top, d in w.loose.items():
        for dev, n in d["by_device"].items():
            _dev_add(by_dev["loose_tensors"], dev, n)
    per_top: Dict[str, Dict[str, int]] = {}

    def _pt(top):
        return per_top.setdefault(top, {"module_bytes": 0, "optimizer_bytes": 0,
                                        "loose_tensor_bytes": 0,
                                        "numpy_bytes": 0, "py_approx_bytes": 0})
    for r in mods:
        _pt(r["top"])["module_bytes"] += r["unique_bytes"]
    for r in opts:
        _pt(r["top"])["optimizer_bytes"] += r["state_bytes"]
    for top, d in w.loose.items():
        _pt(top)["loose_tensor_bytes"] += sum(d["by_device"].values())
    for top, d in w.numpy.items():
        _pt(top)["numpy_bytes"] += d["bytes"]
    for top, d in w.py.items():
        _pt(top)["py_approx_bytes"] += d["approx_bytes"]
    for v in per_top.values():
        v["total"] = sum(v.values())
    top_attrs = sorted(per_top.items(), key=lambda kv: -kv[1]["total"])
    w.big.sort(key=lambda x: -x[0])
    return {
        "modules": mods[:80],
        "modules_listed": min(80, len(mods)), "modules_found": len(mods),
        "optimizers": opts,
        "by_device": by_dev,
        "numpy_total_bytes": int(sum(d["bytes"] for d in w.numpy.values())),
        "py_approx_total_bytes": int(sum(d["approx_bytes"]
                                         for d in w.py.values())),
        "top_attributes": [dict(attr=k, **v) for k, v in top_attrs[:40]],
        "largest_containers": [{"len": n, "path": p, "kind": k}
                               for n, p, k in w.big[:25]],
        # Tensors/arrays NOT owned by a module's parameters or buffers —
        # snapshots, rollout storage, skill weight copies, episodic arrays.
        "loose_array_groups": [
            {"path": g, "kind": dev, "bytes": v[0], "count": v[1]}
            for (g, dev), v in sorted(w.groups.items(),
                                      key=lambda kv: -kv[1][0])[:40]],
        "walk": {"nodes": w.nodes, "seconds": round(time.time() - t0, 3),
                 "truncated": w.truncated[:10], "errors": w.errors,
                 "py_bytes_label": "APPROXIMATE (sys.getsizeof, first-path "
                                   "attribution)"},
    }


# --------------------------------------------------------------------------
# CUDA
# --------------------------------------------------------------------------
def _cuda_section() -> Dict[str, Any]:
    torch = _torch()
    if torch is None:
        return {"available": False}
    try:
        if not torch.cuda.is_available():
            return {"available": False}
        # NEVER create a context just to measure: that alone costs VRAM.
        if not torch.cuda.is_initialized():
            return {"available": True, "initialized": False}
        devs = []
        for i in range(torch.cuda.device_count()):
            d: Dict[str, Any] = {"index": i}
            try:
                d["name"] = torch.cuda.get_device_name(i)
                d["allocated"] = int(torch.cuda.memory_allocated(i))
                d["reserved"] = int(torch.cuda.memory_reserved(i))
                d["max_allocated"] = int(torch.cuda.max_memory_allocated(i))
                d["max_reserved"] = int(torch.cuda.max_memory_reserved(i))
                free, total = torch.cuda.mem_get_info(i)
                d["free"], d["total"] = int(free), int(total)
                # device-wide used minus this process's caching allocator =
                # CUDA context + cuDNN workspace + OTHER processes (Ollama,
                # the VirtualGL'd java clients).
                d["used_outside_torch_allocator"] = int(
                    total - free - d["reserved"])
                ms = torch.cuda.memory_stats(i)
                for k in ("active_bytes.all.current",
                          "inactive_split_bytes.all.current",
                          "num_alloc_retries", "num_ooms"):
                    if k in ms:
                        d[k] = int(ms[k])
            except Exception as exc:
                d["error"] = repr(exc)
            devs.append(d)
        return {"available": True, "initialized": True, "devices": devs}
    except Exception as exc:
        return {"available": None, "error": repr(exc)}


# --------------------------------------------------------------------------
# deep mode
# --------------------------------------------------------------------------
def _deep_section() -> Dict[str, Any]:
    """Every live torch tensor the garbage collector can see. SLOW (walks
    every gc-tracked object) and only on demand. numpy arrays are not
    gc-tracked, so they cannot be counted this way."""
    torch = _torch()
    if torch is None:
        return {"error": "no torch"}
    t0 = time.time()
    seen: set = set()
    by_dev: Dict[str, int] = {}
    shapes: Dict[str, List[int]] = {}
    n = 0
    T = torch.Tensor
    for o in gc.get_objects():
        try:
            if not issubclass(type(o), T):    # no __class__ lookups
                continue
        except Exception:
            continue
        n += 1
        k, b = _storage_key(o)
        if k in seen:
            continue
        seen.add(k)
        dev = str(o.device)
        _dev_add(by_dev, dev, b)
        key = f"{dev}:{str(o.dtype).replace('torch.', '')}{list(o.shape)}"
        s = shapes.setdefault(key, [0, 0])
        s[0] += 1
        s[1] += b
    top = sorted(shapes.items(), key=lambda kv: -kv[1][1])[:30]
    return {"label": "all gc-visible torch tensors, storage-deduplicated",
            "tensors": n, "storages": len(seen), "by_device": by_dev,
            "top_shapes": [{"shape": k, "count": v[0], "bytes": v[1]}
                           for k, v in top],
            "seconds": round(time.time() - t0, 3)}


# --------------------------------------------------------------------------
# public entry
# --------------------------------------------------------------------------
def census(agent, deep: bool = False,
           max_walk_s: float = _MAX_WALK_S) -> Dict[str, Any]:
    """A read-only memory breakdown of `agent`. Never raises."""
    t0 = time.time()
    rec: Dict[str, Any] = {"schema": SCHEMA, "wall_time": round(t0, 3),
                           "deep": bool(deep)}
    errors: Dict[str, str] = {}
    try:
        rec["total_timesteps"] = int((_obj_dict(agent) or {}).get(
            "total_timesteps", 0) or 0)
    except Exception:
        pass
    for name, fn in (("process", _process_section),
                     ("replay", lambda: _replay_section(agent)),
                     ("sampler", lambda: _sampler_section(agent)),
                     ("objects", lambda: _walk_section(
                         agent, time.time() + float(max_walk_s))),
                     ("cuda", _cuda_section)):
        try:
            rec[name] = fn()
        except Exception as exc:
            errors[name] = repr(exc)
            rec[name] = {"error": repr(exc)}
    if deep:
        try:
            rec["deep_tensors"] = _deep_section()
        except Exception as exc:
            errors["deep_tensors"] = repr(exc)
    try:
        rec["summary"] = _summary(rec)
    except Exception as exc:
        errors["summary"] = repr(exc)
    rec["errors"] = errors
    rec["census_seconds"] = round(time.time() - t0, 3)
    try:
        return _plain(rec)
    except Exception:
        return rec


def _plain(o):
    """Plain JSON types only (numpy scalars leak in from buffer attrs)."""
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set, frozenset)):
        return [_plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    return repr(o)


def _summary(rec: Dict[str, Any]) -> Dict[str, Any]:
    pr = rec.get("process") or {}
    rp = rec.get("replay") or {}
    ob = rec.get("objects") or {}
    sm = rec.get("sampler") or {}
    bd = ob.get("by_device") or {}

    def _on(d, cpu):
        return int(sum(v for k, v in (d or {}).items()
                       if (k == "cpu") == cpu))
    cpu = {
        "replay": int(rp.get("total_bytes", 0) or 0)
        + int(rp.get("starts_cache_bytes", 0) or 0),
        "modules": _on(bd.get("modules"), True),
        "optimizers": _on(bd.get("optimizers"), True),
        "loose_tensors": _on(bd.get("loose_tensors"), True),
        "numpy_outside_replay": int(ob.get("numpy_total_bytes", 0) or 0),
        "python_objects_approx": int(ob.get("py_approx_total_bytes", 0) or 0),
        "sampler_queue": _on(sm.get("queued_batch_bytes_by_device"), True),
    }
    gpu = {
        "modules": _on(bd.get("modules"), False),
        "optimizers": _on(bd.get("optimizers"), False),
        "loose_tensors": _on(bd.get("loose_tensors"), False),
        "sampler_queue": _on(sm.get("queued_batch_bytes_by_device"), False),
    }
    acc = sum(cpu.values())
    rss = pr.get("rss")
    st = pr.get("proc_status") or {}
    anon = st.get("RssAnon")
    out = {"cpu_accounted_bytes": cpu, "cpu_accounted_total": acc,
           "rss": rss, "rss_anon": anon, "swap": pr.get("swap",
                                                        st.get("VmSwap")),
           "residual_rss_minus_accounted": (rss - acc) if rss else None,
           "residual_anon_minus_accounted": (anon - acc) if anon else None,
           "residual_note": "interpreter, shared libs, CUDA host context, "
                            "allocator fragmentation, C buffers; RSS excludes "
                            "swapped-out pages, so accounted can exceed RSS "
                            "under swap",
           "gpu_accounted_bytes": gpu, "gpu_accounted_total": sum(gpu.values())}
    cu = rec.get("cuda") or {}
    for d in cu.get("devices") or []:
        if "allocated" in d:
            out["gpu_allocated_minus_accounted"] = int(
                d["allocated"] - out["gpu_accounted_total"])
            out["gpu_note"] = ("allocated - accounted = activations, autograd "
                               "graph, transient batches; reserved - allocated "
                               "= caching-allocator slack")
            break
    return out


# --------------------------------------------------------------------------
# the loop hook
# --------------------------------------------------------------------------
class MemoryCensusHook:
    """Throttled writer called from the loop's single metrics emission site.

    cfg (the `diagnostics:` block):
        memory_census_every_s: seconds between censuses (0 = interval off)
        memory_census_path:    jsonl file; its directory also holds the
                               on-demand trigger file MEMCENSUS
        memory_census_deep:    include the slow gc tensor pass (default off)
        memory_census_max_bytes: rotate to <path>.1 past this size

    With no path configured the hook does nothing at all. A census is written
    at the first tick (the startup baseline), then at most every N seconds;
    `touch <dir>/MEMCENSUS` forces one at the next tick (the file is removed;
    a `deep` trigger file contents runs the deep pass). Never raises.
    """

    TRIGGER = "MEMCENSUS"

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        # .get() on the passed mapping, never dict(cfg): a TrackedConfig
        # records reads only through .get/[] (infra/config_echo.py).
        cfg = cfg if cfg is not None else {}
        self.every_s = float(cfg.get("memory_census_every_s", 0) or 0)
        self.path = str(cfg.get("memory_census_path", "") or "")
        self.deep = bool(cfg.get("memory_census_deep", False))
        self.max_bytes = int(cfg.get("memory_census_max_bytes", 64 << 20))
        self.last_t: Optional[float] = None
        self.written = 0
        self.failures = 0
        self._warned = False

    @property
    def enabled(self) -> bool:
        return bool(self.path)

    @property
    def trigger_path(self) -> str:
        return os.path.join(os.path.dirname(self.path) or ".", self.TRIGGER)

    def _consume_trigger(self) -> Tuple[bool, bool]:
        tp = self.trigger_path
        if not os.path.exists(tp):
            return False, False
        deep = False
        try:
            with open(tp) as fh:
                deep = "deep" in fh.read(64).lower()
        except Exception:
            pass
        try:
            os.remove(tp)
        except Exception:
            pass
        return True, deep

    def tick(self, agent, now: Optional[float] = None) -> bool:
        """Maybe write a census. True if one was written."""
        if not self.enabled:
            return False
        try:
            now = time.time() if now is None else float(now)
            forced, deep = self._consume_trigger()
            due = (self.last_t is None
                   or (self.every_s > 0 and now - self.last_t >= self.every_s))
            if not (forced or due):
                return False
            rec = census(agent, deep=(self.deep or deep))
            rec["trigger"] = "file" if forced else (
                "startup" if self.last_t is None else "interval")
            self.last_t = now
            self._append(rec)
            self.written += 1
            return True
        except Exception as exc:
            self.failures += 1
            if not self._warned:
                self._warned = True
                logger.warning("memory census failed (further failures "
                               "silent): %r", exc)
            return False

    def _append(self, rec: Dict[str, Any]) -> None:
        d = os.path.dirname(self.path)
        if d:
            os.makedirs(d, exist_ok=True)
        try:
            if os.path.getsize(self.path) > self.max_bytes:
                os.replace(self.path, self.path + ".1")
        except OSError:
            pass
        line = json.dumps(rec, default=_json_default, separators=(",", ":"))
        with open(self.path, "a") as fh:
            fh.write(line + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except Exception:
                pass


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return repr(o)
