"""Config echo — track which configuration keys are ACTUALLY read.

WHY THIS EXISTS. Two measured failures, both invisible by construction:

  * A configured value of 20.0 was silently ignored for its entire life
    because the consuming code read a *different* attribute than the one the
    config file set. Every experiment run with that file believed the value
    was in effect; none of them ever felt it. No error, no warning — the
    config parsed fine and the code ran fine, they just never met.
  * Two config keys shipped in every config file for months with NOTHING
    reading them. Anyone tuning those keys was turning a disconnected dial.

The gap between "configured" and "in effect" cannot be seen by inspecting
either the config file or the code in isolation — it only exists at their
intersection, at runtime. So we instrument the intersection: every ``get``
and ``[]`` on the config records the dotted path it touched, and at any
point (typically end-of-run, or periodically in a long run's log) the
difference between "keys present" and "keys read" is reported as data.
A key that appears in ``unread_keys()`` after a full run is a disconnected
dial — exactly the failure class above, made visible.

DESIGN. ``TrackedConfig`` is a ``dict`` subclass, not a proxy object,
because the config is handed to code that legitimately treats it as a plain
mapping (``dict(cfg)``, ``yaml.safe_dump(dict(cfg))``, ``**cfg``, ``len``,
iteration, ``in``). A proxy would need to fake all of that and would break
the first time some library type-checked it. The cost of the subclass
choice is a documented blind spot: C-level access (iteration, ``items()``,
``dict(cfg)``, ``**cfg``) bypasses our overridden methods and is NOT
tracked. That is acceptable because the failure we are hunting is a key
that is *never* read at all — code that iterates the whole config touches
everything and is precisely the "handed wholesale to a third party" case
``mark_all_read()`` exists for.

Nested sections wrap lazily, on read: ``cfg["a"]`` returns a NEW
``TrackedConfig`` view over the same underlying child dict (values are
shared by reference — never deep-copied), bound to the same shared recorder
with the extended path prefix. Views are ephemeral and never stored back
into the tree, which is what keeps ``dict(cfg)`` yaml-serialisable: the
storage only ever holds the original plain objects.

Defensive stance: this is monitoring. A recording failure must degrade to
plain-dict behaviour, never take the read down with it; the reporting
functions catch everything and return the error AS data (an error string /
an error entry) rather than raising into a multi-day run.
"""

from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

__all__ = ["TrackedConfig", "unread_keys", "effective_summary"]

# Sentinel distinguishing "key absent" from "key maps to None" without a
# second (racy) containment check.
_MISSING = object()


def _walk(data: Any, prefix: str, active: Set[int]) -> Iterator[Tuple[str, bool]]:
    """Yield ``(dotted_path, is_leaf)`` for every key in a nested dict.

    A leaf is any non-dict value — including lists (list contents are
    opaque to tracking, so the list itself is the trackable unit) and
    ``None``. A dict-valued key is yielded as a section (``is_leaf=False``)
    and then descended into; an EMPTY dict section therefore contributes no
    leaves, which is what keeps it out of the unread report — there is no
    dial inside it to be disconnected.

    ``active`` guards against reference cycles by tracking the *current
    recursion stack* only (id added before descent, removed after). A
    global visited-set would be wrong here: YAML anchors legitimately alias
    one dict under several paths, and every one of those paths carries real
    keys that deserve independent read-accounting.
    """
    if id(data) in active:
        return
    active.add(id(data))
    try:
        # list() snapshot: a config mutated concurrently (rare, but this
        # runs inside multi-day processes) must not blow up the walk with
        # "dict changed size during iteration".
        for key, value in list(data.items()):
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                yield path, False
                yield from _walk(value, path, active)
            else:
                yield path, True
    finally:
        active.discard(id(data))


class TrackedConfig(dict):
    """A dict whose ``get``/``[]`` record which dotted paths were read.

    All views over one config tree share a single recorder set, so
    ``accessed_paths`` on any view (root or nested) reports the complete
    picture. Paths are absolute from the root the tree was first wrapped
    at, e.g. ``{"llm", "llm.vision", "llm.vision.weight"}`` after
    ``cfg.get("llm").get("vision").get("weight")``.

    Known blind spot (accepted, see module docstring): C-level dict access
    — iteration, ``items()``/``values()``, ``dict(cfg)``, ``**cfg``,
    ``len``, ``in`` — behaves exactly like dict and records nothing. Code
    that consumes a section that way should call ``mark_all_read()`` on it
    so the report does not cry wolf.
    """

    def __init__(self, data: Dict[str, Any],
                 _recorder: Optional[Set[str]] = None,
                 _prefix: str = "") -> None:
        # Shallow-init from `data`: our hash table gets its own entries but
        # every VALUE is the reference-equal original object. No data is
        # re-copied — a nested view constructed twice from the same child
        # dict shares that child's contents, and dict(view) hands back the
        # original sub-objects, not clones.
        super().__init__(data)
        self._recorder: Set[str] = _recorder if _recorder is not None else set()
        self._prefix: str = _prefix

    # ------------------------------------------------------------------ #
    # tracked read paths                                                 #
    # ------------------------------------------------------------------ #

    def __getitem__(self, key: Any) -> Any:
        path = self._child_path(key)
        # Record BEFORE the lookup: a failed probe (KeyError) is still a
        # read attempt, and recording absent paths is harmless — the unread
        # report only ever considers keys actually present in the data.
        self._record(path)
        return self._wrap(super().__getitem__(key), path)

    def get(self, key: Any, default: Any = None) -> Any:
        path = self._child_path(key)
        self._record(path)
        value = super().get(key, _MISSING)
        if value is _MISSING:
            return default
        return self._wrap(value, path)

    # ------------------------------------------------------------------ #
    # reporting surface                                                  #
    # ------------------------------------------------------------------ #

    @property
    def accessed_paths(self) -> Set[str]:
        """Copy of every dotted path recorded so far, tree-wide.

        A copy, not the live set: callers snapshot this for diffing, and a
        live alias would silently mutate under them as the run keeps
        reading config.
        """
        try:
            return set(self._recorder)
        except Exception:                                 # pragma: no cover
            return set()

    def mark_all_read(self, path_prefix: str = "") -> None:
        """Suppression hatch: mark every leaf under ``path_prefix`` read.

        Exists for sections handed WHOLESALE to third-party code via
        ``dict()`` conversion or ``**`` unpacking — untrackable by
        construction (C-level access) but genuinely consumed. Without this
        hatch every such section would sit in the unread report forever and
        train people to ignore it, which is how the original two dead keys
        survived: the report you stop reading is no report at all.

        ``path_prefix`` is dotted and RELATIVE to this view (empty string
        = everything under this view). Section paths along the way are
        recorded too — a superset of "every leaf", kept so accessed_paths
        stays coherent with how organic reads record ancestors.

        Never raises: an unknown or non-dict prefix is silently contained
        (the section it would have suppressed cannot generate unread noise
        if it does not exist).
        """
        try:
            node: Any = self
            base = self._prefix
            if path_prefix:
                for seg in path_prefix.split("."):
                    if not isinstance(node, dict):
                        return
                    # dict.get unbound: bypasses our tracked override so
                    # navigation does not depend on it, and works on both
                    # the TrackedConfig root and plain child dicts.
                    node = dict.get(node, seg, _MISSING)
                    if node is _MISSING:
                        return
                    base = f"{base}.{seg}" if base else seg
                    self._recorder.add(base)
            if isinstance(node, dict):
                for path, _is_leaf in _walk(node, base, set()):
                    self._recorder.add(path)
            elif base:
                # Prefix named a leaf directly: mark just that leaf.
                self._recorder.add(base)
        except Exception:
            # Monitoring hatch — must never be the thing that kills a run.
            pass

    # ------------------------------------------------------------------ #
    # private helpers                                                    #
    # ------------------------------------------------------------------ #

    def _child_path(self, key: Any) -> str:
        k = key if isinstance(key, str) else str(key)
        return f"{self._prefix}.{k}" if self._prefix else k

    def _record(self, path: str) -> None:
        try:
            self._recorder.add(path)
        except Exception:
            # A broken recorder must degrade tracking, not break the read
            # itself: the config value still flows to the caller.
            pass

    def _wrap(self, value: Any, path: str) -> Any:
        """Wrap dict values as views; pass everything else through raw.

        Lists deliberately pass through unwrapped (their contents are
        consumed positionally, not by key — no dotted path to record), as
        does anything already tracked. A fresh view per read is cheap: it
        shares the child's values by reference and the tree-wide recorder,
        so no state is duplicated beyond one small hash table.
        """
        if isinstance(value, TrackedConfig):
            return value
        if isinstance(value, dict):
            return TrackedConfig(value, _recorder=self._recorder, _prefix=path)
        return value


def unread_keys(cfg: "TrackedConfig") -> List[str]:
    """Sorted dotted paths of LEAF keys present in the data, never accessed.

    Leaves only: a section is not a dial, its keys are. Reading an ancestor
    section does NOT mark its leaves read — that asymmetry is the whole
    point, because "the code fetched the section" says nothing about
    whether it fetched the key inside it (the measured 20.0 failure fetched
    the section fine and then read the wrong attribute off it).

    Never raises. A scan failure comes back as a single describing entry
    rather than an empty list, because an empty list here means "everything
    is consumed" and a crash must not be able to impersonate good news.
    """
    try:
        recorder = getattr(cfg, "_recorder", None) or set()
        prefix = getattr(cfg, "_prefix", "") or ""
        return sorted(path for path, is_leaf in _walk(cfg, prefix, set())
                      if is_leaf and path not in recorder)
    except Exception as exc:
        return [f"<unread-scan failed: {exc.__class__.__name__}: {exc}>"]


def effective_summary(cfg: "TrackedConfig", max_unread: int = 40) -> str:
    """One log line answering "did the config actually reach the code?".

    ``"config: <n_read> keys read, <n_unread> never read: <paths...>"`` —
    or ``"config: <n> keys read, all keys consumed"`` when nothing is
    dangling. ``n_read`` counts LEAF keys present in the data that were
    accessed, not raw recorder entries: the recorder also holds section
    paths and probes of absent keys (``.get`` with a default), and counting
    those would claim reads of dials that do not exist in the file.

    At most ``max_unread`` unread paths are listed (a config with hundreds
    of dead keys is itself the finding; flooding the log hides it), with a
    ``(+n more)`` marker so truncation is visible rather than silent.
    Never raises — a failure is returned as the summary string.
    """
    try:
        recorder = getattr(cfg, "_recorder", None) or set()
        prefix = getattr(cfg, "_prefix", "") or ""
        leaves = [path for path, is_leaf in _walk(cfg, prefix, set()) if is_leaf]
        unread = sorted(p for p in leaves if p not in recorder)
        n_read = len(leaves) - len(unread)
        if not unread:
            return f"config: {n_read} keys read, all keys consumed"
        shown = unread[:max(0, int(max_unread))]
        hidden = len(unread) - len(shown)
        tail = f" (+{hidden} more)" if hidden > 0 else ""
        return (f"config: {n_read} keys read, {len(unread)} never read: "
                f"{', '.join(shown)}{tail}")
    except Exception as exc:
        return (f"config: echo unavailable "
                f"({exc.__class__.__name__}: {exc})")
