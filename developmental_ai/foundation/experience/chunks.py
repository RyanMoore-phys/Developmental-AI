"""On-disk line format, content hashes and atomic file writes.

LINE FORMAT (one JSON object per line, store format 1)

    {"seq": <int>, "kind": <str>, "part": <str>, "meta": {...},
     "body": {...}, "hash": "<sha256 hex>"}

    `body` is the contracts codec wire dict for contract records
    (observation, action, prediction, evidence) and a plain JSON dict for the
    store's own records (episode_end, interpretation). `meta` is the index
    entry, computed at write time, so the index rebuilds without decoding
    any body. `hash` is sha256 over the canonical JSON of every other key:
    a line that does not reproduce its hash is corrupt, whatever its JSON
    looks like.

CHUNKS
    open-<cid>.jsonl      the partition's append journal; every append is
                          flushed and fsynced. At most one per partition.
    chunk-<cid>.jsonl.gz  a SEALED chunk: the journal's lines gzipped and
                          written once via tmp + fsync + rename. Never
                          modified afterwards; its sha256 is in the manifest.

ATOMIC WRITE = write `<path>.tmp`, flush, fsync, os.replace, fsync the
directory. A crash leaves either the old file or the new one, and at worst a
stray `.tmp` that recovery deletes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import zlib
from typing import Any, Dict, List, Optional, Tuple

STORE_FORMAT = 1
# What a damaged gzip can raise. zlib.error is NOT an OSError: a corrupt
# deflate block got past `except OSError` until the smoke test flipped a byte.
READ_ERRORS = (OSError, EOFError, ValueError, zlib.error)
LINE_KEYS = ("seq", "kind", "part", "meta", "body", "hash")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def line_hash(seq: int, kind: str, part: str, meta: Dict, body: Dict) -> str:
    return hashlib.sha256(canonical(
        {"seq": seq, "kind": kind, "part": part, "meta": meta,
         "body": body}).encode("utf-8")).hexdigest()


def encode_line(seq: int, kind: str, part: str, meta: Dict,
                body: Dict) -> Tuple[str, str]:
    """-> (newline-terminated line, its content hash)."""
    h = line_hash(seq, kind, part, meta, body)
    return canonical({"seq": seq, "kind": kind, "part": part, "meta": meta,
                      "body": body, "hash": h}) + "\n", h


def decode_line(text: str) -> Tuple[Optional[Dict], str]:
    """-> (parsed line, "") if intact, else (None, reason)."""
    if not text.endswith("\n"):
        return None, "truncated (no newline terminator)"
    try:
        d = json.loads(text)
    except ValueError as e:
        return None, f"not JSON: {e}"
    if not isinstance(d, dict) or set(d) != set(LINE_KEYS):
        return None, "line envelope has the wrong keys"
    if (isinstance(d["seq"], bool) or not isinstance(d["seq"], int)
            or not isinstance(d["kind"], str) or not isinstance(d["part"], str)
            or not isinstance(d["meta"], dict)
            or not isinstance(d["body"], dict)):
        return None, "line envelope has wrongly typed fields"
    try:
        h = line_hash(d["seq"], d["kind"], d["part"], d["meta"], d["body"])
    except (TypeError, ValueError) as e:
        return None, f"unhashable content: {e}"
    if h != d["hash"]:
        return None, "content hash mismatch"
    return d, ""


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fsync_dir(path: str) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:                               # pragma: no cover
        return
    try:
        os.fsync(fd)
    except OSError:                               # pragma: no cover
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: str, data: bytes, fsync: bool = True) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        if fsync:
            os.fsync(f.fileno())
    os.replace(tmp, path)
    if fsync:
        fsync_dir(os.path.dirname(os.path.abspath(path)))


def atomic_write_json(path: str, obj: Any, fsync: bool = True) -> int:
    data = (json.dumps(obj, sort_keys=True, indent=1, allow_nan=False)
            + "\n").encode("utf-8")
    atomic_write_bytes(path, data, fsync=fsync)
    return len(data)


def gzip_lines(lines: List[str]) -> bytes:
    # mtime=0: identical content -> identical bytes -> identical file hash.
    return gzip.compress("".join(lines).encode("utf-8"), compresslevel=6,
                         mtime=0)


def read_sealed(path: str) -> List[str]:
    """Lines of a sealed chunk (raises OSError/EOFError on a damaged gzip)."""
    with open(path, "rb") as f:
        raw = gzip.decompress(f.read())
    text = raw.decode("utf-8")
    return text.splitlines(keepends=True)


def read_journal(path: str) -> Tuple[List[str], bytes]:
    """-> (complete lines, trailing bytes that are not a complete line)."""
    with open(path, "rb") as f:
        raw = f.read()
    cut = raw.rfind(b"\n") + 1
    head, tail = raw[:cut], raw[cut:]
    try:
        lines = head.decode("utf-8").splitlines(keepends=True)
    except UnicodeDecodeError:
        # Decode line by line so one damaged line does not hide the rest.
        lines = [ln.decode("utf-8", errors="replace") + "\n"
                 for ln in head.split(b"\n")[:-1]]
    return lines, tail
