"""Independent server-side truth: is SkyBot actually on the server?

THIS IS A FALSIFIER, NOT A SECOND COLLECTOR.
    The pod already knows more about the agent than the server does — break
    counts by type, crafts, positions, option activity. Duplicating that here
    would create two sources that can disagree, and CLAUDE.md §5 records where
    that leads ("the `places` counter... the heuristic is garbage").

    What the server uniquely provides is an INDEPENDENT observation, whose
    value is contradicting the agent's self-report. The specific thing worth
    catching: `num_envs: 2` means two players should be online, and on an
    offline-mode server `SkyBot1` is a DIFFERENT player from `SkyBot` — so a
    missing whitelist entry shows as a persistent 1/20 while the agent's own
    logs look completely healthy.

Deliberately thin: one server-list ping, appended as JSON. No RCON (the
password is unknown), no plugins, no per-block accounting.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import time

HOST = os.environ.get("MC_HOST", "127.0.0.1")
PORT = int(os.environ.get("MC_PORT", "25565"))
OUT = os.environ.get("SERVER_PATH", "/data/server.jsonl")
EVERY = float(os.environ.get("POLL_SECONDS", "30"))
# 754 = 1.16.5, matching the Paper server this project runs.
PROTOCOL = int(os.environ.get("MC_PROTOCOL", "754"))


def _varint(n: int) -> bytes:
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        out += struct.pack("B", b | (0x80 if n else 0))
        if not n:
            return out


def _read_varint(sock: socket.socket) -> int:
    n = 0
    for i in range(5):
        b = sock.recv(1)
        if not b:
            raise ConnectionError("closed while reading varint")
        n |= (b[0] & 0x7F) << (7 * i)
        if not b[0] & 0x80:
            return n
    raise ValueError("varint too long")


def ping(host: str, port: int, timeout: float = 8.0) -> dict:
    """Server List Ping. Returns the parsed status JSON."""
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.settimeout(timeout)
        host_b = host.encode()
        handshake = (b"\x00" + _varint(PROTOCOL) + _varint(len(host_b))
                     + host_b + struct.pack(">H", port) + b"\x01")
        s.sendall(_varint(len(handshake)) + handshake)
        s.sendall(_varint(1) + b"\x00")          # status request
        _read_varint(s)                          # packet length
        if s.recv(1) != b"\x00":
            raise ValueError("unexpected packet id in status response")
        n = _read_varint(s)
        buf = b""
        while len(buf) < n:
            chunk = s.recv(min(4096, n - len(buf)))
            if not chunk:
                raise ConnectionError("closed mid-payload")
            buf += chunk
        return json.loads(buf.decode("utf-8", "replace"))


def main() -> None:
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    print(f"server_poll: {HOST}:{PORT} every {EVERY}s -> {OUT}", flush=True)
    while True:
        rec = {"wall_time": time.time()}
        try:
            st = ping(HOST, PORT)
            players = st.get("players") or {}
            rec["players_online"] = int(players.get("online", 0))
            rec["max_players"] = int(players.get("max", 0))
            rec["version"] = str((st.get("version") or {}).get("name", ""))
        except Exception as exc:
            # A DOWN SERVER IS A DATA POINT, NOT AN ERROR. Recording -1 keeps
            # the outage visible on the dashboard; skipping the write would
            # leave a gap indistinguishable from the poller itself dying.
            rec["players_online"] = -1
            rec["max_players"] = 0
            rec["version"] = f"unreachable: {exc.__class__.__name__}"
        try:
            with open(OUT, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:
            print(f"server_poll: write failed: {exc!r}", flush=True)
        time.sleep(EVERY)


if __name__ == "__main__":
    main()
