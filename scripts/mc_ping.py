"""Minimal Minecraft Server List Ping (status query) — no deps.

Confirms a server answers and reports version/protocol/player-count. Used by
the socat-bridge pre-flight (a Paper server WITH ViaVersion+ViaBackwards
answers a 1.16.5 client, protocol 754). Usage:
    python3 scripts/mc_ping.py <host> <port> [protocol]
"""
import json
import socket
import struct
import sys


def varint(n):
    out = b""
    while True:
        b7 = n & 0x7F
        n >>= 7
        out += bytes([b7 | (0x80 if n else 0)])
        if not n:
            return out


def read_varint(s):
    n = shift = 0
    while True:
        b = s.recv(1)
        if not b:
            raise EOFError("closed mid-varint")
        v = b[0]
        n |= (v & 0x7F) << shift
        if not v & 0x80:
            return n
        shift += 7


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 25565
    proto = int(sys.argv[3]) if len(sys.argv) > 3 else 754  # 754 = 1.16.5
    s = socket.create_connection((host, port), timeout=12)
    addr = host.encode()
    data = (varint(0) + varint(proto) + varint(len(addr)) + addr
            + struct.pack(">H", port) + varint(1))
    s.sendall(varint(len(data)) + data)
    s.sendall(varint(1) + varint(0))          # status request
    read_varint(s)                            # packet length
    read_varint(s)                            # packet id
    jl = read_varint(s)
    buf = b""
    while len(buf) < jl:
        c = s.recv(jl - len(buf))
        if not c:
            break
        buf += c
    d = json.loads(buf.decode("utf-8", "replace"))
    v = d.get("version", {})
    p = d.get("players", {})
    print("SERVER RESPONDED")
    print("  version:", v.get("name"), "| protocol:", v.get("protocol"))
    print("  players:", p.get("online"), "/", p.get("max"))
    desc = d.get("description")
    print("  motd:", desc if isinstance(desc, str) else (desc or {}).get(
        "text", ""))


if __name__ == "__main__":
    main()
