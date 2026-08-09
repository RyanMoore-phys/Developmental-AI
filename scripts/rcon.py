"""Tiny Minecraft RCON client — no deps (mcrcon isn't in Ubuntu's repos).

Runs on YOUR server box to send console commands over RCON. Enable RCON in
server.properties first (enable-rcon=true, rcon.port=25575, rcon.password=...)
and restart. Keep 25575 LAN-only — never port-forward it. Usage:
    python3 scripts/rcon.py <host> <port> <password> "cmd1" ["cmd2" ...]
e.g. python3 scripts/rcon.py 127.0.0.1 25575 SECRET "whitelist off" "time set day"
"""
import socket
import struct
import sys


def pkt(rid, ptype, payload):
    body = struct.pack("<ii", rid, ptype) + payload.encode() + b"\x00\x00"
    return struct.pack("<i", len(body)) + body


def recv_pkt(s):
    raw = s.recv(4)
    ln = struct.unpack("<i", raw)[0]
    data = b""
    while len(data) < ln:
        data += s.recv(ln - len(data))
    rid, ptype = struct.unpack("<ii", data[:8])
    return rid, ptype, data[8:-2].decode(errors="replace")


def main():
    host, port, pw, *cmds = sys.argv[1:]
    s = socket.create_connection((host, int(port)), timeout=10)
    s.sendall(pkt(1, 3, pw))                   # SERVERDATA_AUTH
    if recv_pkt(s)[0] == -1:
        sys.exit("RCON auth failed (wrong rcon.password?)")
    for c in cmds:
        s.sendall(pkt(2, 2, c))                # SERVERDATA_EXECCOMMAND
        print(recv_pkt(s)[2] or "(ok)")


if __name__ == "__main__":
    main()
