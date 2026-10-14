"""The live viewer plays the GAME at its own speed, not one still per decision.

Born 2026-10-11. The user asked why the Doom viewer ran at ~0.1 fps and
whether they could watch the game live. Measured: the viewer received ONE
frame per agent step, so the picture moved only as fast as SkyBot decided
(4-15 s/step on a Mac CPU), and even at full speed it showed 1 of every 4
game frames. Both games run synchronously (the world is frozen while the
agent thinks), so "live" means: every game frame, played at the game's real
rate on the viewer's own clock, and an honest label when the game is paused
waiting for SkyBot.

Contracts:
    A. GAME SPEED. 35 frames of 1/35 s each are played back at ~35 fps
       (the viewer's clock, not the producer's).
    B. STAYS NEAR LIVE. A producer faster than real time cannot build an
       ever-growing backlog: the queue is capped at max_buffer_s of game
       time and the oldest frames are dropped (and counted).
    C. NEVER FAKES LIVE. With nothing to play, the held frame is relabelled
       "WAITING FOR SKYBOT" and the status reports how long since the last
       step — a stalled agent is visible as stalled.
    D. PAUSE drops pushes (the existing Pause button now covers the game).
    E. THE /game ENDPOINT serves a multipart MJPEG stream of those frames.
    F. ANY STREAM, NOT JUST 0 (2026-10-13). One brain plays several games at
       once and the viewer showed stream 0 only. `stream:N` from the page
       selects a stream, an out-of-range choice is ignored, a switch drops the
       old stream's queued frames (never a mix of two games), and /data lists
       the streams. The loop pushes the SELECTED stream's frame, action,
       reward and belief, and unhooks the previous stream's game clock.

Run: PYTHONPATH=. python tests/_viewer_playback_smoke.py
"""
import os
import socket
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from developmental_ai.core.viewer_server import ViewerServer      # noqa: E402

NAME = "viewer-playback"
N = [0]


def check(cond, msg):
    N[0] += 1
    if not cond:
        print(f"  {N[0]}. FAIL  {msg}")
        raise SystemExit(f"[{NAME}] FAIL: {msg}")
    print(f"  {N[0]}. ok    {msg}")


def frames(n, shade=0):
    return [np.full((48, 64, 3), (shade + i) % 255, np.uint8) for i in range(n)]


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    print(f"[{NAME}]")
    v = ViewerServer(port=0, max_buffer_s=1.5)
    v.start_playback()
    # ---- A
    v.push_game_frames(frames(35), 1 / 35)
    t0, s0 = time.time(), v.game_shown
    while v.game_shown - s0 < 35 and time.time() - t0 < 5:
        time.sleep(0.01)
    dt = time.time() - t0
    check(v.game_shown - s0 == 35 and 0.8 <= dt <= 1.4,
          "A. 35 frames x 1/35 s played in %.2f s (~35 fps, the game's speed)" % dt)
    # ---- B
    v.push_game_frames(frames(300), 1 / 35)            # ~8.6 s of game at once
    st = v.game_status()
    check(st["buffered_s"] <= 1.5 + 1e-6 and v.game_dropped >= 300 - 53,
          "B. backlog capped at 1.5 s of game; %d oldest frames dropped" % v.game_dropped)
    # ---- C
    while v.game_status()["buffered_s"] > 0:
        time.sleep(0.05)
    time.sleep(1.2)
    seq1, j1 = v.get_game_jpeg()
    time.sleep(0.7)
    seq2, j2 = v.get_game_jpeg()
    st = v.game_status()
    check(seq2 > seq1 and st["since_last_frame_s"] is not None
          and st["since_last_frame_s"] >= 1.5,
          "C. starved: frame relabelled (seq %d->%d), status says %.1f s since the last frame"
          % (seq1, seq2, st["since_last_frame_s"]))
    # ---- D
    v.paused = True
    q0 = v.game_status()["buffered_s"]
    v.push_game_frames(frames(10), 1 / 35)
    check(v.game_status()["buffered_s"] == q0, "D. paused: pushes are dropped")
    v.paused = False
    # ---- E
    port = free_port()
    srv = ViewerServer(port=port)
    srv.start()
    srv.push_game_frames(frames(70, 50), 1 / 35)
    time.sleep(0.3)
    r = urllib.request.urlopen(f"http://127.0.0.1:{port}/game", timeout=5)
    ctype = r.headers.get("Content-Type", "")
    data = b""
    t0 = time.time()
    while data.count(b"--frame") < 5 and time.time() - t0 < 5:
        data += r.read1(65536) if hasattr(r, "read1") else r.read(4096)
    r.close()
    check("multipart/x-mixed-replace" in ctype and data.count(b"--frame") >= 5
          and b"\xff\xd8" in data, "E. /game streams MJPEG frames (%d seen)"
          % data.count(b"--frame"))
    page = urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=5).read()
    check(b'src="/game"' in page and b'src="/stream"' in page,
          "E. the page shows the live game beside the world-model dashboard")
    # ---- F
    v2 = ViewerServer(port=0)
    v2.set_streams(["0: minecraft", "1: doom", "2: doom"])
    v2.push_game_frames(frames(20), 1 / 35)
    v2.command("stream:2")
    check(v2.selected_stream == 2 and v2.game_status()["buffered_s"] == 0.0,
          "F. stream:2 selected; the old stream's queued frames dropped")
    v2.command("stream:7"); v2.command("stream:x")
    check(v2.selected_stream == 2, "F. out-of-range / malformed selections ignored")
    import json as _json
    v2.push(np.zeros((8, 8, 3), np.uint8), "t", [("a", "b")])
    d = _json.loads(v2._raw)
    check(d["streams"] == ["0: minecraft", "1: doom", "2: doom"] and d["selected"] == 2,
          "F. /data lists the streams and the selection")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    loop = open(os.path.join(root, "developmental_ai/core/developmental_loop.py")).read()
    check('_vs = int(getattr(self.viewer, "selected_stream", 0) or 0)' in loop
          and "env_actions[_vs], float(rewards[_vs])" in loop
          and "latent[_vs:_vs + 1], episode_length, stream=_vs)" in loop,
          "F. the loop pushes the selected stream's action, reward and belief")
    check("_ps(None)" in loop and "self._viewer_sink_env = _venv" in loop
          and "self.env.render()" not in loop[loop.index("    def _viewer_push"):
                                               loop.index("    def _emit_heartbeat")],
          "F. the previous stream's clock is unhooked; frames come from the watched env")
    print(f"[{NAME}] ALL PASS")


if __name__ == "__main__":
    main()
