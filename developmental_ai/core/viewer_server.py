"""
Live viewer server — a reusable MJPEG dashboard the training loop can stream to.

The DevelopmentalLoop creates a ViewerServer (when `viewer.enabled`) and calls
`push(...)` once per environment step with the live env frame + world-model
state. The browser shows the real env render beside a world-model panel (latent
heatmap, developmental stage, reward mix, return history) — i.e. exactly what the
AI is doing AND what its world model is thinking, in real time.

Zero non-stdlib web deps (stdlib HTTP + multipart MJPEG). Headless-friendly.
Everything here is best-effort and isolated: a viewer failure must never break
training, so callers wrap push() in try/except.
"""
import io, time, threading, json, base64
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")

try:
    _CMAP = matplotlib.colormaps["viridis"]
except Exception:                                   # older matplotlib
    from matplotlib import cm
    _CMAP = cm.get_cmap("viridis")
VIRIDIS = (np.asarray([_CMAP(i / 255.0) for i in range(256)])[:, :3] * 255).astype(np.uint8)

try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 15)
    FONT_BIG = ImageFont.truetype("DejaVuSans-Bold.ttf", 19)
except Exception:
    FONT = FONT_BIG = ImageFont.load_default()

PANEL_W = 330
TARGET_H = 420


def to_jpeg(img, q=80):
    b = io.BytesIO(); img.save(b, "JPEG", quality=q); return b.getvalue()


def _sparkline(values, w, h):
    img = Image.new("RGB", (w, h), (20, 20, 24)); d = ImageDraw.Draw(img)
    if values and len(values) >= 2:
        lo, hi = min(values), max(values); rng = (hi - lo) or 1.0
        pts = [(int(i * (w - 1) / (len(values) - 1)),
                int(h - 1 - (v - lo) / rng * (h - 1))) for i, v in enumerate(values)]
        d.line(pts, fill=(90, 200, 120), width=2)
    return img


def _heatmap(vec, size=270):
    """Render a 1-D latent vector as a square viridis heatmap."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    side = int(np.ceil(np.sqrt(v.size)))
    pad = np.zeros(side * side, dtype=np.float32); pad[:v.size] = v
    g = pad.reshape(side, side)
    g = (g - g.min()) / ((g.max() - g.min()) or 1.0)
    rgb = VIRIDIS[(g * 255).astype(np.uint8)]
    return Image.fromarray(rgb, "RGB").resize((size, size), Image.NEAREST)


def render_recon(env_name, real, pred, w=300, h=140):
    """Overlay the ACTUAL next state (blue) vs the world model's one-step
    PREDICTED next state (orange) for a renderable env. Both poses use the same
    (normalized) mapping, so visual overlap == an accurate prediction. Returns a
    PIL image, or None for envs we don't have a renderer for (the caller then
    falls back to the numeric error readout)."""
    real = np.asarray(real, dtype=np.float32).ravel()
    pred = np.asarray(pred, dtype=np.float32).ravel()
    if "CartPole" not in env_name or real.size < 3:
        return None
    img = Image.new("RGB", (w, h), (18, 18, 24)); d = ImageDraw.Draw(img)
    base = h * 0.72
    d.line([(0, base), (w, base)], fill=(70, 70, 80), width=1)

    def draw(obs, color, width):
        x = float(np.clip(obs[0], -2.5, 2.5)); th = float(np.clip(obs[2], -2.5, 2.5))
        cx = w / 2 + (x / 2.5) * (w * 0.30)
        cw, ch = w * 0.09, h * 0.13
        d.rectangle([cx - cw / 2, base - ch, cx + cw / 2, base], outline=color, width=width)
        L = h * 0.46; ang = (th / 2.5) * 0.7
        tx = cx + L * np.sin(ang); ty = base - ch - L * np.cos(ang)
        d.line([(cx, base - ch), (tx, ty)], fill=color, width=width)

    draw(real, (95, 175, 240), 5)        # actual
    draw(pred, (245, 150, 45), 3)        # WM predicted
    return img


# fixed palette for income sources — deterministic per name so a source keeps
# its colour across frames and runs (a bar that changes colour reads as a
# different source)
_INCOME_COLORS = [(95, 175, 240), (110, 220, 130), (245, 180, 90),
                  (240, 120, 120), (190, 140, 240), (240, 210, 90),
                  (120, 210, 210), (200, 200, 210)]


def _income_color(name):
    return _INCOME_COLORS[hash(str(name)) % len(_INCOME_COLORS)]


def compose(env_rgb, title, lines, paused, fps, z, ret_hist,
            recon_img=None, err=None, errs=None, income=None):
    if env_rgb is not None:
        env_img = Image.fromarray(np.asarray(env_rgb, dtype=np.uint8))
        s = TARGET_H / env_img.height
        env_img = env_img.resize((max(1, int(env_img.width * s)), TARGET_H))
        ew = env_img.width
    else:
        env_img, ew = None, 360
    H = max(TARGET_H, 900)
    canvas = Image.new("RGB", (ew + PANEL_W, H), (12, 12, 16))
    if env_img is not None:
        canvas.paste(env_img, (0, (H - TARGET_H) // 2))
    d = ImageDraw.Draw(canvas); x0 = ew + 16
    d.text((x0, 12), title, font=FONT_BIG, fill=(235, 235, 245))
    d.text((x0, 40), "PAUSED" if paused else "RUNNING", font=FONT,
           fill=(240, 180, 80) if paused else (110, 220, 130))
    d.text((x0 + 90, 40), f"{fps:.1f} fps", font=FONT, fill=(150, 150, 160))
    y = 74
    for k, v in lines:
        d.text((x0, y), f"{k}", font=FONT, fill=(150, 150, 160))
        d.text((x0 + 130, y), str(v), font=FONT, fill=(228, 228, 238)); y += 25
    # --- LIVE INCOME STATEMENT (reward-ledger shares, this segment so far).
    # What the learner is being PAID for right now — the number-one question
    # of every reward-hacking incident this project has had. Bars are shares
    # of |income|; the sign rides in the % label.
    if income:
        d.text((x0, y + 4), "income (live)", font=FONT, fill=(150, 150, 160))
        y += 26
        bar_x, bar_w = x0 + 118, PANEL_W - 200
        for name, frac, signed in income:
            c = _income_color(name)
            d.text((x0, y), str(name)[:14], font=FONT, fill=c)
            w = max(1, int(bar_w * max(0.0, min(1.0, float(frac)))))
            d.rectangle([bar_x, y + 3, bar_x + bar_w, y + 13],
                        outline=(60, 60, 72))
            d.rectangle([bar_x, y + 3, bar_x + w, y + 13], fill=c)
            d.text((bar_x + bar_w + 6, y),
                   f"{100.0 * float(frac):.0f}%"
                   + ("" if float(signed) >= 0 else " (−)"),
                   font=FONT, fill=(200, 200, 210))
            y += 19
        y += 6
    # --- WM accuracy panel: predicted (orange) overlaid on actual (blue) ---
    if recon_img is not None or err is not None:
        d.text((x0, y + 4), "WM prediction", font=FONT, fill=(150, 150, 160))
        d.text((x0 + 110, y + 4), "real", font=FONT, fill=(95, 175, 240))
        d.text((x0 + 150, y + 4), "vs", font=FONT, fill=(150, 150, 160))
        d.text((x0 + 172, y + 4), "predicted", font=FONT, fill=(245, 150, 45))
        y += 26
        if recon_img is not None:
            canvas.paste(recon_img, (x0, y)); y += recon_img.height + 6
        if err is not None:
            d.text((x0, y), "1-step pred err", font=FONT, fill=(150, 150, 160))
            d.text((x0 + 130, y), f"{err:.4f}", font=FONT, fill=(245, 180, 90)); y += 22
        if errs:
            canvas.paste(_sparkline(list(errs), PANEL_W - 32, 38), (x0, y)); y += 46
    d.text((x0, y + 4), "return history", font=FONT, fill=(150, 150, 160))
    canvas.paste(_sparkline(list(ret_hist or []), PANEL_W - 32, 50), (x0, y + 28))
    y += 90
    if z is not None:
        d.text((x0, y), "world-model latent (h⊕z)", font=FONT, fill=(150, 150, 160))
        canvas.paste(_heatmap(z, 220), (x0, y + 22))
    return canvas


_PAGE = b"""<!doctype html><html><head><meta charset=utf-8><title>AI Live Viewer</title>
<style>body{background:#0b0b10;color:#ddd;font-family:system-ui;text-align:center;margin:0;padding:16px}
img{border:1px solid #333;border-radius:8px;max-width:97vw}
#game{width:min(97vw,960px);image-rendering:pixelated}
.row{display:flex;flex-wrap:wrap;gap:12px;justify-content:center;align-items:flex-start}
button{background:#1d1d28;color:#ddd;border:1px solid #444;border-radius:6px;padding:8px 16px;margin:6px;cursor:pointer}
button:hover{background:#2a2a3a}</style></head><body>
<h2>AI Live Viewer &mdash; the game at its own speed, and what the world model thinks</h2>
<div class=row><div><img id=game src="/game"></div><div><img src="/stream"></div></div>
<div id=streams></div>
<div><button onclick="fetch('/cmd?c=toggle')">Pause / Resume</button></div>
<script>
async function refresh(){try{const d=await (await fetch('/data')).json();
 const s=document.getElementById('streams');const L=d.streams||[];
 s.innerHTML=L.length>1?'Watching: '+L.map((n,i)=>'<button style="'+(i==d.selected?'background:#2f4f2f':'')+'" onclick="fetch(\'/cmd?c=stream:'+i+'\')">'+n+'</button>').join(''):'';
}catch(e){}}
setInterval(refresh,1500);refresh();
</script>
</body></html>"""


def _make_handler(srv):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_GET(self):
            if self.path.startswith("/cmd"):
                srv.command(self.path.split("c=")[-1] if "c=" in self.path else "")
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
            if self.path == "/data":
                d = srv._raw.encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(d)))
                self.end_headers(); self.wfile.write(d); return
            if self.path == "/game":
                # THE GAME AT ITS OWN SPEED (2026-10-11): one JPEG per game
                # frame, paced by the playback thread, not by training steps.
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                seen = -1
                try:
                    while True:
                        seq, j = srv.get_game_jpeg()
                        if seq != seen:
                            seen = seq
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                             + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                        time.sleep(0.004)
                except (BrokenPipeError, ConnectionResetError):
                    return
            if self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        j = srv.get_jpeg()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    return
            self.send_response(200); self.send_header("Content-Type", "text/html")
            self.end_headers(); self.wfile.write(_PAGE)
    return H


class ViewerServer:
    def __init__(self, port=8000, max_buffer_s=1.5):
        self.port = port
        self._lock = threading.Lock()
        self._jpeg = to_jpeg(Image.new("RGB", (690, 620), (12, 12, 16)))
        self._raw = "{}"                       # latest raw state JSON (for /data)
        self._errs = deque(maxlen=80)          # recent 1-step prediction errors
        self.paused = False
        self._n, self._last, self._fps = 0, time.time(), 0.0
        # ---- GAME PLAYBACK (2026-10-11) -----------------------------------
        # The training loop is synchronous: the game only advances when the
        # agent steps, and it used to hand this server ONE frame per step,
        # so the picture moved only as fast as SkyBot decided. Now the loop
        # hands over EVERY game frame of the step with the game time each
        # one covers, and this thread plays them back on its own clock at the
        # game's real speed. Agent faster than real time -> oldest frames are
        # dropped to stay near live (max_buffer_s). Agent slower -> the last
        # frame is held and the overlay SAYS it is waiting for SkyBot; it
        # never pretends a stalled game is live.
        self.max_buffer_s = float(max_buffer_s)
        self._game_lock = threading.Lock()
        self._game_q = deque()                 # (HxWx3 uint8, seconds)
        self._game_q_s = 0.0                   # game-seconds queued
        self._game_jpeg = to_jpeg(Image.new("RGB", (320, 240), (12, 12, 16)))
        self._game_seq = 0
        self._game_last_frame = None
        self._game_last_push = None
        self.game_shown = 0                    # frames played (telemetry/tests)
        self.game_dropped = 0                  # frames dropped to stay live
        self._game_fps = 0.0
        # SkyBot's DECISIONS, timed separately from the game's frames. On a
        # real-time game the clock feeds frames every tic whatever SkyBot
        # does, so frame arrival says nothing about whether SkyBot is alive;
        # and `_fps` is only recomputed when a decision arrives, so a stalled
        # loop kept showing its last rate (it misled a live check 2026-10-11).
        self._decision_times = deque(maxlen=400)
        # WHICH STREAM IS ON SCREEN (2026-10-13). One brain plays several
        # games/streams at once; the viewer used to show stream 0 only, so in
        # a mixed run the others played unseen. The loop reads this.
        self.selected_stream = 0
        self.stream_labels = []

    def start(self):
        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), _make_handler(self))
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        self.start_playback()

    def start_playback(self):
        threading.Thread(target=self._playback, daemon=True,
                         name="viewer-playback").start()

    def push_game_frames(self, frames, frame_seconds):
        """Hand over the game frames one agent step produced (oldest first),
        each covering `frame_seconds` of game time. Called by the training
        loop; cheap (a deque append), never blocks on encoding."""
        if self.paused or not frames:
            return
        dt = max(1e-3, float(frame_seconds))
        with self._game_lock:
            for f in frames:
                if f is None:
                    continue
                self._game_q.append((f, dt))
                self._game_q_s += dt
            while self._game_q and self._game_q_s > self.max_buffer_s:
                _f, _dt = self._game_q.popleft()
                self._game_q_s -= _dt
                self.game_dropped += 1
            self._game_last_push = time.time()

    def get_game_jpeg(self):
        with self._game_lock:
            return self._game_seq, self._game_jpeg

    def game_status(self):
        with self._game_lock:
            q_s = self._game_q_s
            last = self._game_last_push
        now = time.time()
        waited = None if last is None else now - last
        dts = list(self._decision_times)
        recent = [t for t in dts if now - t <= 5.0]
        return {"buffered_s": round(q_s, 3), "fps": round(self._game_fps, 1),
                "shown": self.game_shown, "dropped": self.game_dropped,
                "since_last_frame_s": None if waited is None else round(waited, 2),
                # live, never stale: decisions in the last 5 s, and how long
                # since SkyBot's last one
                "decisions_per_s": round(len(recent) / 5.0, 2),
                "since_last_decision_s": (None if not dts
                                          else round(now - dts[-1], 2))}

    def _label(self, frame, text, color):
        img = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 8 + 6 * len(text), 14], fill=(0, 0, 0))
        d.text((4, 2), text, fill=color)
        return to_jpeg(img, 85)

    def _playback(self):
        n, t_fps = 0, time.time()
        next_t = time.time()
        idle_label_t = 0.0
        while True:
            item = None
            with self._game_lock:
                if self._game_q:
                    item = self._game_q.popleft()
                    self._game_q_s -= item[1]
                    q_s = self._game_q_s
            now = time.time()
            if item is None:
                # Nothing to play: the game is frozen until SkyBot's next
                # decision. Say so (refreshed twice a second), hold the frame.
                if self._game_last_frame is not None and now - idle_label_t > 0.5:
                    idle_label_t = now
                    waited = now - (self._game_last_push or now)
                    j = self._label(self._game_last_frame,
                                    f"WAITING FOR SKYBOT  {waited:4.1f}s  (game paused)",
                                    (255, 190, 60))
                    with self._game_lock:
                        self._game_jpeg = j
                        self._game_seq += 1
                time.sleep(0.005)
                next_t = time.time()
                continue
            frame, dt = item
            self._game_last_frame = frame
            j = self._label(frame, f"LIVE  game speed  buffer {q_s:3.1f}s",
                            (120, 230, 120))
            with self._game_lock:
                self._game_jpeg = j
                self._game_seq += 1
            self.game_shown += 1
            n += 1
            if now - t_fps >= 1.0:
                self._game_fps, n, t_fps = n / (now - t_fps), 0, now
            next_t = max(next_t + dt, time.time() - dt)
            delay = next_t - time.time()
            if delay > 0:
                time.sleep(delay)

    def command(self, c):
        if c == "toggle":
            self.paused = not self.paused
        elif c.startswith("stream:"):
            try:
                k = int(c.split(":", 1)[1])
            except ValueError:
                return
            n = len(self.stream_labels)
            if 0 <= k and (n == 0 or k < n) and k != self.selected_stream:
                self.selected_stream = k
                self.clear_game_queue()        # never mix two streams' frames

    def set_streams(self, labels):
        self.stream_labels = [str(x) for x in labels]
        if self.selected_stream >= len(self.stream_labels):
            self.selected_stream = 0

    def clear_game_queue(self):
        with self._game_lock:
            self._game_q.clear()
            self._game_q_s = 0.0

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def push(self, env_rgb, title, lines, z=None, ret_hist=None, recon=None,
             income=None):
        """Called once per env step by the training loop. Best-effort.

        recon (optional): {"env_name", "real", "pred", "err"} — the actual next
        observation vs the world model's one-step prediction, for the live
        accuracy panel.
        """
        if self.paused:
            return
        self._n += 1; now = time.time()
        self._decision_times.append(now)
        if now - self._last >= 1.0:
            self._fps = self._n / (now - self._last); self._n = 0; self._last = now
        recon_img, err = None, None
        if recon:
            err = recon.get("err")
            if err is not None:
                self._errs.append(float(err))
            recon_img = render_recon(recon.get("env_name", ""),
                                     recon.get("real", []), recon.get("pred", []))
        img = compose(env_rgb, title, lines, self.paused, self._fps, z, ret_hist,
                      recon_img, err, list(self._errs), income=income)
        jpeg = to_jpeg(img)
        # Raw state for the terminal client (/data): small env thumbnail +
        # metrics text + a 32x32 normalized latent grid + reconstruction.
        raw = {"title": str(title), "fps": round(self._fps, 1),
               "lines": [[str(k), str(v)] for k, v in lines],
               "game": self.game_status(),
               "streams": self.stream_labels, "selected": self.selected_stream}
        if income:
            raw["income"] = [[str(n), round(float(f), 3),
                              round(float(s), 4)] for n, f, s in income]
        try:
            if err is not None:
                raw["err"] = round(float(err), 4)
                raw["errs"] = [round(float(e), 4) for e in self._errs]
            if recon_img is not None:
                raw["recon"] = base64.b64encode(to_jpeg(recon_img, 75)).decode()
            if env_rgb is not None:
                e = Image.fromarray(np.asarray(env_rgb, dtype=np.uint8))
                e.thumbnail((200, 200))
                raw["env"] = base64.b64encode(to_jpeg(e, 70)).decode()
            if z is not None:
                zz = np.asarray(z, dtype=np.float32).ravel()
                side = int(np.ceil(np.sqrt(zz.size)))
                pad = np.zeros(side * side, dtype=np.float32); pad[:zz.size] = zz
                g = pad.reshape(side, side)
                g = (g - g.min()) / ((g.max() - g.min()) or 1.0)
                raw["z"] = np.round(g, 3).tolist()
        except Exception:
            pass
        with self._lock:
            self._jpeg = jpeg
            self._raw = json.dumps(raw)
