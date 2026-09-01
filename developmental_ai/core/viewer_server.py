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
button{background:#1d1d28;color:#ddd;border:1px solid #444;border-radius:6px;padding:8px 16px;margin:6px;cursor:pointer}
button:hover{background:#2a2a3a}</style></head><body>
<h2>AI Live Viewer &mdash; world model + environment</h2>
<div><img src="/stream"></div>
<div><button onclick="fetch('/cmd?c=toggle')">Pause / Resume</button></div>
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
    def __init__(self, port=8000):
        self.port = port
        self._lock = threading.Lock()
        self._jpeg = to_jpeg(Image.new("RGB", (690, 620), (12, 12, 16)))
        self._raw = "{}"                       # latest raw state JSON (for /data)
        self._errs = deque(maxlen=80)          # recent 1-step prediction errors
        self.paused = False
        self._n, self._last, self._fps = 0, time.time(), 0.0

    def start(self):
        self._httpd = ThreadingHTTPServer(("0.0.0.0", self.port), _make_handler(self))
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def command(self, c):
        if c == "toggle":
            self.paused = not self.paused

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
               "lines": [[str(k), str(v)] for k, v in lines]}
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
