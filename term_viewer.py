"""
Terminal viewer — watch the live agent + world model right inside your terminal
(e.g. the VS Code remote terminal that's already SSH'd into the VM). No browser,
no port tunnel: it polls the viewer server's /data endpoint on localhost and
renders the env as truecolor half-block pixels, the metrics as text, and the
world-model latent as a heatmap, refreshing in place.

Usage (run it in the VM terminal, while training runs with viewer.enabled):
    python term_viewer.py --port 8000
Ctrl-C to quit. Needs a truecolor terminal (VS Code's terminal qualifies).
"""
import sys, time, io, json, base64, shutil, argparse, urllib.request
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
try:
    _CMAP = matplotlib.colormaps["viridis"]
except Exception:
    from matplotlib import cm
    _CMAP = cm.get_cmap("viridis")
VIRIDIS = (np.asarray([_CMAP(i / 255.0)[:3] for i in range(256)]) * 255).astype(np.uint8)

E = "\x1b"
RESET = f"{E}[0m"
CLR_EOL = f"{E}[K"


def _blocks(rgb):
    """RGB ndarray (H,W,3) -> list of half-block lines (2 px rows per line)."""
    H, W = rgb.shape[:2]
    out = []
    for y in range(0, H - 1, 2):
        top, bot = rgb[y], rgb[y + 1]
        s = []
        for x in range(W):
            tr, tg, tb = top[x]; br, bg, bb = bot[x]
            s.append(f"{E}[38;2;{tr};{tg};{tb};48;2;{br};{bg};{bb}m▀")
        out.append("".join(s) + RESET + CLR_EOL)
    return out


def _fit_blocks(img, max_cols, max_char_rows):
    """Half-block render of a PIL image, scaled to fit within max_cols columns
    AND max_char_rows character rows (preserving aspect). Returns ≤ max_char_rows
    lines — this is what keeps the whole frame inside the terminal."""
    w, h = img.size
    scale = min(max_cols / w, (max(1, max_char_rows) * 2) / h)
    cw = max(2, int(w * scale)); ch = max(2, (int(h * scale) // 2) * 2)
    return _blocks(np.asarray(img.resize((cw, ch)).convert("RGB")))


def render(raw, cols, rows):
    """Build a frame that fits in `rows` terminal lines so it never scrolls."""
    head = [f"  {raw.get('title','')}    {raw.get('fps', 0)} fps    "
            f"(Ctrl-C to quit){CLR_EOL}", CLR_EOL]
    metrics = [f"  {k:<16}{v}{CLR_EOL}" for k, v in raw.get("lines", [])]
    has_env, has_recon, has_z = bool(raw.get("env")), bool(raw.get("recon")), bool(raw.get("z"))
    # reserve rows for text/labels, split the remainder among the images
    labels = (1 if has_recon else 0) + (2 if has_z else 0) + 2
    budget = max(6, rows - 1 - len(head) - len(metrics) - labels)
    env_r = int(budget * (0.45 if has_recon else 0.62)) if has_env else 0
    rec_r = int(budget * 0.28) if has_recon else 0
    lat_r = max(3, budget - env_r - rec_r) if has_z else 0

    out = list(head)
    if has_env:
        img = Image.open(io.BytesIO(base64.b64decode(raw["env"]))).convert("RGB")
        out += ["  " + ln for ln in _fit_blocks(img, cols, max(3, env_r))]
        out.append(CLR_EOL)
    if has_recon:
        out.append(f"  WM prediction  (real=blue, predicted=orange){CLR_EOL}")
        rim = Image.open(io.BytesIO(base64.b64decode(raw["recon"]))).convert("RGB")
        out += ["  " + ln for ln in _fit_blocks(rim, min(cols, 64), max(3, rec_r))]
        out.append(CLR_EOL)
    out += metrics
    if has_z:
        out.append(f"  world-model latent (h⊕z){CLR_EOL}")
        g = np.clip(np.asarray(raw["z"], dtype=np.float32), 0, 1)
        rgb = Image.fromarray(VIRIDIS[(g * 255).astype(np.uint8)], "RGB")
        out += ["  " + ln for ln in _fit_blocks(rgb, cols, max(3, lat_r))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--fps", type=float, default=6.0)
    args = ap.parse_args()
    url = f"http://{args.host}:{args.port}/data"

    sys.stdout.write(f"{E}[2J{E}[?25l")            # clear + hide cursor
    try:
        while True:
            t0 = time.time()
            try:
                raw = json.loads(urllib.request.urlopen(url, timeout=4).read())
            except Exception:
                # No run on this port — show a calm idle line (no error text) and
                # wipe any stale frame, so you can leave the terminal open.
                spin = "|/-\\"[int(time.time() * 2) % 4]
                sys.stdout.write(
                    f"{E}[H  {spin} idle — waiting for a run on "
                    f"{args.host}:{args.port}    (Ctrl-C to quit){CLR_EOL}\n{E}[J")
                sys.stdout.flush(); time.sleep(0.5); continue
            ts = shutil.get_terminal_size((90, 40))
            cols = max(20, ts.columns - 4); rows = max(14, ts.lines)
            frame = render(raw, cols, rows)[:rows - 1]   # clamp: never overflow
            sys.stdout.write(f"{E}[H" + "\n".join(frame) + f"{E}[J")
            sys.stdout.flush()
            dt = 1.0 / args.fps - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(f"{E}[?25h{RESET}\n")     # restore cursor


if __name__ == "__main__":
    main()
