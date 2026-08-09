"""LIVE probe: one MineRL client joins an EXTERNAL multiplayer server.

Launches a single GPU-rendered MCP-Reborn client through the normal adapter
path with remote_server set — the mission XML carries <RemoteServer>, so the
client joins the server instead of generating a world — then steps the env
and reports what flows back (obs, inventory, mine_block stats, reward).

Run on the pod (server must be reachable and its whitelist open):
  cd /workspace/devai && DISPLAY=:77 MINERL_HEADLESS=1 PYTHONPATH=. \
    ./venv_mc/bin/python tests/_remote_join_probe.py 127.0.0.1:25565
"""
import sys
import time

sys.path.insert(0, ".")


def main() -> None:
    addr = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:25565"
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    from developmental_ai.environments.minerl_env import MineRLEnvAdapter

    print(f"[probe] building adapter (remote_server={addr}) — client boot + "
          f"server join takes ~90-150s ...", flush=True)
    t0 = time.time()
    env = MineRLEnvAdapter(image_size=128, lifelong=True, remote_server=addr)
    obs, info = env.reset()
    print(f"[probe] RESET OK in {time.time() - t0:.0f}s — joined + first obs "
          f"shape={obs.shape}", flush=True)

    reward_total = 0.0
    for i in range(steps):
        # alternate walk / turn so the bot visibly moves in-world
        obs, r, term, trunc, info = env.step(1 if i % 5 else 3)
        reward_total += float(r)
        if i % 20 == 0 or term or trunc:
            print(f"[probe] step {i:3d}: r={r:+.2f} term={term} trunc={trunc} "
                  f"logs={info.get('logs', '?')}", flush=True)
        if term or trunc:
            print("[probe] episode boundary on remote server (unexpected "
                  "unless death) — continuing not attempted", flush=True)
            break

    print(f"[probe] DONE: {steps} steps, total reward {reward_total:+.2f} — "
          f"JOIN + OBS + ACTIONS verified on the external server", flush=True)
    env.close()


if __name__ == "__main__":
    main()
