# External Minecraft server mode — full setup (both sides)

Point the organism at your OWN Minecraft server instead of a MineRL-generated
world. The AI still learns exactly as on MineRL (same pixel observations, same
actions, same discovery machinery); only the world changes. The bot connects
*out* to your server — the server is never exposed to the internet.

**Every stream, not just env 0.** `remote_server_scope: all` has been set since
2026-08-17, so all `num_envs` clients join the same persistent server (that is
the peer-learning experiment). Each needs its own identity — see
`agent_name_prefix`; slot 0 keeps the bare `SkyBot`, scouts become `SkyBot1`,
`SkyBot2`, … and on an offline-mode server each new name is a NEW player the
whitelist must include.

**TWO WAYS OUT, ONE ENDPOINT (since 2026-09-22).** The java client always
talks to `127.0.0.1:25565`; only the far side of the socat differs, and
`connect_server.sh bridge` picks it from the address you give it:

```
                                        ,--tailnet--> [your server:25565]
[java clients] --127.0.0.1:25565--> [socat]           (ACL-caged, tag:devai)
   env 0..N                             `--LAN TCP--> [your server:25565]
```

| address given to `bridge` | far side |
|---|---|
| `192.168.*`, `10.*`, `172.16–31.*` | plain `socat` TCP over the LAN |
| `100.x` tailnet, MagicDNS names, anything else | `tailscale nc` — unchanged |

Because the endpoint is identical either way, `environment.remote_server` stays
`127.0.0.1:25565` and nothing downstream knows or cares which path is live.
`CONNECT_MODE=lan|tailscale` overrides the choice. **Tailscale is retained in
full** — section A4/A5 below still apply, and switching back is one argument.

**Reality checks up front**
- The client is Minecraft **1.16.5**. A newer server (e.g. 1.21) must run
  **ViaVersion + ViaBackwards** to accept it. Easiest is a 1.16.5 server.
- The bot has no Mojang account → the server needs `online-mode=false`. That
  lets anyone join under any name, so either whitelist (with a caveat below) or
  rely on the tunnel (the server isn't public).
- A live server ticks in real time and won't pause for the agent; the async-WM
  build removes the training freezes that otherwise got the bot kicked.

---

## A. YOUR SERVER (Ubuntu box)

### A1. Version bridge (skip if the server is already 1.16.5)

Drop both plugins into the server's `plugins/` folder and restart:

```bash
cd /path/to/server/plugins
curl -s https://api.github.com/repos/ViaVersion/ViaVersion/releases/latest   | grep -Eo 'https://[^"]+ViaVersion[^"]*\.jar'   | head -1 | xargs curl -LO
curl -s https://api.github.com/repos/ViaVersion/ViaBackwards/releases/latest | grep -Eo 'https://[^"]+ViaBackwards[^"]*\.jar' | head -1 | xargs curl -LO
```

You need **both** — ViaBackwards is the one that lets the *older* 1.16.5 client
join a newer server.

### A2. server.properties

```
online-mode=false
allow-flight=true        # bot freezes briefly during training; without this Paper kicks it for "flying"
```

### A3. Console access via RCON (systemd services have no attachable console)

If the server runs as a systemd service, `journalctl` is read-only — not a
console. Enable RCON in `server.properties` (keep 25575 **LAN-only**, never
forward it):

```
enable-rcon=true
rcon.port=25575
rcon.password=<pick-something-long>
```

Restart, then send commands with the bundled `scripts/rcon.py` (mcrcon isn't in
Ubuntu's repos):

```bash
python3 scripts/rcon.py 127.0.0.1 25575 '<password>' "difficulty peaceful" "time set day" "gamerule doDaylightCycle false" "weather clear" "gamerule doWeatherCycle false"
```

Frozen peaceful daytime helps the bot's vision model (constant lighting), the
same reason its local worlds run frozen-noon.

**Whitelist caveat:** the whitelist matches by **UUID**. In `online-mode=false`
the server uses *offline* UUIDs (derived from the name, case-sensitive), but a
whitelist entry added for a premium name stores the *premium* UUID → "not
whitelisted" even with the right name. Simplest is `whitelist off` (the server
is tunnel-only, not public). To keep it on, compute the offline UUID and edit
`whitelist.json` by hand.

### A4. Tailscale on the server

If not already running:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up          # opens a login URL; sign in
tailscale ip -4            # note this 100.x.y.z — the training host bridges to it
```

### A5. (recommended) Cage the agent with an ACL

So the agent can never reach your other tailnet devices, set the tailnet
policy (admin console → Access Controls) to:

```json
{
  "tagOwners": { "tag:devai": ["autogroup:admin"] },
  "acls": [
    {"action": "accept", "src": ["autogroup:member"], "dst": ["*:*"]},
    {"action": "accept", "src": ["tag:devai"], "dst": ["<server-tailscale-ip>:25565"]}
  ]
}
```

Rule 1 keeps your own devices talking to everything (unchanged); rule 2 lets
anything tagged `tag:devai` reach ONLY the game port. After approving the host
(next section), tag it `tag:devai` on the Machines page.

---

## B. THE TRAINING HOST

Since 2026-09-22 this is the user's own machine (Ubuntu Server, 192.168.1.10),
the training host — installed at `/workspace/devai` and deployed to as
`root@` exactly like one, so every command below is unchanged.

Tailscale + socat are installed by provisioning (stage 1c).

**Over the LAN — one phase.** No login step: nothing is traversing the tailnet.

```bash
bash scripts/connect_server.sh bridge 192.168.1.XX     # the server's LAN IP
#   Prints "bridge mode: lan", then "SERVER REACHABLE" with version/protocol.
```

**Over the tailnet — two phases**, the device-approval step being interactive:

```bash
# Phase 1 — login. Prints an approval URL.
bash scripts/connect_server.sh login
#   Open the URL in a browser on YOUR tailnet, approve the host,
#   then tag it tag:devai in the admin console (Machines page).

# Phase 2 — bridge to the server's tailscale IP (from A4), and ping it.
bash scripts/connect_server.sh bridge <server-tailscale-ip>
#   Prints "bridge mode: tailscale", then "SERVER REACHABLE".
#   protocol 754 in the reply = a 1.16.5 client is accepted (Via is working).
```

`tailscale status` is required **only** on the tailscale path. Gating the LAN
path on it would be a CLAUDE.md §4.1 latch — a precondition nothing on that
path could satisfy. The MC ping runs on both, because it is the check that
actually carries value and it fails for the same reasons either way.

If the ping fails on the LAN path, the usual cause is `server-ip=` in
`server.properties` bound to `127.0.0.1` instead of listening on the LAN, or a
firewall on 25565. The script says which to check per mode.

Then launch external-server mode:

```bash
bash scripts/launch_skybot.sh 1000000
```

Its pre-flight refuses to start unless tailscaled + the bridge + the server ping
all pass, so a dead tunnel can't strand the primary in a connect-fail loop.

NOTE that the launcher's pre-flight still checks `tailscaled` even on the LAN
path. On the main computer tailscaled runs anyway (it is kept for when
ethernet returns), so this passes — but it is a check unrelated to whether the
LAN bridge works, and it would refuse a perfectly healthy LAN run if tailscaled
ever stopped. Left as-is deliberately; worth revisiting if that ever bites.

---

## C. What to expect in-world

- The bot appears in your player list under a **MineRL-random name**
  (`Player###`), not a name you set.
- **It spawns empty-handed** — a foreign server is authoritative over inventory,
  so the iron-axe handout doesn't stick; it punches trees barehanded (slower).
  `python3 scripts/rcon.py ... "give <name> minecraft:iron_axe"` helps a lot.
- **Auto-rejoin**: a kick/disconnect leaves the client on a static screen; the
  adapter detects the frozen POV (150 motion-commanded steps with no frame
  change) and rebuilds → the fresh mission rejoins automatically. Hard socket
  errors rejoin the same way.
- A live server never resets, which suits the lifelong design (each stream is a
  single continuous life); death is respawn, not a world reset.

## D. Config knobs (`configs/minecraft_skybot.yaml`)

```yaml
environment:
  remote_server: "127.0.0.1:25565"   # the host-local bridge endpoint — the
                                     # SAME on the LAN and tailnet paths, which
                                     # is why this line never changes
  remote_server_scope: all           # LIVE VALUE: every stream joins the server
                                     # ("primary" = only env 0; scouts local)
```

The live config uses **`all`** (since 2026-08-17), so no stream runs a local
generated world. Whitelist `SkyBot` **and** `SkyBot1` — with `online-mode=false`
each name is a distinct player.

Set `remote_server: null` (or use `configs/minecraft_lifelong.yaml`) to go back
to pure-MineRL local worlds — the SkyBot tunnel can stay running, idle.
