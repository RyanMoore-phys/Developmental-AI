# External Minecraft server mode — full setup (both sides)

Point the organism's primary stream (env 0) at your OWN Minecraft server
instead of a MineRL-generated world. The AI still learns exactly as on MineRL
(same pixel observations, same actions, same discovery machinery); only the
world changes. The bot lives on the GPU pod and connects *out* to your server
over a private Tailscale tunnel — your server is never exposed to the internet.

```
[pod: java client] --127.0.0.1:25565--> [socat] --tailnet--> [your server:25565]
        env 0                              bridge      (ACL-caged, tag:devai)
```

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
tailscale ip -4            # note this 100.x.y.z — the pod bridges to it
```

### A5. (recommended) Cage the pod with an ACL

So a pod operator can never reach your other tailnet devices, set the tailnet
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
anything tagged `tag:devai` reach ONLY the game port. After approving the pod
(next section), tag it `tag:devai` on the Machines page.

---

## B. THE POD

Tailscale + socat are installed by provisioning (stage 1c). Bring the tunnel up
in two phases (the device-approval step is interactive):

```bash
# Phase 1 — login. Prints an approval URL.
bash scripts/connect_server.sh login
#   Open the URL in a browser on YOUR tailnet, approve "devai-pod",
#   then tag it tag:devai in the admin console (Machines page).

# Phase 2 — bridge to the server's tailscale IP (from A4), and ping it.
bash scripts/connect_server.sh bridge <server-tailscale-ip>
#   Success prints: "SERVER REACHABLE" with the server's version/protocol.
#   protocol 754 in the reply = a 1.16.5 client is accepted (Via is working).
```

Then launch external-server mode:

```bash
bash scripts/launch_skybot.sh 1000000
```

Its pre-flight refuses to start unless tailscaled + the bridge + the server ping
all pass, so a dead tunnel can't strand the primary in a connect-fail loop.

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
- A live server never resets, which suits the lifelong design (env 0 is a single
  continuous life); death is respawn, not a world reset.

## D. Config knobs (`configs/minecraft_skybot.yaml`)

```yaml
environment:
  remote_server: "127.0.0.1:25565"   # the pod-local bridge endpoint
  remote_server_scope: primary       # only env 0 joins; scouts stay local. "all" = every stream joins
```

Set `remote_server: null` (or use `configs/minecraft_lifelong.yaml`) to go back
to pure-MineRL local worlds — the SkyBot tunnel can stay running, idle.
