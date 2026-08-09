"""FAST smoke for the remote-server env mode (no client launch).

Checks the three links of the chain that can be verified offline:
  1. the RemoteServer handler emits the exact XML element the MCP-Reborn
     client's EnvServer.getServerAddress() reads (<RemoteServer>ip:port</...>),
  2. MineRLEnvAdapter accepts + stores remote_server (and defaults to None),
  3. make_env passes remote_server through to the adapter (signature check).

The LIVE join is proven by tests/_remote_join_probe.py (launches a real
client on the pod; needs the external server reachable + whitelist open).

Run: PYTHONPATH=. python tests/_remote_server_smoke.py   (venv_mc on pod)
"""
import inspect
import sys

sys.path.insert(0, ".")


def main() -> None:
    # 1. handler XML matches what EnvServer.java parses
    from minerl.herobraine.hero.handlers.server.world import RemoteServer
    xml = RemoteServer("127.0.0.1:25565").xml_template()
    assert xml == "<RemoteServer>127.0.0.1:25565</RemoteServer>", xml
    # address must be a string — the handler raises on anything else
    try:
        RemoteServer(1234).xml_template()
        raise AssertionError("non-string address should raise")
    except ValueError:
        pass

    # 2. adapter signature + default-off
    from developmental_ai.environments.minerl_env import MineRLEnvAdapter
    sig = inspect.signature(MineRLEnvAdapter.__init__)
    assert "remote_server" in sig.parameters, "adapter missing remote_server"
    assert sig.parameters["remote_server"].default is None

    # 3. make_env plumbs it through
    from developmental_ai.environments.wrappers import make_env
    assert "remote_server" in inspect.signature(make_env).parameters

    # 4. the spec closure wires the handler between Time and Spawning: verify
    #    by source inspection (constructing the spec launches a client, which
    #    the fast smoke must not do).
    import developmental_ai.environments.minerl_env as m
    src = inspect.getsource(m.MineRLEnvAdapter._make_underlying)
    assert "RemoteServer" in src and "conds.append(RemoteServer(_remote))" in src
    t = src.index("TimeInitialCondition")
    r = src.index("conds.append(RemoteServer(_remote))")
    s = src.index("SpawningInitialCondition", r)
    assert t < r < s, "RemoteServer must sit between Time and Spawning (JAXB order)"

    # 5. auto-rejoin watchdog (_remote_frozen) contracts — instantiate the
    #    adapter WITHOUT __init__ (constructing it launches a Java client)
    #    and drive the detector directly.
    import numpy as np
    from developmental_ai.environments.minerl_env import TREECHOP_MACROS

    def fresh(remote):
        a = object.__new__(MineRLEnvAdapter)
        a.remote_server = remote
        a._frozen_limit = 5              # small for the test
        a._frozen_count = 0
        a._prev_pov_digest = None
        a._motion_macros = frozenset(
            i for i, m in enumerate(TREECHOP_MACROS)
            if "camera" in m or "forward" in m or "back" in m)
        return a

    pov_a = {"pov": np.zeros((8, 8, 3), np.uint8)}
    pov_b = {"pov": np.ones((8, 8, 3), np.uint8)}
    MOTION, NOOP, ATTACK = 1, 0, 5

    # local mode: identical frames never trigger
    a = fresh(None)
    assert not any(a._remote_frozen(pov_a, MOTION) for _ in range(50))

    # remote: frozen frames + motion actions trigger at the limit
    a = fresh("127.0.0.1:25565")
    a._remote_frozen(pov_a, MOTION)      # seeds the digest (count 0)
    hits = [a._remote_frozen(pov_a, MOTION) for _ in range(5)]
    assert hits == [False, False, False, False, True], hits

    # remote: frozen frames but only noop/attack never trigger
    a = fresh("127.0.0.1:25565")
    assert not any(a._remote_frozen(pov_a, NOOP) or
                   a._remote_frozen(pov_a, ATTACK) for _ in range(30))

    # remote: a changing frame resets the streak
    a = fresh("127.0.0.1:25565")
    a._remote_frozen(pov_a, MOTION)
    for _ in range(4):
        a._remote_frozen(pov_a, MOTION)
    assert a._frozen_count == 4
    a._remote_frozen(pov_b, MOTION)      # live frame -> reset
    assert a._frozen_count == 0

    print("[remote-server-smoke] ALL PASS: handler XML exact, adapter+make_env "
          "plumbed, JAXB element order preserved, frozen-screen watchdog "
          "triggers/resets correctly")


if __name__ == "__main__":
    main()
