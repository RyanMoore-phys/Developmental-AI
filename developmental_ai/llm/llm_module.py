"""
LLM Integration — Local Open-Source Models via Ollama
=======================================================
Integrates a local LLM (Llama 3.1, Mistral, Qwen, etc.) into the
developmental AI loop via Ollama. Fully free and offline — no API keys,
no cloud dependency.

Setup:
  1. Install Ollama: brew install ollama  (macOS)
  2. Pull a model: ollama pull llama3.1:8b
  3. Start server: ollama serve  (runs on localhost:11434)
  4. pip install ollama

Three capabilities (called periodically, not every step):
  1. Rich Perception: LLM extracts structured symbolic facts from
     observation trajectories
  2. Goal Generation: LLM proposes exploration goals based on
     knowledge and skill state
  3. Causal Reasoning: LLM analyzes the knowledge graph for
     contradictions, hypotheses, and composition opportunities
"""

import concurrent.futures
import json
import logging
import os
import struct
import zlib
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)

try:
    import ollama as ollama_lib
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning(
        "Ollama SDK not installed. LLM features disabled. "
        "pip install ollama"
    )

# Hard wall-clock cap on a single Ollama request. The bare `ollama.generate`
# helper builds a client with NO request timeout, so if the local server stalls
# mid-generation (model reload, a dropped connection, an overloaded GPU) the
# socket read blocks in recvfrom *forever* — and because some goal-generation
# calls run synchronously on the training loop's critical path, that freezes the
# entire developmental loop. A generous timeout lets a genuine stall surface as a
# caught exception (→ return None → the loop falls back to its non-LLM goal path)
# instead of deadlocking. 120s comfortably clears even slow CPU generations.
_OLLAMA_TIMEOUT_S = 120.0
_ollama_client = None
# ---- WHERE THE VLM LIVES (2026-09-01, cluster migration) ------------------
# None = the bare module's own resolution (OLLAMA_HOST env, else
# http://127.0.0.1:11434) — i.e. exactly the shipped behaviour. Set via
# `llm.endpoint` in config to point the labeller at a DIFFERENT NODE. The VLM
# is the largest single GPU tenant and is measurably on the critical path
# (fovea_interval 12 cost 28% of the step rate), so on a cluster it is the
# first thing worth moving off the training device. The symbolizer is already
# async, so a remote endpoint costs latency, not throughput.
_ollama_host = None


def set_ollama_endpoint(host: Optional[str]) -> None:
    """Point every Ollama client at `host` (e.g. http://10.0.0.5:11434).

    Must be called before the first client is built; resets the cached client
    so a late call still takes effect rather than silently doing nothing.
    """
    global _ollama_host, _ollama_client
    if host == _ollama_host:
        return
    _ollama_host = host or None
    _ollama_client = None
    logger.info("Ollama endpoint set to %s",
                _ollama_host or "(default: OLLAMA_HOST or localhost:11434)")


def _get_ollama_client():
    """Lazily build (and cache) an Ollama client that enforces a request timeout."""
    global _ollama_client
    if _ollama_client is None and OLLAMA_AVAILABLE:
        # host=None → same default resolution as the bare module (OLLAMA_HOST env
        # or http://127.0.0.1:11434); timeout flows through to the httpx client.
        _ollama_client = ollama_lib.Client(
            host=_ollama_host, timeout=_OLLAMA_TIMEOUT_S)
    return _ollama_client


def _query_ollama(
    model: str,
    prompt: str,
    max_tokens: int = 1024,
) -> Optional[str]:
    """Send a prompt to the local Ollama server and return the response text.

    Uses a timeout-bearing client so a stalled server degrades gracefully
    (returns None) rather than blocking the training loop indefinitely.
    """
    if not OLLAMA_AVAILABLE:
        return None
    try:
        client = _get_ollama_client()
        response = client.generate(
            model=model,
            prompt=prompt,
            options={"num_predict": max_tokens, "temperature": 0.3},
        )
        return response.get("response", "").strip()
    except Exception as e:
        logger.warning(f"Ollama query failed: {e}")
        return None


def _parse_json_array(text: str) -> List[Any]:
    """Extract a JSON array from LLM response text."""
    if not text:
        return []
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return []


def _parse_json_object(text: str) -> Optional[Dict]:
    """Extract a JSON object from LLM response text."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def _check_ollama_server() -> bool:
    """Check if the Ollama server is running and reachable."""
    if not OLLAMA_AVAILABLE:
        return False
    try:
        ollama_lib.list()
        return True
    except Exception:
        return False


def _synthetic_probe_png(size: int = 64) -> bytes:
    """A tiny solid-gray grayscale PNG, built with stdlib `zlib`+`struct` only.

    No numpy/imageio dependency here (importing vlm_symbolizer for its
    encoder would be a backwards, circular import — it already imports FROM
    this module). 64x64 rather than 1x1: some ViT-tiling vision models choke
    on degenerate image sizes below their patch size, and the point of this
    probe is to catch exactly that class of failure, not paper over it with
    an input real usage would never send.
    """
    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 0, 0, 0, 0))
    row = bytes([0] + [128] * size)            # filter=none, mid-gray pixels
    raw = row * size
    idat = _chunk(b"IDAT", zlib.compress(raw, 6))
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


def probe_ollama_model(model: str, deep: bool = True) -> bool:
    """Is `model` actually present on the Ollama server? Log what we found.

    ---- WHY THIS IS LOUD AND DOES NOT FALL BACK (2026-09-01) -------------
    The VLM is a SENSOR. Swapping it silently is how this project spent a
    run on llava, whose tree_visible drifted to reliability 1.0 unopposed
    because the verifier cannot see false positives. When the configured tag
    is missing, Ollama's default behaviour is to try to pull it — which on a
    fresh cluster node either stalls the run or produces a DIFFERENT sensor
    from the one the config names, and neither shows up anywhere except as
    weird labels days later.

    So: say the resolved model out loud at startup, list what IS available,
    and return False. The caller decides; nothing is substituted here.
    Returns True when unverifiable (no ollama lib / server down) so this can
    never itself become the thing that stops a run.

    ---- STAGE 2, `deep=True` (2026-09-02) --------------------------------
    Tag PRESENCE is not the same claim as "the model actually runs" — a q4
    import can be truncated, a vision head can be missing from a GGUF
    conversion, or the server can OOM on load. All three pass stage 1 and
    then produce ten hours of silent zero-label runs, discovered only when
    someone asks "why are there no facts". Stage 2 sends one real generate
    call with a synthetic image through the exact code path production
    uses, before a single frame of the run depends on it. Failure here is
    just as loud as a missing tag and still never raises — the run must
    survive a broken VLM exactly as it survives a slow one; this just makes
    sure the survival is a MEASURED degradation, not a silent one.
    """
    if not OLLAMA_AVAILABLE:
        return True
    try:
        _resp = ollama_lib.list() or {}
        # ---- THE CLIENT RETURNS OBJECTS, NOT DICTS (fixed 2026-09-20) -----
        # Newer `ollama` python clients return pydantic models: `list()` gives
        # a ListResponse whose `.models` holds Model objects with a `.model`
        # attribute. The old code tested `isinstance(m, dict)`, fell through
        # to `str(m)`, and compared the REPR — so the available set contained
        #     "model='qwen2.5vl:3b' modified_at=datetime.datetime(...)"
        # which never equals "qwen2.5vl:3b".
        #
        # MEASURED CONSEQUENCE, and it is exactly the failure this whole
        # function was written to prevent: on a pod where `ollama list` showed
        # qwen2.5vl:3b present, every run logged VLM MODEL NOT FOUND and
        # proceeded with the labeller disabled — while Ollama kept 4.2 GB of
        # VRAM resident for a model nothing ever called. Zero labels, full
        # cost, and the only symptom was an absence of facts.
        if isinstance(_resp, dict):
            _models = _resp.get("models", _resp) or []
        else:
            _models = getattr(_resp, "models", None) or []
        names = set()
        for m in _models:
            if isinstance(m, dict):
                n = m.get("model") or m.get("name")
            elif isinstance(m, str):
                n = m
            else:
                # pydantic Model (or anything else exposing the field)
                n = getattr(m, "model", None) or getattr(m, "name", None)
            if n:
                names.add(str(n))
                names.add(str(n).split(":")[0])
    except Exception as e:
        logger.info("ollama model probe unavailable (%s) — continuing", e)
        return True
    if model not in names:
        logger.warning(
            "VLM MODEL NOT FOUND: %r is not on the Ollama server. "
            "Available: %s. The labeller is a SENSOR — a missing tag means "
            "either a stall on first use or a different model answering, "
            "and neither is visible later except as strange predicates. "
            "Build it with:\n"
            "  printf 'FROM qwen2.5vl:7b\\n' > /tmp/Modelfile && "
            "ollama create %s -f /tmp/Modelfile -q q4_K_M",
            model, sorted(names) or "(none)", model)
        return False
    logger.info("VLM resolved: %s (present on the Ollama server)", model)
    if not deep:
        return True
    try:
        client = _get_ollama_client()
        if client is None:
            return True             # unverifiable, not unhealthy
        resp = client.generate(
            model=model, prompt='Return {"ok": true} as JSON, nothing else.',
            images=[_synthetic_probe_png()], format="json", keep_alive=-1,
            options={"num_predict": 32, "temperature": 0.0})
        txt = (resp.get("response") or "").strip()
        json.loads(txt)            # must parse; content is not checked
        logger.info("VLM smoke check passed: %s answered a real generate "
                    "call with valid JSON", model)
        return True
    except Exception as e:
        logger.warning(
            "VLM SMOKE CHECK FAILED for %s: %s. The tag is present but a "
            "real generate call did not return valid JSON — this is "
            "usually a truncated import, a missing vision head, or the "
            "server rejecting the request, and none of those show up in "
            "`ollama list`. The run continues (min_labels keeps a silent "
            "VLM safe), but expect zero grounded facts until this is fixed.",
            model, e)
        return False


class LLMPerception:
    """
    Uses a local LLM to extract rich symbolic facts from observation
    trajectories. Called periodically on batches of recent experience.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        env_description: str = "",
        obs_labels: Optional[List[str]] = None,
        call_interval: int = 50,
    ):
        self.model = model
        self.env_description = env_description
        self.obs_labels = obs_labels or []
        self.call_interval = call_interval
        self._available = _check_ollama_server()
        self.total_calls = 0

        if not self._available and OLLAMA_AVAILABLE:
            logger.info(
                "Ollama server not running. Start with: ollama serve"
            )

    @property
    def is_available(self) -> bool:
        return self._available

    def extract_facts(
        self,
        recent_observations: List[Dict[str, Any]],
        recent_actions: List[str],
        recent_rewards: List[float],
        knowledge_summary: str = "",
    ) -> List[Dict[str, str]]:
        if not self.is_available:
            return []

        obs_text = self._format_observations(
            recent_observations, recent_actions, recent_rewards
        )

        prompt = (
            f"You are analyzing observations from an RL agent in:\n"
            f"{self.env_description}\n\n"
            f"Observation labels: {', '.join(self.obs_labels)}\n\n"
            f"Recent trajectory (last {len(recent_observations)} steps):\n"
            f"{obs_text}\n\n"
            f"Current knowledge: {knowledge_summary or 'None yet'}\n\n"
            f"Extract symbolic facts as (subject, relation, object) triples. "
            f"Focus on causal relationships and state descriptions.\n\n"
            f"Return ONLY a JSON array of objects with keys: subject, "
            f"relation, object, confidence (0.0-1.0).\n"
            f'Example: [{{"subject": "push_right", "relation": "causes", '
            f'"object": "cart_moves_right", "confidence": 0.9}}]\n'
            f"Return at most 10 facts. Return [] if nothing notable."
        )

        text = _query_ollama(self.model, prompt)
        facts = _parse_json_array(text)
        if facts:
            self.total_calls += 1
        return facts

    def _format_observations(
        self,
        observations: List[Dict[str, Any]],
        actions: List[str],
        rewards: List[float],
    ) -> str:
        lines = []
        for i, (obs, act, rew) in enumerate(
            zip(observations, actions, rewards)
        ):
            obs_str = ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in obs.items()
            )
            lines.append(
                f"  Step {i}: [{obs_str}] -> action={act} -> reward={rew:.2f}"
            )
        return "\n".join(lines[-20:])


class LLMGoalGenerator:
    """
    Uses a local LLM to generate meaningful exploration goals based on
    the agent's current knowledge and skill repertoire.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        env_description: str = "",
        obs_labels: Optional[List[str]] = None,
    ):
        self.model = model
        self.env_description = env_description
        self.obs_labels = obs_labels or []
        self._available = _check_ollama_server()
        self.goal_history: List[str] = []
        self.total_calls = 0

    @property
    def is_available(self) -> bool:
        return self._available

    def generate_goal(
        self,
        knowledge_summary: str,
        skill_summary: str,
        recent_performance: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Returns dict with 'description', 'target_dims', 'strategy',
        'reasoning', or None if unavailable.
        """
        if not self.is_available:
            return None

        recent_goals = (
            ", ".join(self.goal_history[-5:]) if self.goal_history else "None"
        )

        prompt = (
            f"You are the goal-setting module for an AI agent learning in:\n"
            f"{self.env_description}\n\n"
            f"Observable dimensions: {', '.join(self.obs_labels)}\n\n"
            f"Knowledge: {knowledge_summary}\n"
            f"Skills: {skill_summary}\n"
            f"Performance: {recent_performance}\n"
            f"Recent goals (avoid repeating): {recent_goals}\n\n"
            f"Propose one exploration goal. Return ONLY a JSON object:\n"
            f'{{"description": "one-sentence goal", '
            f'"target_dims": ["dim1", "dim2"], '
            f'"strategy": "explore" or "refine" or "compose", '
            f'"reasoning": "one sentence why"}}'
        )

        text = _query_ollama(self.model, prompt, max_tokens=512)
        goal = _parse_json_object(text)
        if goal:
            self.goal_history.append(goal.get("description", ""))
            self.total_calls += 1
        return goal


class LLMReasoning:
    """
    Uses a local LLM for causal reasoning over the knowledge graph.
    Identifies contradictions, hypotheses, and composition opportunities.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        env_description: str = "",
    ):
        self.model = model
        self.env_description = env_description
        self._available = _check_ollama_server()
        self.total_calls = 0

    @property
    def is_available(self) -> bool:
        return self._available

    def analyze_knowledge(
        self,
        facts: List[Dict[str, str]],
        action_rules: List[Dict[str, Any]],
        skills: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self.is_available:
            return None

        facts_text = "\n".join(
            f"  ({f['subject']}, {f['relation']}, {f['object']})"
            for f in facts[:50]
        )
        rules_text = "\n".join(
            f"  IF {r['preconditions']} AND action={r['action']} "
            f"THEN {r['effects']} (conf={r.get('confidence', 0):.2f})"
            for r in action_rules[:20]
        )
        skills_text = "\n".join(
            f"  {s['name']} (mastery={s.get('mastery_level', 0):.2f})"
            for s in skills
        )

        prompt = (
            f"Analyze this AI agent's knowledge ({self.env_description}):\n\n"
            f"Facts ({len(facts)} total):\n{facts_text}\n\n"
            f"Action rules ({len(action_rules)} total):\n{rules_text}\n\n"
            f"Skills:\n{skills_text}\n\n"
            f"Return ONLY a JSON object:\n"
            f'{{"contradictions": [], '
            f'"hypotheses": ["hypothesis1", "hypothesis2"], '
            f'"composition_suggestions": '
            f'[{{"skill_a": "name", "skill_b": "name", "rationale": "why"}}], '
            f'"curriculum_ready": true, '
            f'"reasoning": "brief assessment"}}'
        )

        text = _query_ollama(self.model, prompt)
        analysis = _parse_json_object(text)
        if analysis:
            self.total_calls += 1
        return analysis

    def suggest_skill_compositions(
        self,
        skills: List[Dict[str, Any]],
        knowledge_facts: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        if not self.is_available:
            return []

        skills_text = "\n".join(
            f"  - {s['name']}: {s.get('description', '')} "
            f"(goal_facts: {s.get('goal_facts', [])})"
            for s in skills
        )

        prompt = (
            f"Given these mastered skills:\n{skills_text}\n\n"
            f"Which pairs could compose into more complex behaviors?\n\n"
            f"Return ONLY a JSON array:\n"
            f'[{{"skill_a": "name", "skill_b": "name", '
            f'"type": "sequential", "rationale": "why"}}]\n'
            f"Return [] if no compositions make sense."
        )

        text = _query_ollama(self.model, prompt, max_tokens=512)
        return _parse_json_array(text)


class LLMRewardShaper:
    """
    Uses a local LLM to judge how much an episode advanced the agent's CURRENT
    developmental goal, returning a scalar progress estimate in [0, 1].

    This is the "Path B" slice: the LLM partially guides reward shaping so the
    agent's learning stays pointed at the goal. Two deliberate safety choices:

      1. It scores GOAL PROGRESS, not open-ended "good behavior" — far harder
         to game, and it ties the signal to the self-generated goal.
      2. The score is consumed downstream as a *potential-based, annealed*
         shaping term: it can guide the policy but provably cannot redefine the
         task's optimum, and it fades toward zero as training proceeds. So it
         never becomes a fixed reward ratio.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        env_description: str = "",
    ):
        self.model = model
        self.env_description = env_description
        self._available = _check_ollama_server()
        self.total_calls = 0

    @property
    def is_available(self) -> bool:
        return self._available

    def score_progress(
        self,
        goal_description: str,
        trajectory_summary: str,
    ) -> Optional[float]:
        """Return goal-progress in [0,1], or None if unavailable/unparseable."""
        if not self.is_available:
            return None

        prompt = (
            f"You are judging an RL agent's progress toward a goal in:\n"
            f"{self.env_description}\n\n"
            f"Current goal: {goal_description or 'make progress on the task'}\n\n"
            f"Recent performance:\n{trajectory_summary}\n\n"
            f"Rate how much the agent has progressed toward the goal, from "
            f"0.0 (no progress / stuck) to 1.0 (goal essentially achieved). "
            f"Judge progress toward THIS goal only, not general activity.\n"
            f'Return ONLY a JSON object: '
            f'{{"progress": 0.0, "reasoning": "one sentence"}}'
        )

        text = _query_ollama(self.model, prompt, max_tokens=256)
        obj = _parse_json_object(text)
        if not obj or "progress" not in obj:
            return None
        try:
            p = float(obj["progress"])
        except (TypeError, ValueError):
            return None
        self.total_calls += 1
        return max(0.0, min(1.0, p))


class _AsyncChannel:
    """
    A single-slot async work channel that decouples an LLM call from the
    training loop.

    The problem it solves: a local Ollama generation on CPU can take several
    SECONDS. Calling it inline stalls the whole developmental loop. This
    channel runs the call on a background worker thread instead — the loop
    SUBMITS a job (non-blocking) and POLLS for the result on a later episode.

    "Single-slot" means at most one job is in flight per channel; if a job is
    still running when the loop tries to submit another, the submit is skipped.
    This naturally throttles the LLM to the rate the loop can consume results,
    and guarantees we never queue up a backlog of stale work.

    When async_mode is False the channel runs the call inline (blocking) and
    buffers the result for the next poll, so callers can use the same
    submit()/poll() interface regardless of mode.
    """

    def __init__(self, executor: Optional["concurrent.futures.ThreadPoolExecutor"], async_mode: bool = True):
        self._executor = executor
        self._async = async_mode and executor is not None
        self._future: Optional["concurrent.futures.Future"] = None
        self._inline_result: Any = None
        self._has_inline = False

    def busy(self) -> bool:
        if self._async:
            return self._future is not None and not self._future.done()
        return False

    def submit(self, fn: Callable, *args, **kwargs) -> bool:
        """Submit a job. Returns False if a job is already in flight.

        Refuses to overwrite a future that is done-but-unpolled: busy() reads
        False for such a future, so a poll()+submit() in the same caller step
        could otherwise silently discard a just-finished result (the caller
        polls next step). Returns False so the completed result survives.
        """
        if self._async:
            if self._future is not None and not self._future.done():
                return False  # still running
            if self._future is not None:  # done but not yet polled
                return False  # let the caller poll it before a new submit
            self._future = self._executor.submit(fn, *args, **kwargs)
            return True
        # Synchronous fallback — run inline, buffer for poll().
        try:
            self._inline_result = fn(*args, **kwargs)
        except Exception as e:  # never let an LLM error crash the loop
            logger.warning(f"LLM call failed: {e}")
            self._inline_result = None
        self._has_inline = True
        return True

    def poll(self) -> Any:
        """Return a finished result, or None if nothing is ready yet."""
        if self._async:
            if self._future is not None and self._future.done():
                try:
                    result = self._future.result()
                except Exception as e:
                    logger.warning(f"LLM background call failed: {e}")
                    result = None
                self._future = None
                return result
            return None
        if self._has_inline:
            self._has_inline = False
            result, self._inline_result = self._inline_result, None
            return result
        return None


class LLMModule:
    """
    Unified LLM integration using Ollama (local, free, open-source).

    Setup:
        brew install ollama
        ollama pull llama3.1:8b
        ollama serve
        pip install ollama

    Usage in the developmental loop:
        llm = LLMModule(model="llama3.1:8b", ...)
        if llm.should_generate_goal(episode):
            goal = llm.goal_generator.generate_goal(...)
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        env_description: str = "",
        obs_labels: Optional[List[str]] = None,
        perception_interval: int = 50,
        goal_interval: int = 25,
        reasoning_interval: int = 100,
        shaping_interval: int = 25,
        async_mode: bool = True,
        enabled: bool = True,
    ):
        self.perception = LLMPerception(
            model=model,
            env_description=env_description,
            obs_labels=obs_labels or [],
            call_interval=perception_interval,
        )
        self.goal_generator = LLMGoalGenerator(
            model=model,
            env_description=env_description,
            obs_labels=obs_labels or [],
        )
        self.reasoning = LLMReasoning(
            model=model,
            env_description=env_description,
        )
        self.reward_shaper = LLMRewardShaper(
            model=model,
            env_description=env_description,
        )
        # Hard kill-switch: when the config sets llm.enabled=False we must NOT
        # touch Ollama at all. Each sub-component independently probes the local
        # server in __init__, so if Ollama happens to be running they would all
        # report is_available=True and fire anyway (ungrounded goals, blocking
        # calls). Force every channel offline so should_*/submit_* short-circuit.
        self.enabled = enabled
        if not enabled:
            self.perception._available = False
            self.goal_generator._available = False
            self.reasoning._available = False
            self.reward_shaper._available = False

        self.perception_interval = perception_interval
        self.goal_interval = goal_interval
        self.reasoning_interval = reasoning_interval
        self.shaping_interval = shaping_interval

        # ---- Async decoupling from the training loop ----
        # One shared single-worker executor: a local Ollama server processes
        # one generation at a time anyway, so a single background thread is the
        # right amount of concurrency. Three channels keep perception/goal/
        # reasoning jobs independent (one in-flight slot each).
        self.async_mode = async_mode
        self._executor: Optional[concurrent.futures.ThreadPoolExecutor] = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="llm"
            )
            if async_mode
            else None
        )
        self.perception_channel = _AsyncChannel(self._executor, async_mode)
        self.goal_channel = _AsyncChannel(self._executor, async_mode)
        self.reasoning_channel = _AsyncChannel(self._executor, async_mode)
        self.progress_channel = _AsyncChannel(self._executor, async_mode)

    # ---- Async submit/poll API (non-blocking from the loop's perspective) ----

    def submit_facts(
        self,
        recent_observations: List[Dict[str, Any]],
        recent_actions: List[str],
        recent_rewards: List[float],
        knowledge_summary: str = "",
    ) -> bool:
        """Queue a fact-extraction job. Inputs are snapshots taken on the main
        thread, so the worker never reads mutating environment state."""
        if not self.perception.is_available:
            return False
        return self.perception_channel.submit(
            self.perception.extract_facts,
            recent_observations,
            recent_actions,
            recent_rewards,
            knowledge_summary,
        )

    def poll_facts(self) -> Optional[List[Dict[str, str]]]:
        return self.perception_channel.poll()

    def submit_goal(self, **kwargs) -> bool:
        if not self.goal_generator.is_available:
            return False
        return self.goal_channel.submit(self.goal_generator.generate_goal, **kwargs)

    def poll_goal(self) -> Optional[Dict[str, Any]]:
        return self.goal_channel.poll()

    def submit_analysis(self, facts, action_rules, skills) -> bool:
        if not self.reasoning.is_available:
            return False
        return self.reasoning_channel.submit(
            self.reasoning.analyze_knowledge, facts, action_rules, skills
        )

    def poll_analysis(self) -> Optional[Dict[str, Any]]:
        return self.reasoning_channel.poll()

    def submit_progress(
        self, goal_description: str, trajectory_summary: str
    ) -> bool:
        """Queue an async goal-progress scoring job (Path B reward shaping)."""
        if not self.reward_shaper.is_available:
            return False
        return self.progress_channel.submit(
            self.reward_shaper.score_progress,
            goal_description,
            trajectory_summary,
        )

    def poll_progress(self) -> Optional[float]:
        return self.progress_channel.poll()

    def close(self) -> None:
        """Shut down the background executor (call on agent shutdown)."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    @property
    def is_available(self) -> bool:
        return (
            self.perception.is_available
            or self.goal_generator.is_available
            or self.reasoning.is_available
        )

    def should_extract_facts(self, episode: int) -> bool:
        return (
            self.perception.is_available
            and episode > 0
            and episode % self.perception_interval == 0
        )

    def should_generate_goal(self, episode: int) -> bool:
        return (
            self.goal_generator.is_available
            and episode > 0
            and episode % self.goal_interval == 0
        )

    def should_score_progress(self, episode: int) -> bool:
        return (
            self.reward_shaper.is_available
            and episode > 0
            and episode % self.shaping_interval == 0
        )

    def should_analyze(self, episode: int) -> bool:
        return (
            self.reasoning.is_available
            and episode > 0
            and episode % self.reasoning_interval == 0
        )

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "llm_available": self.is_available,
            "perception_calls": self.perception.total_calls,
            "goal_calls": self.goal_generator.total_calls,
            "reasoning_calls": self.reasoning.total_calls,
        }
