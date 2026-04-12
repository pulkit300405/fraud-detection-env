"""
Fraud Detect Env — inference.py (Baseline Agent)
=================================================
MANDATORY COMPLIANCE (modelled on passing AntiGravity submission):
- API_BASE_URL : LLM endpoint injected by Meta's validator
- MODEL_NAME   : Model identifier
- HF_TOKEN     : API Key injected by Meta's validator
- TASK_NAME    : Optional task override injected by grader

STDOUT FORMAT — strictly required:
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import json
import re
import sys
import httpx
from openai import OpenAI

# ── Environment Configuration ─────────────────────────────────────────────────
# Use os.getenv() with fallbacks — exactly like the passing submission
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
HF_TOKEN     = os.getenv("HF_TOKEN")

# HF_TOKEN is what Meta injects as the API key
API_KEY = HF_TOKEN

OPENENV_URL  = os.getenv("OPENENV_URL", "https://Pulkit3004-fraud-detect-env.hf.space").rstrip("/")
BENCHMARK    = "fraud_detect_env"

if not API_KEY:
    print("WARNING: HF_TOKEN not set. LLM calls may fail.", file=sys.stderr)

# Build client pointing at Meta's LiteLLM proxy via API_BASE_URL
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "sk-no-key-set")


# ── Structured Log Helpers ────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Internal Helpers ──────────────────────────────────────────────────────────
def _llm_call(system_prompt: str, user_prompt: str) -> str:
    """All LLM calls go through API_BASE_URL — never bypass this."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=500,
        )
        return completion.choices[0].message.content or ""
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}", file=sys.stderr)
        return ""


def _post(path: str, body: dict) -> dict:
    """HTTP POST to the OpenEnv environment."""
    r = httpx.post(f"{OPENENV_URL}{path}", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def _extract_json(raw: str) -> dict:
    """Extract JSON from CoT thinking or markdown blocks."""
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return json.loads(match.group(1))
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return json.loads(text)


# ── Task Runners ──────────────────────────────────────────────────────────────
def run_easy() -> tuple[float, int]:
    """Task: flag-obvious — obvious fraud with multiple IPs/countries."""
    obs = _post("/reset", {})
    obs = obs.get("observation", obs)

    logs_text = json.dumps(obs.get("logs", [])[:5], indent=2)
    signals   = obs.get("signals_revealed", {})
    available = obs.get("available_actions", [])

    system = (
        "You are a fraud detection agent. Investigate app session logs and detect fraud.\n"
        "Available actions: investigate:ip_velocity, investigate:device_fingerprint, "
        "investigate:login_frequency, investigate:geo_anomaly, investigate:request_pattern, "
        "verdict:fraud, verdict:real\n"
        "Reply with ONLY the action string. Example: investigate:ip_velocity"
    )

    rewards = []
    step_count = 0

    # Investigate one signal first
    for step in range(1, 4):
        user = (
            f"Step {step}\nSession logs (sample):\n{logs_text}\n"
            f"Signals revealed: {signals}\nAvailable: {available}\n"
            f"Hint: {obs.get('context_hint', 'Look for obvious fraud signals.')}\n"
            "Choose your action:"
        )
        raw    = _llm_call(system, user)
        action = raw.strip().split("\n")[0].strip()

        # Validate action
        if action not in available:
            for a in available:
                if a.startswith("investigate:"):
                    action = a
                    break
            else:
                action = "verdict:fraud"

        try:
            res = _post("/step", {"action": {"action": action, "reasoning": "LLM agent investigation"}})
            obs     = res.get("observation", res)
            reward  = float(obs.get("reward", 0.0))
            done    = bool(obs.get("done", False))
            error   = (obs.get("metadata") or {}).get("last_error")
            signals = obs.get("signals_revealed", signals)
            available = obs.get("available_actions", available)
        except Exception as e:
            reward, done, error = 0.0, True, str(e)[:80]

        rewards.append(reward)
        step_count = step
        log_step(step=step, action=action, reward=reward, done=done, error=error)

        if done:
            break

    score   = min(max((sum(rewards) + 3.0) / 5.0, 0.0), 1.0)
    return score, step_count


def run_medium() -> tuple[float, int]:
    """Task: explain-subtle — subtle patterns requiring deeper investigation."""
    obs = _post("/reset", {})
    obs = obs.get("observation", obs)

    logs_text = json.dumps(obs.get("logs", []), indent=2)
    signals   = obs.get("signals_revealed", {})
    available = obs.get("available_actions", [])

    system = (
        "You are a fraud detection agent investigating subtle fraud patterns.\n"
        "Investigate multiple signals before giving a verdict.\n"
        "Reply with ONLY the action string."
    )

    rewards    = []
    step_count = 0

    for step in range(1, 6):
        user = (
            f"Step {step}/5\nLogs:\n{logs_text}\n"
            f"Signals revealed so far: {signals}\nAvailable actions: {available}\n"
            "Investigate systematically then give verdict:"
        )
        raw    = _llm_call(system, user)
        action = raw.strip().split("\n")[0].strip()

        if action not in available:
            investigates = [a for a in available if a.startswith("investigate:")]
            action = investigates[0] if investigates else "verdict:fraud"

        try:
            res = _post("/step", {"action": {"action": action, "reasoning": "Systematic investigation"}})
            obs      = res.get("observation", res)
            reward   = float(obs.get("reward", 0.0))
            done     = bool(obs.get("done", False))
            error    = (obs.get("metadata") or {}).get("last_error")
            signals  = obs.get("signals_revealed", signals)
            available = obs.get("available_actions", available)
        except Exception as e:
            reward, done, error = 0.0, True, str(e)[:80]

        rewards.append(reward)
        step_count = step
        log_step(step=step, action=action, reward=reward, done=done, error=error)

        if done:
            break

    score = min(max((sum(rewards) + 3.0) / 5.0, 0.0), 1.0)
    return score, step_count


def run_hard() -> tuple[float, int]:
    """Task: adversarial-hunt — adversarial fraud mimicking normal behavior."""
    obs = _post("/reset", {})
    obs = obs.get("observation", obs)

    logs_text = json.dumps(obs.get("logs", []), indent=2)
    signals   = obs.get("signals_revealed", {})
    available = obs.get("available_actions", [])

    system = (
        "You are an expert fraud analyst hunting adversarial fraud.\n"
        "The session may look legitimate but contains subtle adversarial patterns.\n"
        "Investigate ALL available signals before giving a verdict.\n"
        "Reply with ONLY the action string."
    )

    rewards    = []
    step_count = 0

    for step in range(1, 9):
        user = (
            f"Step {step}/8 — Adversarial Hunt\nLogs:\n{logs_text}\n"
            f"Signals revealed: {signals}\nAvailable: {available}\n"
            "Investigate all signals thoroughly:"
        )
        raw    = _llm_call(system, user)
        action = raw.strip().split("\n")[0].strip()

        if action not in available:
            investigates = [a for a in available if a.startswith("investigate:")]
            action = investigates[0] if investigates else "verdict:fraud"

        try:
            res = _post("/step", {"action": {"action": action, "reasoning": "Deep adversarial investigation"}})
            obs      = res.get("observation", res)
            reward   = float(obs.get("reward", 0.0))
            done     = bool(obs.get("done", False))
            error    = (obs.get("metadata") or {}).get("last_error")
            signals  = obs.get("signals_revealed", signals)
            available = obs.get("available_actions", available)
        except Exception as e:
            reward, done, error = 0.0, True, str(e)[:80]

        rewards.append(reward)
        step_count = step
        log_step(step=step, action=action, reward=reward, done=done, error=error)

        if done:
            break

    score = min(max((sum(rewards) + 3.0) / 5.0, 0.0), 1.0)
    return score, step_count


# ── Entry Point ───────────────────────────────────────────────────────────────
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all", help="Task to run, or 'all'")
    args = parser.parse_args()

    # TASK_NAME env var injected by hackathon grader overrides CLI
    task_override = os.getenv("TASK_NAME")
    if task_override:
        args.task = task_override

    task_mappings = [
        ("flag-obvious",    run_easy),
        ("explain-subtle",  run_medium),
        ("adversarial-hunt", run_hard),
    ]

    if args.task != "all":
        task_mappings = [t for t in task_mappings if t[0] == args.task]
        if not task_mappings:
            print(
                f"Error: Unknown task '{args.task}'. "
                "Valid: flag-obvious, explain-subtle, adversarial-hunt, all",
                file=sys.stderr,
            )
            return

    for task_id, runner in task_mappings:
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        task_rewards: list[float] = []
        try:
            score, steps = runner()
            task_rewards.append(score)
            success = score > 0.5
            log_end(success=success, steps=steps, score=score, rewards=task_rewards)
        except Exception as e:
            log_step(step=1, action=f"{task_id}_failed", reward=0.0, done=True, error=str(e)[:80])
            log_end(success=False, steps=1, score=0.0, rewards=[0.0])
        print("")  # separator between tasks


if __name__ == "__main__":
    main()