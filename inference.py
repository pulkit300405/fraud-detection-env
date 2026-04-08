"""
inference.py - Fraud Detection RL Environment Baseline
"""
import os
import sys
import requests
from typing import List, Optional
from openai import OpenAI

# ── Debug: confirm env vars are injected by Meta's validator ──────────────────
print(f"[DEBUG] API_BASE_URL={os.environ.get('API_BASE_URL', 'NOT SET')}", flush=True, file=sys.stderr)
print(f"[DEBUG] API_KEY={'SET' if os.environ.get('API_KEY') else 'NOT SET'}", flush=True, file=sys.stderr)

# ── Config ────────────────────────────────────────────────────────────────────
# Use ONLY env vars injected by Meta's LiteLLM proxy — never hardcode
API_KEY      = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]

# meta-llama/Llama-3.3-70B-Instruct is available on HuggingFace router
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
ENV_URL      = os.getenv("ENV_URL", "http://localhost:8000")
BENCHMARK    = "fraud_detect_env"
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS    = ["flag-obvious", "explain-subtle", "adversarial-hunt"]
MAX_STEPS = {"flag-obvious": 3, "explain-subtle": 5, "adversarial-hunt": 8}

SYSTEM_PROMPT = """You are a fraud detection agent investigating suspicious app sessions.
You will receive logs of user behavior and must determine if the session is fraudulent.

Available actions:
- investigate:ip_velocity
- investigate:device_fingerprint
- investigate:login_frequency
- investigate:geo_anomaly
- investigate:request_pattern
- verdict:fraud
- verdict:real

Strategy: investigate signals first to gather evidence, then give a verdict.
Reply with ONLY the action string, nothing else. Example: investigate:ip_velocity"""


# ── Logging helpers ───────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM action selection via Meta's LiteLLM proxy ────────────────────────────
def get_action(client: OpenAI, obs: dict, step: int) -> str:
    logs_summary = f"Session {obs.get('session_id')} has {len(obs.get('logs', []))} log entries."
    signals   = obs.get("signals_revealed", {})
    available = obs.get("available_actions", [])

    user_prompt = (
        f"Step {step} - Fraud Investigation\n"
        f"Task: {obs.get('task')} ({obs.get('difficulty')})\n"
        f"{logs_summary}\n"
        f"Signals revealed so far: {signals}\n"
        f"Available actions: {available}\n"
        f"Hint: {obs.get('context_hint', 'None')}\n\n"
        f"Choose your next action from available actions."
    )

    # This call MUST go through API_BASE_URL (Meta's LiteLLM proxy)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=50,
    )
    action = (completion.choices[0].message.content or "").strip()

    # Validate — fall back to first investigate action if model output is invalid
    if action in available:
        return action
    for a in available:
        if a.startswith("investigate:"):
            return a
    return "verdict:fraud"


# ── Episode runner ────────────────────────────────────────────────────────────
def run_episode(client: OpenAI, task: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        resp = requests.post(f"{ENV_URL}/reset", json={}, timeout=30)
        resp.raise_for_status()
        obs = resp.json().get("observation", resp.json())

        for step in range(1, MAX_STEPS[task] + 1):
            if obs.get("done", False):
                break

            action = get_action(client, obs, step)

            resp = requests.post(
                f"{ENV_URL}/step",
                json={"action": {"action": action, "reasoning": "LLM agent decision"}},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            obs = result.get("observation", result)

            reward      = float(obs.get("reward", 0.0))
            done        = bool(obs.get("done", False))
            error       = (obs.get("metadata") or {}).get("last_error")

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action, reward=reward, done=done, error=error)

            if done:
                break

        total_reward = sum(rewards)
        score   = min(max((total_reward + 3.0) / 5.0, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[ERROR] Episode failed: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    # Remove any stale OpenAI env vars that could bypass the proxy
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL"):
        os.environ.pop(key, None)

    print(f"[INFO] Using API_BASE_URL={API_BASE_URL}", flush=True)
    print(f"[INFO] Using MODEL_NAME={MODEL_NAME}", flush=True)

    # Build client pointing exclusively at Meta's LiteLLM proxy
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    for task in TASKS:
        run_episode(client, task)


if __name__ == "__main__":
    main()