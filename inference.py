"""
inference.py - Fraud Detection RL Environment Baseline
"""
import os
import requests
from typing import List, Optional
from openai import OpenAI

API_KEY = os.environ["API_KEY"]
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")
BENCHMARK = "fraud_detect_env"
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = ["flag-obvious", "explain-subtle", "adversarial-hunt"]
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


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_action(client: OpenAI, obs: dict, step: int) -> str:
    logs_summary = f"Session {obs.get('session_id')} has {len(obs.get('logs', []))} log entries."
    signals = obs.get('signals_revealed', {})
    available = obs.get('available_actions', [])
    
    user_prompt = f"""Step {step} - Fraud Investigation
Task: {obs.get('task')} ({obs.get('difficulty')})
{logs_summary}
Signals revealed so far: {signals}
Available actions: {available}
Hint: {obs.get('context_hint', 'None')}

Choose your next action from available actions."""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=50,
        )
        action = (completion.choices[0].message.content or "").strip()
        # validate action is in available list
        if action in available:
            return action
        # fallback: pick first investigate, else verdict
        for a in available:
            if a.startswith("investigate:"):
                return a
        return "verdict:fraud"
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        raise


def run_episode(client: OpenAI, task: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        # reset
        resp = requests.post(f"{ENV_URL}/reset", json={}, timeout=30)
        obs = resp.json().get("observation", resp.json())

        for step in range(1, MAX_STEPS[task] + 1):
            if obs.get("done", False):
                break

            action = get_action(client, obs, step)

            resp = requests.post(f"{ENV_URL}/step",
                json={"action": {"action": action, "reasoning": "LLM agent decision"}},
                timeout=30)
            result = resp.json()
            obs = result.get("observation", result)

            reward = float(obs.get("reward", 0.0))
            done = bool(obs.get("done", False))
            error = obs.get("metadata", {}).get("last_error") if obs.get("metadata") else None

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action, reward=reward, done=done, error=error)

            if done:
                break

        # score based on grader
        total_reward = sum(rewards)
        score = min(max((total_reward + 3.0) / 5.0, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task in TASKS:
        run_episode(client, task)


if __name__ == "__main__":
    main()