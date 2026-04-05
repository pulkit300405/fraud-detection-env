"""
tasks.py - task configs and graders
"""
from typing import Any, Dict, Optional

TASK_CONFIGS = {
    "flag-obvious": {
        "difficulty": "easy",
        "max_steps": 3,
        "hint": "Look for multiple IPs or countries — obvious fraud leaves clear traces.",
    },
    "explain-subtle": {
        "difficulty": "medium",
        "max_steps": 5,
        "hint": None,
    },
    "adversarial-hunt": {
        "difficulty": "hard",
        "max_steps": 8,
        "hint": None,
    },
}

SIGNALS = ["ip_velocity", "device_fingerprint", "login_frequency", "geo_anomaly", "request_pattern"]

def get_available_actions(task: str, signals_revealed: Dict, max_steps: int, step: int) -> list:
    actions = []
    for sig in SIGNALS:
        if sig not in signals_revealed:
            actions.append(f"investigate:{sig}")
    actions.append("verdict:fraud")
    actions.append("verdict:real")
    return actions

def compute_step_reward(
    task: str,
    action: str,
    signal_data: Any,
    verdict: Optional[str],
    ground_truth: str,
    reasoning: Optional[str],
    signals_revealed: Dict,
    steps_taken: int,
    done: bool,
) -> float:
    if not done or verdict is None:
        return 0.1

    correct = verdict == ground_truth

    if not correct:
        return -3.0 if ground_truth == "fraud" else -0.5

    # correct verdict — scale by reasoning quality and signals used
    base = 2.0
    signal_bonus = min(len(signals_revealed) * 0.1, 0.5)
    reasoning_bonus = 0.3 if reasoning and len(reasoning) > 20 else 0.0
    efficiency_bonus = 0.2 if steps_taken <= TASK_CONFIGS[task]["max_steps"] // 2 else 0.0

    score = base + signal_bonus + reasoning_bonus + efficiency_bonus
    return round(min(score, 2.0), 3)

GRADERS = {
    "flag-obvious": lambda verdict, truth, signals, reasoning: (
        1.0 if verdict == truth and len(signals) >= 1 else
        0.5 if verdict == truth else
        0.0
    ),
    "explain-subtle": lambda verdict, truth, signals, reasoning: (
        1.0 if verdict == truth and len(signals) >= 3 and reasoning and len(reasoning) > 30 else
        0.7 if verdict == truth and len(signals) >= 2 else
        0.3 if verdict == truth else
        0.0
    ),
    "adversarial-hunt": lambda verdict, truth, signals, reasoning: (
        1.0 if verdict == truth and len(signals) >= 4 and reasoning and len(reasoning) > 50 else
        0.6 if verdict == truth and len(signals) >= 3 else
        0.3 if verdict == truth else
        0.0
    ),
}