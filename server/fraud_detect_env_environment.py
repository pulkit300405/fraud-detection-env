"""
fraud_detect_env_environment.py - core env class
"""
import os
import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import FraudDetectAction, FraudDetectObservation
except ImportError:
    from models import FraudDetectAction, FraudDetectObservation

try:
    from .data_gen import generate_session, SIGNAL_EXTRACTORS
    from .tasks import TASK_CONFIGS, GRADERS, get_available_actions, compute_step_reward
except ImportError:
    from data_gen import generate_session, SIGNAL_EXTRACTORS
    from tasks import TASK_CONFIGS, GRADERS, get_available_actions, compute_step_reward

VALID_TASKS = list(TASK_CONFIGS.keys())

class FraudDetectEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._task: str = "flag-obvious"
        self._logs: list = []
        self._ground_truth: str = ""
        self._signals_revealed: Dict[str, Any] = {}
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._last_error: Optional[str] = None

    def reset(self, seed=None, episode_id=None, **kwargs) -> FraudDetectObservation:
        task = os.getenv("FRAUD_ENV_TASK", "flag-obvious")
        if task not in VALID_TASKS:
            task = "flag-obvious"

        cfg = TASK_CONFIGS[task]
        logs, ground_truth = generate_session(cfg["difficulty"])

        self._state = State(
            episode_id=episode_id if episode_id else str(uuid.uuid4()),
            step_count=0
        )
        self._task = task
        self._logs = logs
        self._ground_truth = ground_truth
        self._signals_revealed = {}
        self._done = False
        self._cumulative_reward = 0.0
        self._last_error = None

        available = get_available_actions(task, {}, cfg["max_steps"], 0)

        return FraudDetectObservation(
            session_id=logs[0]["session_id"] if logs else "unknown",
            logs=logs,
            step_num=0,
            signals_revealed={},
            available_actions=available,
            task=task,
            difficulty=cfg["difficulty"],
            context_hint=cfg["hint"],
            done=False,
            reward=0.0,
        )

    def step(self, action: FraudDetectAction) -> FraudDetectObservation:
        if self._done:
            return FraudDetectObservation(
                session_id=self._logs[0]["session_id"] if self._logs else "done",
                logs=[],
                step_num=self._state.step_count,
                signals_revealed=self._signals_revealed,
                available_actions=[],
                task=self._task,
                difficulty=TASK_CONFIGS[self._task]["difficulty"],
                context_hint=None,
                ground_truth=self._ground_truth,
                done=True,
                reward=0.0,
                metadata={"error": "Episode already ended"},
            )

        self._state.step_count += 1
        step = self._state.step_count
        cfg = TASK_CONFIGS[self._task]
        act_str = action.action.strip().lower()
        reasoning = action.reasoning
        self._last_error = None

        signal_data = None
        verdict = None
        done = False
        reward = 0.0

        if act_str.startswith("investigate:"):
            sig = act_str.split(":", 1)[1]
            if sig not in SIGNAL_EXTRACTORS:
                self._last_error = f"Unknown signal: {sig}"
                reward = -0.1
            elif sig in self._signals_revealed:
                self._last_error = f"Already investigated: {sig}"
                reward = -0.05
                signal_data = self._signals_revealed[sig]
            else:
                signal_data = SIGNAL_EXTRACTORS[sig](self._logs)
                self._signals_revealed[sig] = signal_data
                reward = 0.1

        elif act_str.startswith("verdict:"):
            v = act_str.split(":", 1)[1]
            if v not in ("fraud", "real"):
                self._last_error = f"Invalid verdict: {v}"
                reward = -0.1
            else:
                verdict = v
                done = True
                reward = compute_step_reward(
                    task=self._task,
                    action=act_str,
                    signal_data=None,
                    verdict=verdict,
                    ground_truth=self._ground_truth,
                    reasoning=reasoning,
                    signals_revealed=self._signals_revealed,
                    steps_taken=step,
                    done=True,
                )
        else:
            self._last_error = f"Invalid action: {act_str}"
            reward = -0.1

        if step >= cfg["max_steps"] and not done:
            done = True
            if self._ground_truth == "fraud":
                reward += -3.0
            else:
                reward += -0.5

        self._done = done
        self._cumulative_reward += reward

        available = [] if done else get_available_actions(
            self._task, self._signals_revealed, cfg["max_steps"], step
        )

        return FraudDetectObservation(
            session_id=self._logs[0]["session_id"] if self._logs else "unknown",
            logs=self._logs if not done else [],
            step_num=step,
            signals_revealed=self._signals_revealed,
            available_actions=available,
            task=self._task,
            difficulty=cfg["difficulty"],
            context_hint=None,
            ground_truth=self._ground_truth if done else None,
            done=done,
            reward=reward,
            metadata={
                "cumulative_reward": round(self._cumulative_reward, 3),
                "last_error": self._last_error,
                "verdict": verdict,
                "signal_data": signal_data,
            },
        )

    @property
    def state(self) -> State:
        return self._state