"""
tests/test_env.py - Core environment tests
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server.data_gen import generate_session, SIGNAL_EXTRACTORS
from server.tasks import TASK_CONFIGS, compute_step_reward, get_available_actions
from server.models import FraudDetectAction, FraudDetectObservation
from server.fraud_detect_env_environment import FraudDetectEnvironment


# ── Data generation tests ─────────────────────────────────────────────────────
class TestDataGen:
    def test_easy_session_generates_logs(self):
        logs, label = generate_session("easy")
        assert len(logs) >= 6
        assert label in ("fraud", "real")

    def test_medium_session_generates_logs(self):
        logs, label = generate_session("medium")
        assert len(logs) >= 6
        assert label in ("fraud", "real")

    def test_hard_session_generates_logs(self):
        logs, label = generate_session("hard")
        assert len(logs) >= 6
        assert label in ("fraud", "real")

    def test_log_schema_has_required_fields(self):
        logs, _ = generate_session("easy")
        for log in logs:
            assert "session_id" in log
            assert "timestamp" in log
            assert "ip" in log
            assert "device" in log
            assert "country" in log
            assert "event" in log
            assert "endpoint" in log

    def test_signal_extractors_all_work(self):
        logs, _ = generate_session("easy")
        for name, extractor in SIGNAL_EXTRACTORS.items():
            result = extractor(logs)
            assert isinstance(result, dict)
            assert "suspicious" in result


# ── Task config tests ─────────────────────────────────────────────────────────
class TestTaskConfigs:
    def test_all_tasks_defined(self):
        assert "flag-obvious" in TASK_CONFIGS
        assert "explain-subtle" in TASK_CONFIGS
        assert "adversarial-hunt" in TASK_CONFIGS

    def test_max_steps_increase_with_difficulty(self):
        assert TASK_CONFIGS["flag-obvious"]["max_steps"] < TASK_CONFIGS["explain-subtle"]["max_steps"]
        assert TASK_CONFIGS["explain-subtle"]["max_steps"] < TASK_CONFIGS["adversarial-hunt"]["max_steps"]

    def test_easy_has_hint(self):
        assert TASK_CONFIGS["flag-obvious"]["hint"] is not None

    def test_available_actions_returns_list(self):
        actions = get_available_actions("flag-obvious", {}, 3, 0)
        assert "verdict:fraud" in actions
        assert "verdict:real" in actions
        assert any(a.startswith("investigate:") for a in actions)

    def test_investigated_signals_removed_from_actions(self):
        actions = get_available_actions("flag-obvious", {"ip_velocity": {}}, 3, 1)
        assert "investigate:ip_velocity" not in actions


# ── Reward tests ──────────────────────────────────────────────────────────────
class TestRewards:
    def test_correct_fraud_verdict_positive(self):
        r = compute_step_reward(
            task="flag-obvious", action="verdict:fraud",
            signal_data=None, verdict="fraud", ground_truth="fraud",
            reasoning="Multiple IPs detected", signals_revealed={"ip_velocity": {}},
            steps_taken=2, done=True,
        )
        assert r > 0

    def test_missed_fraud_heavy_penalty(self):
        r = compute_step_reward(
            task="flag-obvious", action="verdict:real",
            signal_data=None, verdict="real", ground_truth="fraud",
            reasoning=None, signals_revealed={},
            steps_taken=1, done=True,
        )
        assert r == -3.0

    def test_false_alarm_light_penalty(self):
        r = compute_step_reward(
            task="flag-obvious", action="verdict:fraud",
            signal_data=None, verdict="fraud", ground_truth="real",
            reasoning=None, signals_revealed={},
            steps_taken=1, done=True,
        )
        assert r == -0.5

    def test_asymmetric_penalty_ratio(self):
        missed = compute_step_reward(
            task="flag-obvious", action="verdict:real",
            signal_data=None, verdict="real", ground_truth="fraud",
            reasoning=None, signals_revealed={}, steps_taken=1, done=True,
        )
        false_alarm = compute_step_reward(
            task="flag-obvious", action="verdict:fraud",
            signal_data=None, verdict="fraud", ground_truth="real",
            reasoning=None, signals_revealed={}, steps_taken=1, done=True,
        )
        assert abs(missed) > abs(false_alarm)


# ── Environment tests ─────────────────────────────────────────────────────────
class TestEnvironment:
    def setup_method(self):
        os.environ["FRAUD_ENV_TASK"] = "flag-obvious"
        self.env = FraudDetectEnvironment()

    def test_reset_returns_observation(self):
        obs = self.env.reset()
        assert isinstance(obs, FraudDetectObservation)
        assert obs.session_id != ""
        assert len(obs.logs) > 0
        assert obs.done is False
        assert obs.task == "flag-obvious"

    def test_investigate_action_reveals_signal(self):
        self.env.reset()
        action = FraudDetectAction(action="investigate:ip_velocity", reasoning="test")
        obs = self.env.step(action)
        assert "ip_velocity" in obs.signals_revealed
        assert obs.reward == 0.1

    def test_duplicate_investigate_penalised(self):
        self.env.reset()
        action = FraudDetectAction(action="investigate:ip_velocity", reasoning="test")
        self.env.step(action)
        obs = self.env.step(action)
        assert obs.reward == -0.05

    def test_invalid_action_penalised(self):
        self.env.reset()
        action = FraudDetectAction(action="nonsense:action", reasoning="test")
        obs = self.env.step(action)
        assert obs.reward == -0.1

    def test_verdict_ends_episode(self):
        self.env.reset()
        action = FraudDetectAction(action="verdict:fraud", reasoning="fraud detected")
        obs = self.env.step(action)
        assert obs.done is True

    def test_ground_truth_revealed_on_done(self):
        self.env.reset()
        action = FraudDetectAction(action="verdict:fraud", reasoning="fraud detected")
        obs = self.env.step(action)
        assert obs.ground_truth in ("fraud", "real")

    def test_step_after_done_safe(self):
        self.env.reset()
        action = FraudDetectAction(action="verdict:fraud", reasoning="fraud detected")
        self.env.step(action)
        obs = self.env.step(action)
        assert obs.done is True
        assert obs.reward == 0.0
