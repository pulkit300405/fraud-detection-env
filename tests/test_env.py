import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from server.data_gen import generate_session, SIGNAL_EXTRACTORS
from server.tasks import TASK_CONFIGS, compute_step_reward, get_available_actions
from server.fraud_detect_env_environment import FraudDetectEnvironment
from server.models import FraudDetectAction, FraudDetectObservation

def test_easy_session():
    logs, label = generate_session("easy")
    assert len(logs) >= 6
    assert label in ("fraud", "real")

def test_medium_session():
    logs, label = generate_session("medium")
    assert len(logs) >= 6
    assert label in ("fraud", "real")

def test_hard_session():
    logs, label = generate_session("hard")
    assert len(logs) >= 6
    assert label in ("fraud", "real")

def test_log_schema():
    logs, _ = generate_session("easy")
    for log in logs:
        assert "session_id" in log
        assert "timestamp" in log
        assert "ip" in log
        assert "country" in log
        assert "event" in log

def test_all_tasks_defined():
    assert "flag-obvious" in TASK_CONFIGS
    assert "explain-subtle" in TASK_CONFIGS
    assert "adversarial-hunt" in TASK_CONFIGS

def test_missed_fraud_heavy_penalty():
    r = compute_step_reward(
        task="flag-obvious", action="verdict:real",
        signal_data=None, verdict="real", ground_truth="fraud",
        reasoning=None, signals_revealed={}, steps_taken=1, done=True,
    )
    assert r == -3.0

def test_false_alarm_light_penalty():
    r = compute_step_reward(
        task="flag-obvious", action="verdict:fraud",
        signal_data=None, verdict="fraud", ground_truth="real",
        reasoning=None, signals_revealed={}, steps_taken=1, done=True,
    )
    assert r == -0.5

def test_asymmetric_penalty():
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

def test_reset_returns_observation():
    os.environ["FRAUD_ENV_TASK"] = "flag-obvious"
    env = FraudDetectEnvironment()
    obs = env.reset()
    assert isinstance(obs, FraudDetectObservation)
    assert obs.done is False
    assert len(obs.logs) > 0

def test_investigate_reveals_signal():
    os.environ["FRAUD_ENV_TASK"] = "flag-obvious"
    env = FraudDetectEnvironment()
    env.reset()
    action = FraudDetectAction(action="investigate:ip_velocity", reasoning="test")
    obs = env.step(action)
    assert "ip_velocity" in obs.signals_revealed
    assert obs.reward == 0.1

def test_verdict_ends_episode():
    os.environ["FRAUD_ENV_TASK"] = "flag-obvious"
    env = FraudDetectEnvironment()
    env.reset()
    action = FraudDetectAction(action="verdict:fraud", reasoning="fraud detected")
    obs = env.step(action)
    assert obs.done is True
