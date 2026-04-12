---
title: Fraud Detect Env
emoji: 🔍
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - fraud-detection
  - security
  - multi-turn
  - real-world
short_description: Multi-turn RL environment for AI-powered fraud detection
---

# 🔍 Fraud Detection RL Environment

A multi-turn reinforcement learning environment where AI agents investigate synthetic app behavior logs to detect fraudulent sessions — built for the **Meta × HuggingFace × Scaler OpenEnv Hackathon 2026**.

## Why This Problem?

Fraud detection costs companies billions annually. Unlike simple classifiers, real fraud analysts gather evidence across multiple signals before issuing a verdict. This environment trains agents to:

- **Investigate** — gather signals across IP velocity, device fingerprints, geo anomalies, login patterns
- **Reason under uncertainty** — decide when enough evidence justifies a verdict
- **Handle asymmetric costs** — missing real fraud (−3.0) costs 6× more than a false alarm (−0.5)

## Quick Start

```python
from client import FraudDetectEnv
from models import FraudDetectAction

with FraudDetectEnv(base_url="https://Pulkit3004-fraud-detect-env.hf.space") as env:
    result = env.reset()
    obs = result.observation
    print(f"Session: {obs.session_id}, Task: {obs.task}")

    action = FraudDetectAction(
        action="investigate:ip_velocity",
        reasoning="Check for IP anomalies first"
    )
    result = env.step(action)
    print(f"Reward: {result.reward}, Done: {result.done}")
```

## Try It Now (No Setup Required)

```bash
# Health check
curl -X GET https://Pulkit3004-fraud-detect-env.hf.space/health

# Start episode
curl -X POST https://Pulkit3004-fraud-detect-env.hf.space/reset \
     -H "Content-Type: application/json" -d '{}'

# Investigate a signal
curl -X POST https://Pulkit3004-fraud-detect-env.hf.space/step \
     -H "Content-Type: application/json" \
     -d '{"action": {"action": "investigate:ip_velocity", "reasoning": "Check IP patterns"}}'

# Give a verdict
curl -X POST https://Pulkit3004-fraud-detect-env.hf.space/step \
     -H "Content-Type: application/json" \
     -d '{"action": {"action": "verdict:fraud", "reasoning": "Multiple IPs from different countries"}}'
```

## Tasks

| Task | Difficulty | Max Steps | Description |
|------|-----------|-----------|-------------|
| `flag-obvious` | 🟢 Easy | 3 | Obvious fraud with multiple IPs and geo anomalies |
| `explain-subtle` | 🟡 Medium | 5 | Subtle patterns requiring 3+ signal investigations |
| `adversarial-hunt` | 🔴 Hard | 8 | Adversarially crafted logs mimicking legitimate behavior |

## Action Space

| Action | Description |
|--------|-------------|
| `investigate:ip_velocity` | Check unique IPs per session |
| `investigate:device_fingerprint` | Check device variety |
| `investigate:login_frequency` | Check login attempt count |
| `investigate:geo_anomaly` | Check country spread |
| `investigate:request_pattern` | Check suspicious endpoints |
| `verdict:fraud` | Final verdict — session is fraudulent |
| `verdict:real` | Final verdict — session is legitimate |

## Reward Structure

| Event | Reward |
|-------|--------|
| Investigate new signal | +0.1 |
| Correct fraud verdict | up to +2.0 |
| Missed fraud | **−3.0** |
| False alarm | −0.5 |
| Duplicate investigation | −0.05 |
| Invalid action | −0.1 |

Asymmetric penalty: missing real fraud costs **6× more** than a false alarm.

## Baseline Scores

Evaluated using `Qwen/Qwen2.5-72B-Instruct` at `TEMPERATURE=0.0`:

| Task | Mean Score |
|------|-----------|
| `flag-obvious` | 0.65 |
| `explain-subtle` | 0.52 |
| `adversarial-hunt` | 0.41 |
| **Overall** | **0.53** |

## Run Inference

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export ENV_URL=https://Pulkit3004-fraud-detect-env.hf.space

python inference.py

# Run specific task
TASK_NAME=flag-obvious python inference.py
```

## Local Setup

```bash
git clone https://github.com/Pulkit3004/fraud-detect-env
cd fraud-detect-env
uv sync
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Docker

```bash
docker build -t fraud_detect_env .
docker run -p 7860:7860 fraud_detect_env
```

## Run Tests

```bash
pytest tests/ -v
```

## Architecture

- `server/fraud_detect_env_environment.py` — core RL environment logic
- `server/tasks.py` — task configs, graders, reward computation
- `server/data_gen.py` — synthetic fraud session generator
- `server/models.py` — typed action and observation models
- `server/app.py` — FastAPI server
- `inference.py` — baseline agent using LLM via proxy
- `client.py` — OpenEnv client
- `tests/` — full test suite

## Built For

Meta × HuggingFace × Scaler OpenEnv Hackathon 2026
