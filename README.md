---
title: Fraud Detect Env
emoji: 🔍
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
tags:
  - openenv
---

# Fraud Detection RL Environment

A multi-turn reinforcement learning environment where agents investigate synthetic app behavior logs to detect fraudulent sessions. Agents must gather evidence across multiple signals before issuing a verdict.

## Motivation
Fraud detection is a critical real-world task. This environment trains agents to reason under uncertainty, balance investigation cost against verdict confidence, and handle adversarial patterns.

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

## Observation Space
| Field | Type | Description |
|-------|------|-------------|
| `session_id` | str | Session being investigated |
| `logs` | list | App behavior log entries |
| `step_num` | int | Current episode step |
| `signals_revealed` | dict | Signals investigated so far |
| `available_actions` | list | Valid actions this step |
| `task` | str | Current task name |
| `difficulty` | str | easy / medium / hard |
| `context_hint` | str | Hint shown in easy tier only |
| `reward` | float | Step reward |
| `done` | bool | Episode complete |

## Tasks
| Task | Difficulty | Max Steps | Description |
|------|-----------|-----------|-------------|
| `flag-obvious` | Easy | 3 | Obvious fraud with multiple IPs and countries |
| `explain-subtle` | Medium | 5 | Subtle patterns requiring deeper investigation |
| `adversarial-hunt` | Hard | 8 | Adversarial fraud mimicking normal behavior |

## Reward Function
| Event | Reward |
|-------|--------|
| Investigate new signal | +0.1 |
| Correct verdict | up to +2.0 |
| Missed fraud | -3.0 |
| False alarm | -0.5 |
| Duplicate investigation | -0.05 |
| Invalid action | -0.1 |

## Baseline Scores
| Task | Score |
|------|-------|
| flag-obvious | 0.65 |
| explain-subtle | 0.52 |
| adversarial-hunt | 0.41 |

## Setup
```bash
pip install openenv-core
git clone https://huggingface.co/spaces/Pulkit3004/fraud-detect-env
cd fraud-detect-env
uv sync
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## API Usage
```bash
# Start episode
curl -X POST https://Pulkit3004-fraud-detect-env.hf.space/reset -H "Content-Type: application/json" -d '{}'

# Take a step
curl -X POST https://Pulkit3004-fraud-detect-env.hf.space/step -H "Content-Type: application/json" -d '{"action": {"action": "investigate:ip_velocity", "reasoning": "Check IP patterns"}}'
```

## Running Inference
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=your_token
export ENV_URL=https://Pulkit3004-fraud-detect-env.hf.space
python inference.py
```