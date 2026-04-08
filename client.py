# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fraud Detect Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import FraudDetectAction, FraudDetectObservation


class FraudDetectEnv(
    EnvClient[FraudDetectAction, FraudDetectObservation, State]
):
    """
    Client for the Fraud Detect Env Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step fraud investigation episodes.

    Example:
        >>> with FraudDetectEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     result = client.step(FraudDetectAction(
        ...         action="investigate:ip_velocity",
        ...         reasoning="Check for IP anomalies first"
        ...     ))
    """

    def _step_payload(self, action: FraudDetectAction) -> Dict:
        """Convert FraudDetectAction to JSON payload for step message."""
        return {
            "action": action.action,
            "reasoning": action.reasoning or "",
        }

    def _parse_result(self, payload: Dict) -> StepResult[FraudDetectObservation]:
        """Parse server response into StepResult[FraudDetectObservation]."""
        obs_data = payload.get("observation", {})

        observation = FraudDetectObservation(
            session_id=obs_data.get("session_id", ""),
            logs=obs_data.get("logs", []),
            step_num=obs_data.get("step_num", 0),
            signals_revealed=obs_data.get("signals_revealed", {}),
            available_actions=obs_data.get("available_actions", []),
            task=obs_data.get("task", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            context_hint=obs_data.get("context_hint"),
            ground_truth=obs_data.get("ground_truth"),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )