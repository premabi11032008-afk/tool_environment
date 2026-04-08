# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimal Tool Environment Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from models import OptimalToolEnvironmentAction, OptimalToolEnvironmentObservation
from openenv.core.env_server.types import State


class OptimalToolEnvironmentEnv(
    EnvClient[OptimalToolEnvironmentAction, OptimalToolEnvironmentObservation, State]
):
    """
    Client for the Optimal Tool Environment Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    """

    def _step_payload(self, action: OptimalToolEnvironmentAction) -> Dict:
        """
        Convert OptimalToolEnvironmentAction to JSON payload for step message.

        Args:
            action: OptimalToolEnvironmentAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "message": action.message,
        }

    def _parse_result(self, payload: Dict) -> StepResult[OptimalToolEnvironmentObservation]:
        """
        Parse server response into StepResult[OptimalToolEnvironmentObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with OptimalToolEnvironmentObservation
        """
        obs_data = payload.get("observation", {})
        #print(obs_data)
        observation = OptimalToolEnvironmentObservation(
            is_tool_action=obs_data.get("is_tool_action",False),
            message=obs_data.get("message",""),
            last_echoed=obs_data.get("last_echoed",""),
            metadata=obs_data.get("metadata",{}),
            tool_result=obs_data.get("tool_result",[])
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
