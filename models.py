# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from openenv.core.env_server.types import Action, Observation
from pydantic import Field



class OptimalToolEnvironmentAction(Action):
    """Action for the Optimal Tool Environment environment - Replied answer from the ai agent."""

    message: str = Field(..., description="The Answer given by the ai agent")


class OptimalToolEnvironmentObservation(Observation):
    """Observation from the Optimal Tool Environment environment - the what the ideal answer for the generated question."""
    message : str = Field(...,description="simple confirmation messages")
    is_tool_action : bool = Field(...,description="used to decide if the message is a tool action is included")
    tool_result: list | None= Field(...,description="is a tool action is used then the tool result is added ")
    last_echoed: str = Field(...,description="simple last echoed mesages")
