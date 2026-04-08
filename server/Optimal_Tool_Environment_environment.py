# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Optimal Tool Environment Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from Optimal_Tool_Environment.models import OptimalToolEnvironmentObservation,OptimalToolEnvironmentAction
from generate_questions import generate_episode,execute_according_to_the_query
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re

load_dotenv()

class OptimalToolEnvironmentEnvironment(Environment):
    """

    """

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the Optimal_Tool_Environment environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.questions=["hi"]
        self.index=0
        self._reset_count = 0

    def reset(self) -> OptimalToolEnvironmentObservation:
        """
        Reset the environment.

        Returns:
            OptimalToolEnvironmentObservation with a ready message
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.questions=generate_episode()
        self._reset_count += 1
        self.index=0
        self.last_echoed="Optimal Tool Environment environment ready!"

        return OptimalToolEnvironmentObservation(
            message="Optimal Tool Environment environment ready!",
            last_echoed=f"question : {self.questions[self.index]['question']}",
            is_tool_action=False,
            tool_result=[],
            done=False,
            reward=0.0,
        )

    def step(self, action: OptimalToolEnvironmentAction) -> OptimalToolEnvironmentObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: OptimalToolEnvironmentAction containing the message  from ai agent to calculate reward and tool result

        Returns:
            OptimalToolEnvironmentObservation with reward,tool_result, curent_question,last_question
        """
        self._state.step_count += 1

        message = action.message
        tool_action="<tools>" in message
        last_message=self.last_echoed
        self.last_echoed=message

        reward = self.calculate_reward(message,tool_action)
        tool_result=[]

        if tool_action:
            match = re.search(r'<tools>(.*?)</tools>', message, re.DOTALL)
            tools_list=[]

            if match:
                try:
                    tools_list = json.loads(match.group(1))
                except:
                    pass
            
            for tool in tools_list:
                tool_result.append(execute_according_to_the_query(tool["tool"],tool["params"]))
        
        if not tool_action:
            self.index+=1
            
        current_q = self.questions[self.index]["question"] if self.index < len(self.questions) else "Done"
        done = self.index >= len(self.questions)

        return OptimalToolEnvironmentObservation(
            message=f"question = {current_q}",
            last_echoed=last_message,
            is_tool_action=tool_action,
            tool_result=tool_result,
            reward=reward,
            metadata={"original_message": message, "step": self._state.step_count},
            done=done
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

    def is_similar(self,ans1, ans2):
        emb = TfidfVectorizer().fit_transform([ans1, ans2])
        sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
        return sim

    def calculate_reward(self,message :str,tool_action :bool):
        # assuming message tool call from the llm with the tag <tools></tools>
        reward=0

        if tool_action:
            ideal_tools = self.questions[self.index]["tools"]
            match = re.search(r'<tools>(.*?)</tools>', message, re.DOTALL)
            tools_list = []

            if match:
                try:
                    tools_list = json.loads(match.group(1))
                except:
                    return -0.1

            matched = set()
            valid_tools = [t["tool"] for t in ideal_tools]

            for predicted in tools_list:
                if predicted["tool"] not in valid_tools:
                    reward -= 0.1
                    continue

                for i, expected in enumerate(ideal_tools):
                    if i in matched:
                        continue

                    if predicted["tool"] != expected["tool"]:
                        continue

                    reward += 0.3  # correct tool

                    for key in expected["params"]:
                        if key in predicted["params"]:
                            sim_score = self.is_similar(
                                str(expected["params"][key]),
                                str(predicted["params"][key])
                            )
                            reward += 0.5 * sim_score
                        else:
                            reward -= 0.05

                    matched.add(i)
                    break

            reward = min(reward, 3.0)
            reward = max(reward, -0.5)

            
        else:
            ideal_answer=self.questions[self.index]["answer"]
            sim=self.is_similar(ideal_answer["text"],message)

            semantic_similarity = max(0, sim)  # range 0 to 1
            keywords = ideal_answer.get("keywords", []) 

            keyword_similarity = sum(
                1 for key in keywords if key.lower() in message.lower()
            )

            reward = (
                0.6 * semantic_similarity +
                0.4 * (keyword_similarity / max(len(keywords), 1))
            )

            reward = max(reward, -0.5)
                    
        return reward

