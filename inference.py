"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import asyncio
import os
import textwrap
from typing import List, Optional
from dotenv import load_dotenv
from openai import OpenAI
from client import OptimalToolEnvironmentAction,OptimalToolEnvironmentEnv

load_dotenv()

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL ="https://router.huggingface.co/v1"
MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
MAX_STEPS = 8
TEMPERATURE = 0.0
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]


# Max possible reward: each token contributes 0.1, across all steps
_MAX_REWARD_PER_STEP = 1.0
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = textwrap.dedent("""
You are an intelligent agent interacting with a tool-using environment.

You will receive ONE question at a time.

CRITICAL RULES (STRICT):

1. ALWAYS focus ONLY on the CURRENT question.
   - Do NOT answer previous questions
   - Do NOT reuse previous answers unless explicitly needed

2. TOOL USAGE:
   - weather → ALWAYS use for ANY weather-related question
   - sql → ALWAYS use for ANY database-related question
   - websearch → use for general knowledge if needed

3. NEVER answer a tool-based question without calling the tool first.

4. After calling a tool:
   - WAIT for tool results
   - THEN answer using the tool result
   - DO NOT call the tool again unless necessary

5. RESPONSE FORMAT:
   - Tool call:
     <tools>
     [
       {"tool": "tool_name", "params": {"key": "value"}}
     ]
     </tools>

   - Final answer:
     Plain natural language answer ONLY

6. DO NOT:
   - Mix answers from different questions
   - Skip tool usage when required
   - Answer before tool result is available

7. Each step follows STRICTLY:
   Question → Tool (if needed) → Answer → Next Question

Your goal is to maximize reward by being accurate and correctly using tools.
""").strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, last_echoed: str, is_tool_action:bool, tool_result:list, last_reward: float, history: List[str],current_question:str) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}

Last Action:
{last_echoed}

Current_Question:
{current_question}

Was Tool Call: {is_tool_action}

Tool Results:
{tool_result}

Reward Received: {last_reward:.2f}

Previous Steps:
{history_block}

Instructions:
- Decide whether to answer directly or use tools
- Use <tools>
  [
    {{"tool": "tool_name", "params": {{"key": "value"}}}}
  ]
  </tools> format if calling tools

Available tools:
1. weather → use for weather queries(parameter:city:=string fo the city name)
2. websearch → use for general knowledge or summaries(parameter:query:= valid search topic in string)

- Match tool names and parameters carefully
- Provide semantically accurate answers

Send your next response:
        """
    ).strip()


def get_model_message(client: OpenAI, step: int, last_echoed: str, last_reward: float, history: List[str],tool_result:list,is_tool_action:bool,current_question:str) -> str:

    user_prompt = build_user_prompt(step=step, last_echoed=last_echoed,current_question=current_question, last_reward=last_reward, history=history,
                                    is_tool_action=is_tool_action,tool_result=tool_result)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        #print(text)
        return text if text else "hello"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "hello"


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    try:
        env = await OptimalToolEnvironmentEnv.from_docker_image(IMAGE_NAME)
    except Exception as e:
        print(str(e))

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset() # OpenENV.reset()
        last_echoed = result.observation.last_echoed
        current_message=result.observation.message
        last_reward = 0.0
        is_tool=False
        tool_result=[]

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            message = get_model_message(client=client, step=step,  last_reward=last_reward, history=history,
                                        last_echoed=last_echoed,tool_result=tool_result,is_tool_action=is_tool,
                                        current_question=current_message)

            result = await env.step(OptimalToolEnvironmentAction(message=message))

            obs = result.observation
            tool_result=obs.tool_result
            is_tool=obs.is_tool_action
            current_message=obs.message
            last_echoed=obs.last_echoed

            #print(obs)

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {message!r} is tool used {is_tool} tools_result {tool_result} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as err:
        print("[ERROR] : ",err)
        print(str(err))

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())