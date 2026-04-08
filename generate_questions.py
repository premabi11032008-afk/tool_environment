import random as rd
from Tools import websearch,get_weather
import os
from groq import Groq
import json
from dotenv import load_dotenv
import re

load_dotenv()

def generate_question(context, q_type,llm):

    prompt = f"""
        Generate a realistic user question for {q_type}.

        Context:
        {context}

        Rules:
        - Make it natural (like real user)
        - 55 percent require tool usage
        - 45 percent depend on previous answer
        - Keep it short
        -avoid any king to amigious question at all cost

        Return only the question.
        """
    response = llm.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0,
    max_tokens=100
)
    return response.choices[0].message.content

def decide_tool(question, context,llm):
    prompt = f"""
You are an intelligent assistant that decides tool usage.

Available tools:
1. weather → use for weather queries(parameter:city:=string fo the city name)
2. websearch → use for general knowledge or summaries(parameter:query:= valid search topic in string)

Context:
{context}

User Question:
{question}

Rules:
- If answer can be derived from context → DO NOT use tools
- If external data is needed → choose correct tool
- Extract correct parameters

Return like:
<tools>[
{{"tool":"tool_name_1","params":{"#parameter as dictionary"}}}
{{"tool:"tools_name_2,"params":{"#parameter as dictionary"}}} #if any
]</tools>
"""

    response = llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150
    )

    match = re.search(r'<tools>(.*?)</tools>', response.choices[0].message.content, re.DOTALL)
    tools_list=[]

    if match:
        try:
            tools_list = json.loads(match.group(1))
        except:
            pass

    return tools_list

def execute_according_to_the_query(tool,params):

    if tool=="weather":
        return get_weather(params["city"])
    else:
        return websearch(params["query"])
    


def summarize(question,text,llm,retries=2):
    prompt="""You are an expert evaluator and communicator.

Your task is to analyze a given question and its corresponding answer, then generate an improved, ideal response that fully captures the intent of the question while preserving the correct information from the provided answer.

Do NOT repeat or restate the question or the original answer.
Instead, produce a refined answer that:
- Clearly conveys what the response is trying to say
- Improves clarity, completeness, and coherence
- Retains factual correctness and context relevance

IMPORTANT RULES:

1. If the required information is already available from previous tool outputs,
   DO NOT call the tool again.

2. Reuse previous results to answer follow-up questions.

3. Only call tools if the information is missing.

Also evaluate how truthful and context-aware the original answer is.

Return ONLY a valid JSON object in the following format:
{{
  "text": "<refined ideal answer>",
  "score": <number between 0 and 1 representing truthfulness and context awareness>
  "keywords:<important keywords that can be include limit it to 5>
}}

If the output is not valid JSON, it will be rejected.

Question: {}
Answer: {}""".format(question,text)
    
    for _ in range(retries):
        response = llm.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=150
        )

        content = response.choices[0].message.content

        try:
            return json.loads(content)
        except:
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    continue

    return {"text": "", "score": 0}

    
def generate_episode():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    QUESTION_TYPES = ["weather", "websummarization","random_non_tool_question"]

    episode=[]
    context=[]

    for _ in range(3):

        q_type = rd.choice(QUESTION_TYPES)

        question=generate_question(context=context,q_type=q_type,llm=client)
        tools_usage=decide_tool(question=question,context=context,llm=client)
        all_answer=[]
        for tool in tools_usage:
            #print(tool)
            all_answer.append(execute_according_to_the_query(tool["tool"],tool["params"]))

        ideal_answer=summarize(question=question,text=all_answer,llm=client)

        #pprint(question)
        #pprint(tools_usage)
        #pprint(ideal_answer)

        episode.append({
            "question":question,
            "tools":tools_usage,
            "answer":ideal_answer
        })

        context.append({
            "question":question,
            "answer":ideal_answer["text"]
        })
    
    return episode

#pprint(generate_episode())


