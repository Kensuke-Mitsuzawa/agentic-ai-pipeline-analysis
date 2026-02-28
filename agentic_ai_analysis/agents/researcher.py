from agentic_ai_analysis.core.llm_client import get_llm
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from typing import List, Dict, Any

def run_researcher(keywords: str) -> Dict[str, Any]:
    """
    Agent 2: Researcher (ReAct Pattern).
    Uses ArxivQueryRun to retrieve academic context based on the extracted keywords.
    Returns the final answer and the intermediate ReAct steps (Thoughts and Observations)
    so they can be embedded for CKA.
    """
    llm = get_llm()
    tool = ArxivQueryRun()
    tools = [tool]
    
    # Standard ReAct prompt instructions
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: Find recent academic papers related to these keywords: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate(
        input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
        template=template
    )
    
    agent = create_react_agent(llm, tools, prompt)
    
    # We must set return_intermediate_steps=True to extract Thoughts/Observations for CKA
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=False, 
        return_intermediate_steps=True,
        max_iterations=3
    )
    
    response = agent_executor.invoke({"input": keywords})
    
    # Format intermediate steps for embedding
    # intermediate_steps is a list of tuples: (AgentAction, observation)
    formatted_steps = []
    if "intermediate_steps" in response:
        for action, observation in response["intermediate_steps"]:
            formatted_steps.append({
                "thought_action": action.log,
                "observation": str(observation)
            })
            
    return {
        "final_answer": response.get("output", ""),
        "intermediate_steps": formatted_steps
    }
