from agentic_ai_analysis.core.llm_client import get_llm
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langgraph.prebuilt import create_react_agent
from typing import Dict, Any

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
    
    agent = create_react_agent(llm, tools)
    
    # Run the langgraph agent with the input keywords
    messages = agent.invoke({"messages": [("user", f"Find recent academic papers related to these keywords: {keywords}")]})
    
    # Extract the final answer and intermediate thoughts/observations from the message history
    formatted_steps = []
    final_answer = ""
    
    if "messages" in messages:
        msg_list = messages["messages"]
        if msg_list:
            final_answer = msg_list[-1].content
            
        # Parse through the graph state to find Tool calls (Thoughts) and Tool results (Observations)
        for msg in msg_list:
            if msg.type == "ai" and msg.tool_calls:
                # This is a thought/action
                call = msg.tool_calls[0]
                formatted_steps.append({
                    "type": "thought",
                    "content": f"Thought: I should use {call['name']} with args {call['args']}\nAction: {call['name']}\nAction Input: {call['args']}"
                })
            elif msg.type == "tool":
                # This is an observation
                formatted_steps.append({
                    "type": "observation",
                    "content": str(msg.content)
                })
                
    # Group them so they match the expected (thought, observation) schema for `main.py`
    paired_steps = []
    current_thought = ""
    for step in formatted_steps:
        if step["type"] == "thought":
            current_thought = step["content"]
        elif step["type"] == "observation":
            paired_steps.append({
                "thought_action": current_thought,
                "observation": step["content"]
            })
            current_thought = ""
            
    return {
        "final_answer": final_answer,
        "intermediate_steps": paired_steps
    }
