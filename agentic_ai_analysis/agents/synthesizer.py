from agentic_ai_analysis.core.llm_client import get_llm
from langchain_core.prompts import PromptTemplate

from typing import Optional, Any

def run_synthesizer(original_prompt: str, judge_explanations: list[str], generation_parameters: Optional[Any] = None) -> str:
    """
    Agent 5: Synthesizer Agent.
    Formulates the final expert answer using the original prompt and the filtered 
    explanations from the judge (which contain the filtered context).
    """
    llm = get_llm(generation_parameters=generation_parameters)
    
    # Join all valid contexts
    combined_context = "\n\n---\n\n".join(judge_explanations)
    
    if not combined_context.strip():
        combined_context = "No relevant academic context was found by the system."
        
    template = """You are an expert Synthesizer Agent. Your task is to provide a final, comprehensive answer to the user's prompt using the filtered academic context provided.

User Prompt: {prompt}

Filtered Academic Context:
{context}

Please provide a detailed, well-structured final answer based ONLY on the provided context. If the context is empty or unhelpful, state that you cannot answer based on the retrieved information.
"""

    prompt_template = PromptTemplate(input_variables=["prompt", "context"], template=template)
    chain = prompt_template | llm
    
    response = chain.invoke({
        "prompt": original_prompt,
        "context": combined_context
    })
    
    return str(response.content).strip()
