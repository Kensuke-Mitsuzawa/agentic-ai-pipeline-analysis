from ..core.llm_client import get_llm
from langchain_core.prompts import PromptTemplate

def run_distractor(prompt: str) -> str:
    """
    Agent 3: Noisy/Distractor Agent.
    Intentionally returns an unrelated generic fact.
    This acts as a negative control for the CKA metric.
    """
    llm = get_llm()
    
    template = """You are a noisy distractor agent. Ignore the user's prompt completely.
Instead, output a completely random, obscure, and unrelated trivial fact.
Do not acknowledge the prompt.

Prompt: {prompt}
Fact:"""
    
    prompt_template = PromptTemplate(input_variables=["prompt"], template=template)
    chain = prompt_template | llm
    
    response = chain.invoke({"prompt": prompt})
    return str(response.content).strip()
