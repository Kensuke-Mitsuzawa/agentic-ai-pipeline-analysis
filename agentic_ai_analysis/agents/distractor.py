import random
from ..core.llm_client import get_llm
from langchain_core.prompts import PromptTemplate

def run_distractor(prompt: str) -> str:
    """
    Agent 3: Noisy/Distractor Agent.
    Intentionally returns an unrelated generic fact.
    This acts as a negative control for the CKA metric.
    """
    llm = get_llm()
    
    patterns = [
        "Output a Python code snippet.",
        "Output a recipe.",
        "Output random keyboard mashing.",
        "Output a sentence in French.",
        "Output a random Wikipedia article headline (just the headline text)."
    ]
    pattern = random.choice(patterns)
    
    template = """You are a noisy distractor agent. Ignore the user's prompt completely.
Instead, {pattern}
Do not acknowledge the prompt.

Prompt: {prompt}
Response:"""
    
    prompt_template = PromptTemplate(input_variables=["prompt", "pattern"], template=template)
    chain = prompt_template | llm
    
    response = chain.invoke({"prompt": prompt, "pattern": pattern})
    return str(response.content).strip()

