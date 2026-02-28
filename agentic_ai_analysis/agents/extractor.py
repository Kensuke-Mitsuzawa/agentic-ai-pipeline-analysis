from langchain_core.prompts import PromptTemplate
from agentic_ai_analysis.core.llm_client import get_llm

def extract_keywords(prompt: str) -> str:
    """
    Agent 1: Keyword Extractor.
    Takes the user query and outputs a comma-separated list of keywords.
    """
    llm = get_llm()
    
    template = """You are a helpful academic keyword extractor. 
Given the user prompt, extract the most important keywords and return ONLY a comma-separated list.
Do not provide any conversational text or explanation.

Prompt: {prompt}
Keywords:"""
    
    prompt_template = PromptTemplate(input_variables=["prompt"], template=template)
    chain = prompt_template | llm
    
    response = chain.invoke({"prompt": prompt})
    return str(response.content).strip()
