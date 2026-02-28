import time
from typing import Dict, Any, List

from langchain_core.prompts import PromptTemplate
from agentic_ai_analysis.core.llm_client import get_llm
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from .data_models import ResearcherNodeOutcome, BaseNodeOutcome

def run_researcher(keywords_str: str, budget_max: int = 2) -> ResearcherNodeOutcome:
    """
    Agent 2: Iterative Researcher.
    Loops exactly up to `budget_max` times:
    1. Prompts LLM with current keyword stack to form an Arxiv query.
    2. Runs Arxiv tool to retrieve docs.
    3. Prompts LLM to extract new 'keyword: description' pairs from documents.
    4. Adds new keywords to the stack.
    """
    start_time = time.perf_counter()
    llm = get_llm()
    arxiv_tool = ArxivQueryRun()
    
    # Initialize the keyword stack from the string
    stack_keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
    
    # Prompts
    query_prompt = PromptTemplate.from_template(
        "Based on these keywords: {keywords}, formulate a single concise search query for Arxiv to find the most relevant papers. Only output the query string."
    )
    
    extraction_prompt = PromptTemplate.from_template(
        "Read the following academic documents:\n{docs}\n\n"
        "Extract a list of new, specific technical keywords and their descriptions from these documents. "
        "Format each as strictly 'keyword: description'. Do not include the original keywords: {current_keywords}."
    )
    
    paired_steps = []
    all_extracted_docs = []
    
    # Run the iterative search loop
    iter_i = 0
    while iter_i < budget_max:
        current_kws_str = ", ".join(stack_keywords)
        
        # 1. Form an Arxiv query
        query_chain = query_prompt | llm
        arxiv_query = query_chain.invoke({"keywords": current_kws_str}).content.strip()
        
        # 2. Run the tool to retrieve documents
        try:
            retrieved_docs = arxiv_tool.invoke({"query": arxiv_query})
        except Exception as e:
            retrieved_docs = f"Error retrieving docs: {str(e)}"
            
        all_extracted_docs.append(str(retrieved_docs))
        
        # 3. Extract new keywords
        extract_chain = extraction_prompt | llm
        extraction_result = extract_chain.invoke({
            "docs": retrieved_docs, 
            "current_keywords": current_kws_str
        }).content
        
        # Record the thought (query intent) and observation (retrieved docs + new keywords)
        paired_steps.append(BaseNodeOutcome(
            node_name="agent_2_researcher_intermediate",
            execution_time_seconds=0,
            input=f"Thought: I should search Arxiv for [{arxiv_query}]\nAction: Arxiv\nAction Input: {arxiv_query}",
            outcome=f"Retrieved:\n{retrieved_docs}\n\nExtracted keywords:\n{extraction_result}",
            args={}
        ))
        
        # 4. Parse extraction result to add to stack_keyword
        for line in extraction_result.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("Thought"):
                new_kw = line.split(":")[0].strip().strip("-* ")
                if new_kw and new_kw.lower() not in [k.lower() for k in stack_keywords]:
                    stack_keywords.append(new_kw)
                    
        iter_i += 1

    final_execution_time = time.perf_counter() - start_time
    final_answer = "\n\n---\n\n".join(all_extracted_docs)
    
    return ResearcherNodeOutcome(
        node_name="agent_2_researcher",
        execution_time_seconds=final_execution_time,
        input=keywords_str,
        outcome=final_answer,
        intermediate_steps=paired_steps
    )
