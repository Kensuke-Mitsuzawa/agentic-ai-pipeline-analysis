import os
import submitit
from typing import Dict, Any, List

# Agent imports
from agentic_ai_analysis.agents.extractor import extract_keywords
from agentic_ai_analysis.agents.researcher import run_researcher
from agentic_ai_analysis.agents.distractor import run_distractor
from agentic_ai_analysis.agents.judge import run_judge
from agentic_ai_analysis.agents.synthesizer import run_synthesizer

def process_single_query(query: str, query_id: int) -> Dict[str, Any]:
    """
    Executes the multi-agent DAG for a single query.
    Returns the textual outputs for all embedded nodes.
    """
    try:
        # Agent 1: Keyword Extractor
        keywords = extract_keywords(query)
        
        # Agent 2: Researcher (ReAct Loop)
        researcher_data = run_researcher(keywords)
        extracted_docs = researcher_data["final_answer"]
        intermediate_steps = researcher_data["intermediate_steps"]
        
        # Agent 3: Distractor
        distractor_fact = run_distractor(query)
        
        # Agent 4: Judge (Evaluate 1) On retrieved context
        judge_xml_docs, parsed_docs = run_judge(query, extracted_docs)
        
        # Agent 4: Judge (Evaluate 2) On distractor context
        judge_xml_distractor, parsed_distractor = run_judge(query, distractor_fact)
        
        # Agent 5: Synthesizer
        # Only pass explanations where the judge marked them as highly related
        valid_explanations = []
        if parsed_docs["is_related"]:
            valid_explanations.append(parsed_docs["explanation"])
            
        if parsed_distractor["is_related"]:
            valid_explanations.append(parsed_distractor["explanation"])
            
        final_answer = run_synthesizer(query, valid_explanations)
        
        # Collect all text payloads that will form our node embeddings
        return {
            "query_id": query_id,
            "success": True,
            "agent_1_keywords": keywords,
            "agent_2_docs": extracted_docs,
            "agent_2_thought_1": intermediate_steps[0]["thought_action"] if len(intermediate_steps) > 0 else "",
            "agent_2_observation_1": intermediate_steps[0]["observation"] if len(intermediate_steps) > 0 else "",
            "agent_2_thought_2": intermediate_steps[1]["thought_action"] if len(intermediate_steps) > 1 else "",
            "agent_2_observation_2": intermediate_steps[1]["observation"] if len(intermediate_steps) > 1 else "",
            "agent_3_distractor": distractor_fact,
            "agent_4_judge_docs": judge_xml_docs,
            "agent_4_judge_distractor": judge_xml_distractor,
            "agent_5_final": final_answer
        }
    except Exception as e:
        return {
            "query_id": query_id,
            "success": False,
            "error": str(e)
        }

def run_orchestration(queries: List[str], log_folder: str = "submitit_logs") -> List[Dict[str, Any]]:
    """
    Uses submitit to dispatch tasks.
    In a local prototyping setting, we use the local executor. 
    On HPC this easily scales by changing LocalExecutor to AutoExecutor.
    """
    os.makedirs(log_folder, exist_ok=True)
    
    # Using local executor for local GPU prototype
    executor = submitit.LocalExecutor(folder=log_folder)
    
    # Configure parameters. Limiting to physical cores for concurrency.
    executor.update_parameters(timeout_min=60, cpus_per_task=1)
    
    # Map the tasks to workers
    jobs = []
    for i, query in enumerate(queries):
        job = executor.submit(process_single_query, query, i)
        jobs.append(job)
        
    # Wait for completion and collect results
    results = [job.result() for job in jobs]
    return results
