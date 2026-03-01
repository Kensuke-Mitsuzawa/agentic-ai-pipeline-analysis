from pathlib import Path
import os
import submitit
import logging
import time

from typing import Dict, Any, List

# Agent imports
from ..agents import data_models
from agentic_ai_analysis.agents.researcher import run_researcher
from agentic_ai_analysis.agents.distractor import run_distractor
from agentic_ai_analysis.agents.judge import run_judge
from agentic_ai_analysis.agents.synthesizer import run_synthesizer

logger = logging.getLogger(__name__)


def process_single_query(query: str, query_id: int) -> data_models.PipelineOutcome:
    """
    Executes the multi-agent DAG for a single query.
    Returns the textual outputs for all embedded nodes using the Pydantic model.
    """
    nodes: List[Any] = []

    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Processing query {query_id}: {query}")    
    try:
        logger.info("Running researcher...")
        # Agent 2: Researcher (Modular State-Machine Workflow)
        start = time.perf_counter()
        researcher_data = run_researcher(query)
        t_researcher = time.perf_counter() - start
        extracted_docs = researcher_data.outcome

        nodes.append(data_models.ResearcherNodeOutcome(
            node_name="agent_2_researcher",
            execution_time_seconds=t_researcher,
            input=query,
            outcome=extracted_docs,
            args={},
            intermediate_steps=researcher_data.intermediate_steps
        ))
        # Expose intermediate steps as separate nodes for CKA matrix
        nodes.extend(researcher_data.intermediate_steps)
                    
        # Agent 3: Distractor
        logger.info("Running distractor...")        
        start = time.perf_counter()
        distractor_fact = run_distractor(query)
        t_distractor = time.perf_counter() - start
        nodes.append(data_models.BaseNodeOutcome(
            node_name="agent_3_distractor",
            execution_time_seconds=t_distractor,
            input=query,
            outcome=distractor_fact,
            args={}
        ))
        
        # Agent 4: Judge (Evaluate 1) On retrieved context
        logger.info("Running judge...")
        start = time.perf_counter()
        judge_xml_docs, parsed_docs = run_judge(query, extracted_docs)
        t_judge1 = time.perf_counter() - start
        nodes.append(data_models.BaseNodeOutcome(
            node_name="agent_4_judge_docs",
            execution_time_seconds=t_judge1,
            input=extracted_docs,
            outcome=judge_xml_docs,
            args={"is_related": parsed_docs["is_related"]}
        ))
        
        # Agent 4: Judge (Evaluate 2) On distractor context
        logger.info("Running judge...")
        start = time.perf_counter()
        judge_xml_distractor, parsed_distractor = run_judge(query, distractor_fact)
        t_judge2 = time.perf_counter() - start
        nodes.append(data_models.BaseNodeOutcome(
            node_name="agent_4_judge_distractor",
            execution_time_seconds=t_judge2,
            input=distractor_fact,
            outcome=judge_xml_distractor,
            args={"is_related": parsed_distractor["is_related"]}
        ))
        
        # Agent 5: Synthesizer
        logger.info("Running synthesizer...")
        valid_explanations = []
        if parsed_docs["is_related"]:
            valid_explanations.append(parsed_docs["explanation"])
            
        if parsed_distractor["is_related"]:
            valid_explanations.append(parsed_distractor["explanation"])
            
        start = time.perf_counter()
        final_answer = run_synthesizer(query, valid_explanations)
        t_synth = time.perf_counter() - start
        nodes.append(data_models.BaseNodeOutcome(
            node_name="agent_5_final",
            execution_time_seconds=t_synth,
            input=str(valid_explanations),
            outcome=final_answer,
            args={}
        ))
        
        return data_models.PipelineOutcome(
            query_id=query_id,
            success=True,
            error=None,
            nodes={n.node_name: n for n in nodes}
        )
        
    except Exception as e:
        err_str = f"Error querying LLM: {str(e)}"
        for node_name in [
            "agent_2_node_a_retriever", "agent_2_node_b_filter", "agent_2_node_c_judge",
            "agent_2_researcher", "agent_3_distractor", 
            "agent_4_judge_docs", "agent_4_judge_distractor", "agent_5_final"
        ]:
            nodes.append(data_models.BaseNodeOutcome(
                node_name=node_name,
                execution_time_seconds=0.0,
                input="",
                outcome=f"{err_str} - {node_name} (Query {query_id})",
                args={}
            ))
            
        return data_models.PipelineOutcome(
            query_id=query_id,
            success=False,
            error=err_str,
            nodes={n.node_name: n for n in nodes}
        )

# task: the submitit parameters should be controlled by a config object.
def run_orchestration(queries: List[str], log_folder: Path = Path("submitit_logs")) -> List[data_models.PipelineOutcome]:
    """
    Uses submitit to dispatch tasks.
    In a local prototyping setting, we use the local executor. 
    On HPC this easily scales by changing LocalExecutor to AutoExecutor.
    """
    log_folder.mkdir(parents=True, exist_ok=True)
    
    # Using local executor for local GPU prototype
    executor = submitit.AutoExecutor(folder=log_folder.as_posix(), cluster="debug")
    
    # Configure parameters. Limiting to physical cores for concurrency.
    executor.update_parameters(timeout_min=60, cpus_per_task=1)
    
    # Map the tasks to workers
    jobs = []
    for i, query in enumerate(queries):
        job = executor.submit(process_single_query, query, i)
        logger.info(f"Submitted job {i}: {query}")
        jobs.append(job)
        logger.info(f"Job {i} submitted. Job-id: {job.job_id}")
    # end for
        
    # Wait for completion and collect results
    results = [job.result() for job in jobs]
    return results
