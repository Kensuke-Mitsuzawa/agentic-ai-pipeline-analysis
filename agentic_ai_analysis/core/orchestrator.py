from pathlib import Path
import os
import submitit
import logging
import time
import joblib

from typing import Dict, Any, List, Optional, NamedTuple

# Agent imports
from ..agents import data_models
from agentic_ai_analysis.agents.researcher import run_researcher
from agentic_ai_analysis.agents.distractor import run_distractor
from agentic_ai_analysis.agents.judge import run_judge
from agentic_ai_analysis.agents.synthesizer import run_synthesizer

from agentic_ai_analysis.core.configs_hpc import SlurmSystemConfig
import math


logger = logging.getLogger(__name__)



def process_single_query(query: str, query_id: int) -> Optional[data_models.PipelineOutcome]:
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
        researcher_data = run_researcher(query, node_order=0)
        extracted_docs = researcher_data.outcome

        nodes.append(researcher_data)
                    
        # Agent 3: Distractor
        logger.info("Running distractor...")        
        start = time.perf_counter()
        distractor_fact = run_distractor(query)
        t_distractor = time.perf_counter() - start
        nodes.append(data_models.BaseNodeOutcome(
            node_order=1,
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
            node_order=2,
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
            node_order=3,
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
            node_order=4,
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
        logger.error(err_str)

        return None


def save_agent_outcomes(results: data_models.PipelineOutcome, path_file: Path) -> Path:
    """
    Saves the aggregated textual outcomes for each agent across all N queries 
    into separate pickle files in the output directory, as requested.
    """
    path_file.parent.mkdir(parents=True, exist_ok=True)
    _obj = results.model_dump()
    joblib.dump(_obj, path_file)
    return path_file

class WorkerFunctionArgs(NamedTuple):
    query: str
    query_id: int
    log_folder: Path
    chunk_idx: int


class WorkerEnvelope(NamedTuple):
    args: WorkerFunctionArgs
    path_results: Optional[Path]
    job_status: str


def main_worker(args: WorkerFunctionArgs) -> WorkerEnvelope:
    result = process_single_query(args.query, args.query_id) 
    # Save results for each query in the chunk
    path_file = args.log_folder / f"{args.query_id}_result.pkl"
    if result is not None:
        _path_file = save_agent_outcomes(result, path_file)
        logger.info(f"saved results for job {path_file}")
        envelope_obj = WorkerEnvelope(
            args=args,
            path_results=_path_file,
            job_status="success",
        )
    else:
        logger.error(f"failed to process query {args.query_id}")
        envelope_obj = WorkerEnvelope(
            args=args,
            path_results=None,
            job_status="failed",
        )
    # end if
    
    return envelope_obj


# task: the submitit parameters should be controlled by a config object.
def run_orchestration(
    queries: List[str], 
    hpc_config: SlurmSystemConfig, 
    profile_name: str = None) -> List[WorkerEnvelope]:
    """
    Uses submitit to dispatch tasks.
    In a local prototyping setting, we use the local executor. 
    On HPC this easily scales by changing LocalExecutor to AutoExecutor.
    """
    log_folder = hpc_config.log_folder
    log_folder.mkdir(parents=True, exist_ok=True)
    
    profile_name = profile_name or hpc_config.default_profile
    profile = hpc_config.profiles[profile_name]
    
    # task: `queries` should be chunked into smaller lists of queries.
    # task: the chunk number is `SlurmProfile.n_nodes_budget`
    n_chunks = profile.n_nodes_budget
    if n_chunks > 0:
        chunk_size = math.ceil(len(queries) / n_chunks)
    else:
        chunk_size = len(queries)
        
    if chunk_size < 1:
        chunk_size = 1
        
    chunks = [queries[i:i + chunk_size] for i in range(0, len(queries), chunk_size) if len(queries[i:i + chunk_size]) > 0]

    # Using local executor for local GPU prototype
    executor = submitit.AutoExecutor(folder=log_folder.as_posix())
    
    # Configure parameters. Limiting to physical cores for concurrency.
    # task: the parameter should follow `SlurmProfile`
    time_parts = list(map(int, profile.time.split(':')))
    timeout_min = time_parts[0] * 60 + time_parts[1] + time_parts[2] / 60.0
    
    update_kwargs = {
        "timeout_min": int(timeout_min),
        "slurm_partition": profile.partition,
        "slurm_nodes": 1,
        "slurm_ntasks_per_node": profile.n_tasks_per_node,
        "cpus_per_task": profile.n_cpus_per_task,
        "slurm_cpus_per_task": profile.n_cpus_per_task,
    }
    if profile.gres:
        update_kwargs["slurm_gres"] = profile.gres
        
    executor.update_parameters(**update_kwargs)

    # this should be the loop over the chunks.
    # task: update the logic below. submiting a set of jobs in a chunk.    
    # task: collect result and save it to a file, for each.

    all_results = []

    start_id = 0
    for chunk_idx, chunk in enumerate(chunks):
        jobs = []        
        for _item_chunk in chunk:
            _worker_func_args = WorkerFunctionArgs(
                query=_item_chunk,
                query_id=start_id,
                log_folder=log_folder,
                chunk_idx=chunk_idx
            )
            job = executor.submit(main_worker, _worker_func_args)
            logger.info(f"Submitted chunk job {chunk_idx}: job_id {job.job_id}")
            jobs.append(job)
            start_id += 1
        # end for

        # task: saving procedure should come here.
        for _job in jobs:
            chunk_results = _job.result()
            logger.info(f"collected results for job {_job.job_id}")

            # Save results for each query in the chunk
            all_results.append(chunk_results)
            # end for
        # end for
    # end for
    
    return all_results
