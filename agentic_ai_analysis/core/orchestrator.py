from pathlib import Path
import os
import submitit
import logging
import time
import joblib
import math
import hashlib

from typing import Dict, Any, List, Optional, NamedTuple, Union, Optional

# Agent imports
from ..agents import data_models
from agentic_ai_analysis.agents.researcher import run_researcher
from agentic_ai_analysis.agents.distractor import run_distractor
from agentic_ai_analysis.agents.judge import run_judge
from agentic_ai_analysis.agents.synthesizer import run_synthesizer

from agentic_ai_analysis.core.configs_hpc import SlurmSystemConfig
import math


logger = logging.getLogger(__name__)



def process_single_query(query: str, query_id: str) -> Optional[data_models.PipelineOutcome]:
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
            prompt=query,
            query_id=query_id,
            final_outcome=final_answer,
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
    query_id: str
    log_folder: Path
    chunk_idx: int


class WorkerEnvelope(NamedTuple):
    args: WorkerFunctionArgs
    path_results: Optional[Path]
    job_status: str


def main_worker(args: WorkerFunctionArgs) -> WorkerEnvelope:
    result = process_single_query(args.query, args.query_id) 
    # Save results for each query in the chunk
    path_file = args.log_folder / "outcomes" /  f"{args.query_id}_result.pkl"
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



def run_orchestration(
    queries: List[str], 
    hpc_config: SlurmSystemConfig, 
    profile_names: Optional[List[str]] = None) -> List[Any]:
    """
    Uses submitit to dispatch tasks across one or multiple heterogeneous SLURM partitions.
    """
    log_folder = hpc_config.log_folder
    log_folder.mkdir(parents=True, exist_ok=True)
    
    # Normalize profiles into a list
    if profile_names is None:
        profile_names = hpc_config.default_profiles
    # end

    profiles = [hpc_config.profiles[p] for p in profile_names]

    # 1. Distribute queries proportionally across the chosen profiles based on node budget
    total_budget = sum(p.n_nodes_budget for p in profiles)
    if total_budget == 0: 
        total_budget = len(profiles) # Fallback to even split
    # end if

    # TODO filter the existing outcomes

    profile_query_splits = []
    start_idx = 0
    for p in profiles:
        # Calculate how many queries this profile should handle
        share = math.ceil(len(queries) * (p.n_nodes_budget / total_budget))
        profile_query_splits.append(queries[start_idx : start_idx + share])
        start_idx += share

    all_jobs = []
    # global_query_id = 0

    # 2. Setup Executors and Submit Jobs per Profile
    for p_name, profile, assigned_queries in zip(profile_names, profiles, profile_query_splits):
        if not assigned_queries:
            continue
            
        logger.info(f"Configuring executor for partition '{p_name}' with {len(assigned_queries)} queries.")

        # Initialize the specific executor
        executor = submitit.AutoExecutor(folder=log_folder.as_posix())
        
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

        # Chunk the queries assigned to this specific profile
        n_chunks = profile.n_nodes_budget
        chunk_size = math.ceil(len(assigned_queries) / n_chunks) if n_chunks > 0 else len(assigned_queries)
        chunk_size = max(1, chunk_size)
            
        chunks = [assigned_queries[i:i + chunk_size] for i in range(0, len(assigned_queries), chunk_size) if len(assigned_queries[i:i + chunk_size]) > 0]

        # Submit jobs for this profile
        for chunk_idx, chunk in enumerate(chunks):
            # submitit.batch() is highly recommended for submitting multiple jobs quickly
            with executor.batch():
                _item_chunk: str
                for _item_chunk in chunk:
                    _query_id = hashlib.sha256(_item_chunk.encode("utf-8")).hexdigest()
                    _worker_func_args = WorkerFunctionArgs(
                        query=_item_chunk,
                        query_id=_query_id,
                        log_folder=log_folder,
                        chunk_idx=chunk_idx
                    )
                    job = executor.submit(main_worker, _worker_func_args)
                    all_jobs.append(job)
                    # global_query_id += 1
                    
        logger.info(f"Finished submitting jobs to {p_name}.")

    # 3. Wait and collect results AFTER all jobs are successfully submitted to the cluster
    logger.info(f"Waiting for all {len(all_jobs)} jobs to complete across all partitions...")
    
    all_results = []
    for _job in all_jobs:
        # This will block until the specific job finishes
        chunk_results = _job.result() 
        logger.info(f"Collected results for job {_job.job_id}")
        all_results.append(chunk_results)
    
    return all_results