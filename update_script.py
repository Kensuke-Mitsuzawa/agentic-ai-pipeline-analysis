import re

with open('agentic_ai_analysis/core/orchestrator.py', 'r') as f:
    content = f.read()

new_logic = """
from .configs_hpc import SlurmSystemConfig

def process_query_chunk(queries: List[str], start_idx: int) -> List[data_models.PipelineOutcome]:
    results = []
    for i, q in enumerate(queries):
        results.append(process_single_query(q, start_idx + i))
    return results

# task: the submitit parameters should be controlled by a config object.
def run_orchestration(queries: List[str], config: SlurmSystemConfig, profile_name: str = None) -> List[Any]:
    \"\"\"
    Uses submitit to dispatch tasks.
    In a local prototyping setting, we use the local executor. 
    On HPC this easily scales by changing LocalExecutor to AutoExecutor.
    \"\"\"
    log_folder = config.log_folder
    log_folder.mkdir(parents=True, exist_ok=True)
    
    profile_name = profile_name or config.default_profile
    profile = config.profiles[profile_name]
    
    # task: `queries` should be chunked into smaller lists of queries.
    # task: the chunk number is `SlurmProfile.n_nodes_budget`
    n_chunks = profile.n_nodes_budget
    if n_chunks > 0:
        chunk_size = max(1, len(queries) // n_chunks)
        # Handle remainder nicely if division is not even
        chunk_size = __import__('math').ceil(len(queries) / n_chunks)
    else:
        chunk_size = len(queries)
        
    chunks = [queries[i:i + chunk_size] for i in range(0, len(queries), chunk_size)]

    # Using local executor for local GPU prototype
    # AutoExecutor will use slurm if available, else local
    executor = submitit.AutoExecutor(folder=log_folder.as_posix())
    
    # Configure parameters. Limiting to physical cores for concurrency.
    # task: the parameter should follow `SlurmProfile`
    time_parts = list(map(int, profile.time.split(':')))
    timeout_min = time_parts[0] * 60 + time_parts[1] + time_parts[2] / 60.0
    
    update_kwargs = {
        "timeout_min": int(timeout_min),
        "slurm_partition": profile.partition,
        "slurm_nodes": 1, # each task (chunk) will be a job running on 1 node
        "slurm_ntasks_per_node": profile.n_tasks_per_node,
        "cpus_per_task": profile.n_cpus_per_task, # local executor uses this
        "slurm_cpus_per_task": profile.n_cpus_per_task,
    }
    if profile.gres:
        update_kwargs["slurm_gres"] = profile.gres
        
    executor.update_parameters(**update_kwargs)

    # this should be the loop over the chunks.
    # task: update the logic below. submiting a set of jobs in a chunk.    
    # task: collect result and save it to a file, for each.
    jobs = []
    start_id = 0
    for chunk_idx, chunk in enumerate(chunks):
        job = executor.submit(process_query_chunk, chunk, start_id)
        logger.info(f"Submitted chunk job {chunk_idx} starting at index {start_id}")
        jobs.append(job)
        start_id += len(chunk)
        
    # Wait for completion and collect results
    all_results = []
    for chunk_idx, job in enumerate(jobs):
        chunk_results = job.result()
        all_results.extend(chunk_results)
        
        for res in chunk_results:
            save_agent_outcomes(res, log_folder / f"outcome_{res.query_id}")

    return all_results
"""

# Replace from `# task: the submitit parameters should be controlled by a config object.` to the end
start_idx = content.find('# task: the submitit parameters should be controlled by a config object.')
content = content[:start_idx] + new_logic

with open('agentic_ai_analysis/core/orchestrator.py', 'w') as f:
    f.write(content)
