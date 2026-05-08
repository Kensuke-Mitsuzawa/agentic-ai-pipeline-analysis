from pathlib import Path
import os
import joblib
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from .core.local_server import start_local_server, stop_local_server, LocalServerConfig
from .core.configs_hpc import SubmititSystemConfig
from .agents.data_models import PipelineOutcome
from .core.orchestrator import run_orchestration
from .core.llm_client import get_embeddings
from .cka import compute_cka_agent_nodes
from .llm_ops.langfuse_tracing import get_tracer
from .llm_ops.evaluation.runner import evaluate_outcomes

import pydantic

logger = logging.getLogger(__name__)



def load_results(path_results: list[Path]) -> list[PipelineOutcome]:
    """
    Loads the results from the pickle files and returns a dictionary of node texts.
    """
    _seq_stack = []
    for result in path_results:
        _obj = joblib.load(result)
        _obj_pipeline = PipelineOutcome(**_obj)
        _seq_stack.append(_obj_pipeline)
    # end
    
    return _seq_stack


def run_evaluation_pipeline(
    queries: List[str],
    hpc_config: SubmititSystemConfig,
    output_dir: Path,
    server_config: Optional[LocalServerConfig] = None,
    generation_parameters: Optional[Any] = None,
    evaluation_sampling_rate: float = 0.0,
    compute_cka: bool = False,
):
    """
    The main reusable entrypoint function.
    Given a list of prompts/queries, evaluates them asynchronously across our multi-agent DAG 
    and saves the metrics and node text outputs to the specified output directory.
    If server_config is provided, it launches a local LLM server prior to running.
    """
    logger.info(f"Starting orchestration pipeline for {len(queries)} queries...")
    
    # if server_config is not None:
    #     logger.info("Launching local LLM server...")
    #     start_local_server(server_config)
    # # end if

    try:
        _seq_worker_envelopes = run_orchestration(
            queries, 
            hpc_config=hpc_config,
            local_server_config=server_config,
            generation_parameters=generation_parameters)
        _n_total_tasks = len(_seq_worker_envelopes)
        _n_success_tasks = sum([1 for _env in _seq_worker_envelopes if _env.job_status == "success"])
        _n_failed_tasks = _n_total_tasks - _n_success_tasks
        logger.info(f"Total tasks: {_n_total_tasks}, Success: {_n_success_tasks}, Failed: {_n_failed_tasks}")

        path_results = [_obj.path_results for _obj in _seq_worker_envelopes if _obj.path_results is not None]
        pipeline_objects = load_results(path_results)

        # Optional quality evaluation (sampled) + Langfuse score logging.
        if evaluation_sampling_rate > 0.0:
            tracer = get_tracer()
            eval_map = evaluate_outcomes(pipeline_objects, sampling_rate=evaluation_sampling_rate)
            outcome_map = {o.query_id: o for o in pipeline_objects}
            for qid, metrics in eval_map.items():
                outcome = outcome_map.get(qid)
                # Re-fetch/update trace info so Langfuse UI shows the full context in the evaluation view
                trace = tracer.start_trace(
                    trace_id=qid, 
                    name="rag_pipeline", 
                    input=outcome.prompt if outcome else "",
                    output=outcome.final_outcome if outcome else None,
                    metadata={}
                )
                for m in metrics:
                    tracer.score(trace, name=m.name, value=m.value)

        # Optional CKA metric reporting (custom metric).
        if compute_cka:
            tracer = get_tracer()
            cka_matrix = compute_cka_agent_nodes.compute_and_visualize_cka(pipeline_objects, output_dir)
            tri = np.tril(cka_matrix, k=-1)
            vals = tri[tri != 0]
            cka_mean = float(vals.mean()) if vals.size else 0.0
            for o in pipeline_objects:
                trace = tracer.start_trace(
                    trace_id=o.query_id, 
                    name="rag_pipeline", 
                    input=o.prompt, 
                    output=o.final_outcome,
                    metadata={}
                )
                tracer.score(trace, name="cka_mean", value=cka_mean)
        
        # Ensure all traces are sent to the server
        get_tracer().flush()

        return pipeline_objects
    except Exception as e:
        logger.error(f"{e}")
        raise
    # finally:
        # if server_config is not None:
        #     logger.info("Stopping local LLM server...")
        #     stop_local_server()
        # # end if
    # end try
