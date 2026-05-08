from pathlib import Path
import os
import joblib
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from .core.local_server import start_local_server, stop_local_server, LocalServerConfig
from .core.configs_hpc import SlurmSystemConfig
from .agents.data_models import PipelineOutcome
from .core.orchestrator import run_orchestration
from .core.llm_client import get_embeddings
from .cka import compute_cka_agent_nodes

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
    hpc_config: SlurmSystemConfig,
    output_dir: Path,
    server_config: Optional[LocalServerConfig] = None
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
            local_server_config=server_config)
        _n_total_tasks = len(_seq_worker_envelopes)
        _n_success_tasks = sum([1 for _env in _seq_worker_envelopes if _env.job_status == "success"])
        _n_failed_tasks = _n_total_tasks - _n_success_tasks
        logger.info(f"Total tasks: {_n_total_tasks}, Success: {_n_success_tasks}, Failed: {_n_failed_tasks}")

        path_results = [_obj.path_results for _obj in _seq_worker_envelopes if _obj.path_results is not None]
        pipeline_objects = load_results(path_results)

        # logger.info("Computing metrics based on outcomes...")
        # compute_cka_agent_nodes.compute_and_visualize_cka(pipeline_objects, output_dir)
        # logger.info(f"Pipeline finished successfully. Outputs saved to {output_dir}")
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
