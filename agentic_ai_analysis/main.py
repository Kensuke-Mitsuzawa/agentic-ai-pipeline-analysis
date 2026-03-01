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
from .cka.metrics import compute_cka
from .scripts.visualize import render_cka_heatmap

import pydantic

logger = logging.getLogger(__name__)



def compute_and_visualize_cka(pipeline_objects: List[PipelineOutcome], output_dir: Path):
    """
    Embeds the texts, formats the feature matrices, and computes the CKA heatmap.
    """
    embeddings_model = get_embeddings()
    
    _seq_n_nodes_pipeline = [len(_out.get_node_names()) for _out in pipeline_objects]
    assert len(set(_seq_n_nodes_pipeline)) == 1, "All pipeline objects must have the same number of nodes."
    num_nodes = _seq_n_nodes_pipeline[0]
    
    # Dictionary to store the embedded feature matrices: {node_name: np.ndarray shape (N, d)}
    node_embeddings: dict[str, np.ndarray] = {}
    
    logger.info("Embedding node texts...")
    for node in pipeline_objects[0].get_node_names():
        texts = [p.nodes[node].outcome for p in pipeline_objects]
        # Embed the batch
        vectors = embeddings_model.embed_documents(texts)
        node_embeddings[node] = np.array(vectors)
    # end

    logger.info("Computing CKA Matrix...")
    cka_matrix = np.zeros((num_nodes, num_nodes))
    
    for i, node_i in enumerate(pipeline_objects[0].get_node_names()):
        for j, node_j in enumerate(pipeline_objects[0].get_node_names()):
            # Optimization: matrix is symmetric, diagonals are 1.0
            if i == j:
                cka_matrix[i, j] = 1.0
            elif j < i:
                cka_matrix[i, j] = cka_matrix[j, i]
            else:
                X = node_embeddings[node_i]
                Y = node_embeddings[node_j]
                score = compute_cka(X, Y)
                cka_matrix[i, j] = score
        # end
    # end
    
    # Save the raw matrix
    _path_output_matrix: Path = output_dir / "cka_matrix.npy"
    np.save(_path_output_matrix, cka_matrix)
    
    # Render and save heatmap
    _path_output_heatmap: Path = output_dir / "cka_heatmap.png"
    render_cka_heatmap(
        cka_matrix=cka_matrix,
        labels=[n.replace("agent_", "") for n in pipeline_objects[0].get_node_names()],
        output_path=_path_output_heatmap.as_posix()
    )
    logger.info(f"Heatmap saved to {_path_output_heatmap}")


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
    
    if server_config is not None:
        logger.info("Launching local LLM server...")
        start_local_server(server_config)
    # end if

    try:
        path_results = run_orchestration(queries, hpc_config=hpc_config)
        logger.info("Computing metrics based on outcomes...")

        pipeline_objects = load_results(path_results)
        compute_and_visualize_cka(pipeline_objects, output_dir)
        logger.info(f"Pipeline finished successfully. Outputs saved to {output_dir}")
        return pipeline_objects
    finally:
        if server_config is not None:
            logger.info("Stopping local LLM server...")
            stop_local_server()
        # end if
    # end try
