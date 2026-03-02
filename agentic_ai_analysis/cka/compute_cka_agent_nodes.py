from pathlib import Path
import os
import joblib
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from ..core.llm_client import get_embeddings
from ..cka import metrics
from ..scripts.visualize import render_cka_heatmap
from ..agents.data_models import PipelineOutcome

logger = logging.getLogger(__name__)


def generate_mermaid_graph(
    cka_matrix: np.ndarray, 
    original_node_names: list[str], 
    dict_node_name2node_order: dict[str, int], 
    graph_dependency: dict[str, list[str]],
    dependency_start_node: dict[str, list[str]]) -> str:
    """
    Generates a Mermaid graph description string from CKA dependencies.
    """
    lines = ["graph LR"]

    for name in dependency_start_node:
        clean_name = name
        lines.append(f"    {name}[{clean_name}]")
    # end

    for name in original_node_names:
        clean_name = name
        lines.append(f"    {name}[{clean_name}]")
    # end

    for from_node, to_nodes in dependency_start_node.items():
        for to_node in to_nodes:
            lines.append(f"    {from_node} --> {to_node}")
    # end

    for from_node, to_nodes in graph_dependency.items():
        if from_node in dict_node_name2node_order:
            u = dict_node_name2node_order[from_node]
            for to_node in to_nodes:
                if to_node in dict_node_name2node_order:
                    v = dict_node_name2node_order[to_node]
                    weight = cka_matrix[u, v]
                    lines.append(f"    {from_node} -->|{weight:.2f}| {to_node}")
    # end
    
    return "\n".join(lines)


def compute_and_visualize_cka(
    pipeline_objects: List[PipelineOutcome], 
    output_dir: Path, 
    graph_dependency: dict,
    dependency_start_node: dict[str, list[str]]):
    """
    Embeds the texts, formats the feature matrices, and computes the CKA heatmap.
    """
    embeddings_model = get_embeddings()
    
    _seq_n_nodes_pipeline = [len(_out.get_node_names()) for _out in pipeline_objects]
    assert len(set(_seq_n_nodes_pipeline)) == 1, "All pipeline objects must have the same number of nodes."
    num_nodes = _seq_n_nodes_pipeline[0]
    
    dict_node_name2node_order: dict[str, int] = {}
    for node_name, node_outcome in pipeline_objects[0].nodes.items():
        dict_node_name2node_order[node_name] = node_outcome.node_order
    # end for
    
    _sorted_nodes = sorted(pipeline_objects[0].nodes.values(), key=lambda x: x.node_order)
    original_node_names = [_node.node_name for _node in _sorted_nodes]
    
    seq_dependency: list[tuple[str, str]] = []
    for i in range(len(_sorted_nodes)):
        for j in range(i, len(_sorted_nodes)):
            seq_dependency.append((_sorted_nodes[i].node_name, _sorted_nodes[j].node_name))
        # end for
    # end for

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
    
    for _t_node_dep in seq_dependency:
        _node_name_from, _node_name_to = _t_node_dep
        _node_order_from = dict_node_name2node_order[_node_name_from]
        _node_order_to = dict_node_name2node_order[_node_name_to]
 
        _sample_X = node_embeddings[_node_name_from]
        _sample_Y = node_embeddings[_node_name_to]
        score = metrics.main(_sample_X, _sample_Y)
        # Assign symmetrically since visualization masks upper triangle
        cka_matrix[_node_order_from, _node_order_to] = score
        cka_matrix[_node_order_to, _node_order_from] = score
    # end
    
    # Save the raw matrix
    _path_output_matrix: Path = output_dir / "cka_matrix.npy"
    np.save(_path_output_matrix, cka_matrix)
    
    # Render and save heatmap
    _path_output_heatmap: Path = output_dir / "cka_heatmap.png"
    render_cka_heatmap(
        cka_matrix=cka_matrix,
        labels=[n.replace("agent_", "") for n in original_node_names],
        output_path=_path_output_heatmap.as_posix(),
        graph_dependency=graph_dependency,
        original_node_names=original_node_names
    )
    logger.info(f"Graph saved to {_path_output_heatmap}")
    
    # Render and save mermaid graph
    mermaid_str = generate_mermaid_graph(
        cka_matrix=cka_matrix, 
        original_node_names=original_node_names, 
        dict_node_name2node_order=dict_node_name2node_order, 
        graph_dependency=graph_dependency,
        dependency_start_node=dependency_start_node)
    _path_output_mermaid: Path = output_dir / "cka_graph.mmd"
    with open(_path_output_mermaid, "w") as f:
        f.write(mermaid_str)
    logger.info(f"Mermaid graph saved to {_path_output_mermaid}")

    return cka_matrix
