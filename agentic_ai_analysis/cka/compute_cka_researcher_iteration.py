import logging
from typing import List, Tuple
import numpy as np
from pathlib import Path

from ..core.llm_client import get_embeddings
from ..cka import metrics
from ..agents.data_models import PipelineOutcome, ResearcherNodeOutcome
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

def generate_mermaid_graph_iterations(seq_cka_values: List[Tuple[int, float]]) -> str:
    lines = ["graph LR"]
    lines.append("    final_outcome[final_outcome]")
    
    for i, (order, cka_val) in enumerate(seq_cka_values):
        node_name = f"researcher_{i}"
        lines.append(f"    {node_name}[{node_name}]")
        
    for i, (order, cka_val) in enumerate(seq_cka_values):
        node_name = f"researcher_{i}"
        # Edge to final outcome with CKA value
        lines.append(f"    {node_name} -->|{cka_val:.2f}| final_outcome")
        # Connect sequence of iterations without weights
        if i < len(seq_cka_values) - 1:
            next_node = f"researcher_{i+1}"
            lines.append(f"    {node_name} --> {next_node}")
            
    return "\n".join(lines)



def calculate_cka_over_researcher_iterations(
    pipeline_objects: List[PipelineOutcome],
    embeddings_model: Embeddings,
    node_key: str = "agent_2_researcher") -> List[Tuple[int, float]]:
    """Computing a CKA between researcher's intermediate outcome and the final-outcome of the whole agent system.
    """
    # Extract sequence of intermediate node orders
    first_researcher = pipeline_objects[0].nodes[node_key]
    assert isinstance(first_researcher, ResearcherNodeOutcome), f"Node '{node_key}' is not a ResearcherNodeOutcome"
    seq_orders = [step.node_order for step in first_researcher.intermediate_steps]

    # Extract final_outcome texts across all pipelines and embed
    final_outcomes_texts = []
    for p in pipeline_objects:
        r_node = p.nodes[node_key]
        # In BaseModel, 'outcome' represents the final outcome of the node
        final_outcome = getattr(r_node, "final_outcome", r_node.outcome)
        final_outcomes_texts.append(final_outcome)
    
    logger.info("Embedding final researcher outcomes...")
    vecs_final = np.array(embeddings_model.embed_documents(final_outcomes_texts))
    
    results = []
    # 3. For each intermediate step node_order, extract outcomes and compute CKA against final
    for order in seq_orders:
        inter_texts = []
        for p in pipeline_objects:
            # Extract an ResearcherNodeOutcome object from PipelineOutcome.nodes
            r_node = p.nodes[node_key]
            assert isinstance(r_node, ResearcherNodeOutcome), f"Node '{node_key}' is not a ResearcherNodeOutcome"
        
            # Extract ResearcherNodeOutcome.intermediate_steps list
            intermediate_steps = r_node.intermediate_steps
            
            # Form list [(node_order, outcome), (node_order, outcome), ...]
            seq_researcher_outcomes = [(step.node_order, step.outcome) for step in intermediate_steps]
            
            # Find the outcome corresponding to the current node_order
            curr_outcome = None
            for node_order, outcome in seq_researcher_outcomes:
                if node_order == order:
                    curr_outcome = outcome
                    break
            
            if curr_outcome is None:
                curr_outcome = ""
                
            inter_texts.append(curr_outcome)
        
        logger.info(f"Embedding intermediate outcomes for step order {order}...")
        vecs_inter = np.array(embeddings_model.embed_documents(inter_texts))
        
        # Compute the CKA value between (outcome, final_outcome) -> i.e. across all N samples
        cka_val = metrics.main(vecs_inter, vecs_final)
        results.append((order, float(cka_val)))
    # end for

    return results


def compute_cka_researcher_iteration(
    pipeline_objects: List[PipelineOutcome],
    path_dir_cka_researcher: Path,
    file_name_mermaid: str = "cka_researcher_iterations_graph.mmd"
) -> List[Tuple[int, float]]:
    """
    Extracts intermediate outcomes and the final outcome from researcher nodes,
    computes embeddings, and calculates the CKA values between them.
    """
    embeddings_model = get_embeddings()
    
    node_key = "agent_2_researcher"
    
    seq_cka_values = calculate_cka_over_researcher_iterations(
        pipeline_objects,
        embeddings_model,
        node_key=node_key
    )
    
    # creating the mermaid graph
    mermaid_str = generate_mermaid_graph_iterations(seq_cka_values)
    _path_output_mermaid: Path = path_dir_cka_researcher / file_name_mermaid
    with open(_path_output_mermaid, "w") as f:
        f.write(mermaid_str)
    
    logger.info(f"Mermaid graph saved to {_path_output_mermaid}")

    return seq_cka_values
