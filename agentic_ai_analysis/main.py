import os
import pickle
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

from agentic_ai_analysis.agents.data_models import PipelineOutcome
from agentic_ai_analysis.core.orchestrator import run_orchestration
from agentic_ai_analysis.core.llm_client import get_embeddings
from agentic_ai_analysis.cka.metrics import compute_cka
from agentic_ai_analysis.scripts.visualize import render_cka_heatmap

# Agent node identifiers for the CKA matrix
AGENT_NODES = [
    "agent_1_keywords",
    "agent_2_thought_1",
    "agent_2_observation_1",
    "agent_2_docs", # Final output of ReAct loop
    "agent_3_distractor",
    "agent_4_judge_docs",
    "agent_4_judge_distractor",
    "agent_5_final"
]

def save_agent_outcomes(results: List[PipelineOutcome], output_dir: str):
    """
    Saves the aggregated textual outcomes for each agent across all N queries 
    into separate pickle files in the output directory, as requested.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # We collect list of texts for each agent node
    node_texts = {node: [] for node in AGENT_NODES}
    
    for res in results:
        node_map = {n.node_name: n.outcome for n in res.nodes.values()}
        for node in AGENT_NODES:
            val = node_map.get(node, "")
            if not str(val).strip():
                # If truly empty (e.g. no second thought), give it a pseudo-random unique string 
                # to prevent zero variance across the column
                val = f"[EMPTY_NODE_{node}_{res.query_id}_{np.random.randint(1000)}]"
            node_texts[node].append(val)
            
    # Save as separate pickle files
    for node, texts in node_texts.items():
        file_path = os.path.join(output_dir, f"{node}_outcomes.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(texts, f)
            
    return node_texts

def compute_and_visualize_cka(node_texts: Dict[str, List[str]], output_dir: str):
    """
    Embeds the texts, formats the feature matrices, and computes the CKA heatmap.
    """
    embeddings_model = get_embeddings()
    num_nodes = len(AGENT_NODES)
    
    # Dictionary to store the embedded feature matrices: {node_name: np.ndarray shape (N, d)}
    node_embeddings = {}
    
    logger.info("Embedding node texts...")
    for node in AGENT_NODES:
        texts = node_texts[node]
        # Embed the batch
        vectors = embeddings_model.embed_documents(texts)
        node_embeddings[node] = np.array(vectors)
        
    logger.info("Computing CKA Matrix...")
    cka_matrix = np.zeros((num_nodes, num_nodes))
    
    for i, node_i in enumerate(AGENT_NODES):
        for j, node_j in enumerate(AGENT_NODES):
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
                
    # Save the raw matrix
    np.save(os.path.join(output_dir, "cka_matrix.npy"), cka_matrix)
    
    # Render and save heatmap
    heatmap_path = os.path.join(output_dir, "cka_heatmap.png")
    render_cka_heatmap(
        cka_matrix=cka_matrix,
        labels=[n.replace("agent_", "") for n in AGENT_NODES],
        output_path=heatmap_path
    )
    logger.info(f"Heatmap saved to {heatmap_path}")

def run_evaluation_pipeline(queries: List[str], output_dir: str):
    """
    The main reusable entrypoint function.
    Given a list of prompts/queries, evaluates them asynchronously across our multi-agent DAG 
    and saves the metrics and node text outputs to the specified output directory.
    """
    logger.info(f"Starting orchestration pipeline for {len(queries)} queries...")
    # Run the DAG asynchronously across queries
    # Note: For testing locally without an LLM running, this will fail. 
    # Assumes vLLM is running locally or you mock `get_llm()`.
    results = run_orchestration(queries)
    
    logger.info(f"Completed {len(results)} queries. Saving outcomes...")
    node_texts = save_agent_outcomes(results, output_dir)
    
    logger.info("Computing metrics based on outcomes...")
    compute_and_visualize_cka(node_texts, output_dir)
    logger.info(f"Pipeline finished successfully. Outputs saved to {output_dir}")
