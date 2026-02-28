import os
import pickle
import numpy as np
from typing import List, Dict, Any

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

def load_mini_dataset(n: int = 10) -> List[str]:
    """
    Mocking a HuggingFace ArxivQA dataset load for local prototype.
    In real HPC deployment, this uses `datasets.load_dataset`.
    """
    return [
        "What are the recent advancements in quantum error correction?",
        "Can you explain the difference between LoRA and QLoRA for LLM fine-tuning?",
        "How do transformers handle long context windows efficiently?",
        "What is the principle behind Centered Kernel Alignment in neural networks?",
        "Describe the mathematical foundation of diffusion models.",
        "How are graph neural networks applied in drug discovery?",
        "What are the main challenges in multi-agent reinforcement learning?",
        "Explain the mechanism of flash attention.",
        "How does federated learning ensure data privacy?",
        "What is the state-of-the-art in text-to-video generation?"
    ][:n]

def save_agent_outcomes(results: List[Dict[str, Any]], output_dir: str):
    """
    Saves the aggregated textual outcomes for each agent across all N queries 
    into separate pickle files in the output directory, as requested.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # We collect list of texts for each agent node
    node_texts = {node: [] for node in AGENT_NODES}
    
    for res in results:
        if not res.get("success", False):
            # Pad with empty string on failure to maintain matrix shape
            for node in AGENT_NODES:
                node_texts[node].append("")
            continue
            
        for node in AGENT_NODES:
            node_texts[node].append(res.get(node, ""))
            
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
    
    print("Embedding node texts...")
    for node in AGENT_NODES:
        texts = node_texts[node]
        # Replace empty strings with a generic placeholder so embeddings model doesn't crash
        texts = [t if str(t).strip() else "[NO OUTPUT]" for t in texts]
        
        # Embed the batch
        vectors = embeddings_model.embed_documents(texts)
        node_embeddings[node] = np.array(vectors)
        
    print("Computing CKA Matrix...")
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
    print(f"Heatmap saved to {heatmap_path}")

def main():
    n_queries = 5  # Small N for local prototyping
    queries = load_mini_dataset(n=n_queries)
    output_dir = "pipeline_outcomes"
    
    print(f"Starting orchestration pipeline for {n_queries} queries...")
    # Run the DAG asynchronously across queries
    # Note: For testing locally without an LLM running, this will fail. 
    # Assumes vLLM is running locally or you mock `get_llm()`.
    results = run_orchestration(queries)
    
    print(f"Completed {len(results)} queries. Saving outcomes...")
    node_texts = save_agent_outcomes(results, output_dir)
    
    print("Computing metrics based on outcomes...")
    compute_and_visualize_cka(node_texts, output_dir)
    print("Pipeline finished successfully.")

if __name__ == "__main__":
    main()
