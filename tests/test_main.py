import logging
import argparse
from typing import List
from pathlib import Path

from agentic_ai_analysis.main import run_evaluation_pipeline
from agentic_ai_analysis.core.local_server import LocalServerConfig
from agentic_ai_analysis.core.configs_hpc import SlurmSystemConfig, SlurmProfile

logger = logging.getLogger(__name__)

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

def test_mini_dataset():
    """Test mode 1: Test with the input from `load_mini_dataset()`"""
    n_queries = 10  # Small N for local prototyping
    queries = load_mini_dataset(n=n_queries)
    output_dir = Path("./pipeline_outcomes_mini")
    
    server_config = LocalServerConfig(
        model_id="Qwen/Qwen2.5-3B-Instruct", # Super small model for fast test bootup
        port=8000,
        quantization_config_dict=dict(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16")
    )

    hpc_config = SlurmSystemConfig(
        log_folder=Path("./pipeline_outcomes_mini"),
        default_profile="local",
        profiles={
            "local": SlurmProfile(
                partition="dev",
                time="00:10:00",
                n_nodes_budget=1,
                n_tasks_per_node=1,
                n_cpus_per_task=1,
                n_gpus_per_task=1,
                gres="gpu:1",
                mem_per_gpu="16G",
            )
        }
    )
    
    logger.info("=== Running Test Mode 1: Mini Dataset ===")
    results = run_evaluation_pipeline(
        queries=queries, 
        hpc_config=hpc_config, 
        output_dir=output_dir, 
        server_config=server_config)
    
    for r in results:
        assert r.success is True, f"Query {r.query_id} failed with error: {r.error}"

def test_hf_dataset(n_samples: int = 15):
    """Test mode 2: Test with the input from the Hugging face dataset MMInstruction/ArxivQA"""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("The 'datasets' package is required for this mode. Install via 'pip install datasets'.")
        return
        
    logger.info("=== Running Test Mode 2: HuggingFace ArxivQA ===")
    logger.info("Loading dataset from HuggingFace...")
    dataset = load_dataset("MMInstruction/ArxivQA", split="train")
    
    # We use the 'question' column from the dataset as the query
    queries = dataset["question"][:n_samples]
    output_dir = Path("./pipeline_outcomes_hf")
    
    server_config = LocalServerConfig(
        model_id="Qwen/Qwen2.5-3B-Instruct", # Super small model for fast test bootup
        port=8000
    )

    hpc_config = SlurmSystemConfig(
        log_folder=Path("./pipeline_outcomes_hf"),
        default_profile="local",
        profiles={
            "local": SlurmProfile(
                partition="dev",
                time="00:10:00",
                n_nodes_budget=1,
                n_tasks_per_node=1,
                n_cpus_per_task=1,
                n_gpus_per_task=1,
                gres="gpu:1",
                mem_per_gpu="16G",
            )
        }
    )
    
    results = run_evaluation_pipeline(
        queries=queries, 
        output_dir=output_dir, 
        hpc_config=hpc_config, 
        server_config=server_config)
    for r in results:
        assert r.success is True, f"Query {r.query_id} failed with error: {r.error}"

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run the Agentic AI CKA Analysis Pipeline")
    # parser.add_argument("--mode", type=str, choices=["mini", "hf"], default="mini", 
    #                     help="Test mode to run: 'mini' for local mock data or 'hf' for HuggingFace ArxivQA dataset.")
    # parser.add_argument("--n_samples", type=int, default=15, 
    #                     help="Number of samples to run when using the 'hf' mode.")
    # args = parser.parse_args()
    
    # Configure root logger to output INFO to console, and DEBUG to a file
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File handler (DEBUG and above)
    file_handler = logging.FileHandler('debug.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    test_mini_dataset()
    
    # if args.mode == "mini":
    #     test_mini_dataset()
    # elif args.mode == "hf":
    #     test_hf_dataset(n_samples=args.n_samples)
