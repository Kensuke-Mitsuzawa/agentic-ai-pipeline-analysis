import logging
import argparse
from typing import List

from agentic_ai_analysis.main import run_evaluation_pipeline

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
    n_queries = 5  # Small N for local prototyping
    queries = load_mini_dataset(n=n_queries)
    output_dir = "pipeline_outcomes_mini"
    
    logger.info("=== Running Test Mode 1: Mini Dataset ===")
    run_evaluation_pipeline(queries, output_dir)

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
    output_dir = "pipeline_outcomes_hf"
    
    run_evaluation_pipeline(queries, output_dir)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run the Agentic AI CKA Analysis Pipeline")
    # parser.add_argument("--mode", type=str, choices=["mini", "hf"], default="mini", 
    #                     help="Test mode to run: 'mini' for local mock data or 'hf' for HuggingFace ArxivQA dataset.")
    # parser.add_argument("--n_samples", type=int, default=15, 
    #                     help="Number of samples to run when using the 'hf' mode.")
    # args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    test_mini_dataset()
    
    # if args.mode == "mini":
    #     test_mini_dataset()
    # elif args.mode == "hf":
    #     test_hf_dataset(n_samples=args.n_samples)
