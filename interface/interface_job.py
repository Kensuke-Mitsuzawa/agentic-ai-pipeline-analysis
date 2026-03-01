import os
import argparse
import logging
import tomllib
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

from agentic_ai_analysis.main import run_evaluation_pipeline
from agentic_ai_analysis.core.local_server import LocalServerConfig
from agentic_ai_analysis.core.configs_hpc import SlurmSystemConfig

logger = logging.getLogger()

# 
def main():
    parser = argparse.ArgumentParser(description="Run the Agentic AI Pipeline")
    parser.add_argument('-c', "--path_config", type=str, 
                        required=True, 
                        help="Path to the configuration file.")
    parser.add_argument('-n', "--n_samples", type=int, default=15, 
                        help="Number of samples to run from the HuggingFace dataset.")
    parser.add_argument('-e', 
                        "--env_file", 
                        type=str, default=".env",
                        help="Path to the .env file")
    args = parser.parse_args()
    
    # Configure root logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load .env file if it exists
    load_dotenv(args.env_file)

    config_path = Path(args.path_config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "rb") as f:
        config_dict = tomllib.load(f)
        
    # Build configurations
    slurm_dict = config_dict.get("slurm_system", {})
    server_dict = config_dict.get("local_server", {})
    
    hpc_config = SlurmSystemConfig(**slurm_dict)
    server_config = LocalServerConfig(**server_dict) if server_dict else None
    
    logger.info("=== Running HuggingFace ArxivQA ===")
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("The 'datasets' package is required. Install via 'uv pip install datasets'.")
        return
        
    logger.info("Loading dataset from HuggingFace...")
    dataset = load_dataset("MMInstruction/ArxivQA", split="train")
    
    # We use the 'question' column from the dataset as the query
    queries = dataset["question"][:args.n_samples]
    output_dir = Path(hpc_config.log_folder)
    
    logger.info("Invoking run_evaluation_pipeline...")
    results = run_evaluation_pipeline(
        queries=queries, 
        output_dir=output_dir, 
        hpc_config=hpc_config, 
        server_config=server_config
    )
            
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
