import os
import argparse
import sys
import logging
import tomllib
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Early loading of dotenv so that env vars like HF_HOME are set before HF libraries are imported
_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument('-e', '--env_file', type=str, default='.env')
_early_args, _ = _early_parser.parse_known_args()
load_dotenv(_early_args.env_file, override=True)

print(f"HF_HOME: {os.environ.get('HF_HOME')}")
from agentic_ai_analysis.main import run_evaluation_pipeline
from agentic_ai_analysis.core.local_server import LocalServerConfig
from agentic_ai_analysis.core.configs_hpc import SubmititSystemConfig

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
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    config_path = Path(args.path_config)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "rb") as f:
        config_dict = tomllib.load(f)
        
    # Build configurations
    submitit_dict = config_dict.get("submitit_system", config_dict.get("slurm_system", {}))
    server_dict = config_dict.get("local_server", {})
    llm_client_dict = config_dict.get("llm_client", {})
    
    hpc_config = SubmititSystemConfig(**submitit_dict)

    # If you're using a port-forwarded *remote* server (already running at localhost:8000),
    # we should NOT attempt to start a local HF server. Make it opt-in via config:
    #   [local_server]
    #   start = true
    should_start_local_server = bool(server_dict.get("start", False))
    server_config = LocalServerConfig(**server_dict) if (server_dict and should_start_local_server) else None

    # Optional: allow the config to specify the OpenAI-compatible endpoint/model for the agents.
    # This is useful for port-forward setups to a remote GPU machine.
    # Example:
    #   [llm_client]
    #   openai_api_base = "http://localhost:8000/v1/"
    #   model = "Qwen/Qwen3.5-27B-FP8"
    if "openai_api_base" in llm_client_dict:
        os.environ["OPENAI_API_BASE"] = str(llm_client_dict["openai_api_base"])
    if "model" in llm_client_dict:
        os.environ["OPENAI_MODEL"] = str(llm_client_dict["model"])
    
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
