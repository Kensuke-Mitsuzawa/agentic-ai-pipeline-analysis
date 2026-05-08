import os
import argparse
import sys
import logging
import tomllib
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv, find_dotenv

# Early loading of dotenv so that env vars like HF_HOME are set before HF libraries are imported
_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument('-e', '--env_file', type=str, default='.env')
_early_args, _ = _early_parser.parse_known_args()
load_dotenv(_early_args.env_file, override=True)

print(f"HF_HOME: {os.environ.get('HF_HOME')}")
from agentic_ai_analysis.main import run_evaluation_pipeline
from agentic_ai_analysis.core.configs_hpc import SubmititSystemConfig
from agentic_ai_analysis.core.local_server import (
    LocalServerConfig,
    LLMClientConfig
)
from agentic_ai_analysis.core.llm_client import LlmGenerationParameters


logger = logging.getLogger()


class InterfaceJobConfig(BaseModel):
    submitit_system: SubmititSystemConfig
    local_server: LocalServerConfig
    llm_client: LLMClientConfig
    llm_generation_params: LlmGenerationParameters = Field(default_factory=LlmGenerationParameters)


def start_local_server(server_config: LocalServerConfig):
    # If you're using a port-forwarded *remote* server (already running at localhost:8000),
    # we should NOT attempt to start a local HF server. Make it opt-in via config:
    #   [local_server]
    #   start = true
    should_start_local_server = bool(server_config.get("start", False))
    server_config = LocalServerConfig(**server_config) if (server_config and should_start_local_server) else None


def load_dataset(dataset_name: str, split: str, n_samples: int) -> ty.List[str]:
    """Loading the daastet.

    Returns: a list of queries.
    """
    logger.info("=== Running HuggingFace ArxivQA ===")
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("The 'datasets' package is required. Install via 'uv pip install datasets'.")
        return
    # end try

    logger.info("Loading dataset from HuggingFace...")
    dataset = load_dataset("MMInstruction/ArxivQA", split="train")
    
    # We use the 'question' column from the dataset as the query
    queries = dataset["question"][:n_samples]

    return queries

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
        
    job_config = InterfaceJobConfig(**config_dict)
    
    hpc_config = job_config.submitit_system

    start_local_server(job_config.local_server)

    os.environ["OPENAI_API_BASE"] = str(job_config.llm_client.openai_api_base)
    os.environ["OPENAI_MODEL"] = str(job_config.llm_client.model)
    
    output_dir = Path(hpc_config.log_folder)

    queries = load_dataset(dataset_name="MMInstruction/ArxivQA", split="train", n_samples=args.n_samples)
    
    logger.info("Invoking run_evaluation_pipeline...")
    results = run_evaluation_pipeline(
        queries=queries, 
        output_dir=output_dir, 
        hpc_config=hpc_config, 
        server_config=job_config.local_server,
        generation_parameters=job_config.llm_generation_params
    )
    # end try
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
