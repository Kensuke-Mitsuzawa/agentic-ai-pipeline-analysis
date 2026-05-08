"""CLI entrypoint for running the pipeline with a TOML config.

This script loads a config file (e.g. `interface/app_configs/config_local.toml`),
optionally launches a local OpenAI-compatible LLM server, and then runs the
pipeline against a dataset.
"""

import os
import argparse
import typing as ty
import logging
import tomllib
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Early loading of dotenv so that env vars like HF_HOME are set before HF libraries are imported
_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument('-e', '--env_file', type=str, default='.env')
_early_args, _ = _early_parser.parse_known_args()
load_dotenv(_early_args.env_file, override=True)
print(f"loading env file from {_early_args.env_file}")
from agentic_ai_analysis.env_object import EnvConfig
env_config = EnvConfig.load(_early_args.env_file)

from agentic_ai_analysis.main import run_evaluation_pipeline
from agentic_ai_analysis.core.configs_hpc import SubmititSystemConfig
from agentic_ai_analysis.core.local_server import LocalServerConfig, LLMClientConfig, start_local_server as _start_local_server


logger = logging.getLogger()


class LLMOpsEvaluationConfig(BaseModel):
    sampling_rate_evaluation: float = Field(default=1.0, description="Sampling rate for evaluation", ge=0.0, le=1.0)
    compute_cka: bool = Field(default=False, description="Compute CKA metric")
    


class InterfaceJobConfig(BaseModel):
    submitit_system: SubmititSystemConfig
    llm_client: LLMClientConfig
    local_server: ty.Optional[LocalServerConfig] = Field(default=None)
    llm_ops_evaluation: LLMOpsEvaluationConfig = Field(default=LLMOpsEvaluationConfig())


def maybe_start_local_server(server_config: ty.Optional[LocalServerConfig]) -> bool:
    """
    Returns True if a local server was started.
    By default we assume the LLM server is already running (e.g., port-forward to remote).
    """
    if server_config is None:
        return False
    should_start = bool(getattr(server_config, "start", False))
    if not should_start:
        return False
    _start_local_server(server_config)
    return True


def load_dataset(dataset_name: str, split: str, n_samples: int) -> ty.List[str]:
    """Loading the daastet.

    Returns: a list of queries.
    """
    logger.info("=== Running HuggingFace ArxivQA ===")
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        logger.error("The 'datasets' package is required. Install via 'uv pip install datasets'.")
        return []
    # end try

    logger.info("Loading dataset from HuggingFace...")
    dataset = hf_load_dataset(dataset_name, split=split)
    
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
        
    logger.info("Loading configuration from %s", config_path)
    with open(config_path, "rb") as f:
        config_dict = tomllib.load(f)
        
    job_config = InterfaceJobConfig.model_validate(config_dict)
    
    hpc_config = job_config.submitit_system

    started_local = maybe_start_local_server(job_config.local_server)

    os.environ["OPENAI_API_BASE"] = str(job_config.llm_client.openai_api_base)
    os.environ["OPENAI_MODEL"] = str(job_config.llm_client.model)
    
    output_dir = Path(hpc_config.log_folder)

    queries = load_dataset(dataset_name="MMInstruction/ArxivQA", split="train", n_samples=args.n_samples)
    
    logger.info("Invoking run_evaluation_pipeline...")
    run_evaluation_pipeline(
        queries=queries, 
        output_dir=output_dir, 
        hpc_config=hpc_config, 
        server_config=job_config.local_server if started_local else None,
        evaluation_sampling_rate=job_config.llm_ops_evaluation.sampling_rate_evaluation,
        compute_cka=job_config.llm_ops_evaluation.compute_cka,
    )
    # end try
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
