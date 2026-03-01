import os
import argparse
import submitit
import logging
import subprocess
from pathlib import Path
from dotenv import load_dotenv, find_dotenv


def run_script(
    env_path: Path,
    path_config: Path,
    n_samples: int,
    ):
    cmd = ["uv", "run", "python", 
    "interface/interface_job.py", 
    "--env_file", env_path.as_posix(),
    "--path_config", path_config.as_posix(),
    "--n_samples", str(n_samples)]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Launch HPC Job")
    parser.add_argument(
        "-e",
        "--env_file", 
        type=str, 
        default=".env_base",
        help="Path to the .env file")
    parser.add_argument(
        "-c",
        "--path_config", 
        type=str, 
        default="interface/config_local.toml",
        help="Path to the config file")
    parser.add_argument(
        "-n",
        "--n_samples", 
        type=int, 
        default=15,
        help="Number of samples to run from the HuggingFace dataset.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    load_dotenv(find_dotenv())

    slurm_partition = os.environ.get("SLURM_PARTITION")
    if not slurm_partition:
        logger.warning("SLURM_PARTITION is not set. Using 'dev' as fallback.")
        slurm_partition = "dev"
    
    log_folder = Path("submitit_logs")
    log_folder.mkdir(exist_ok=True)
    
    executor = submitit.AutoExecutor(folder=log_folder)
    
    executor.update_parameters(
        timeout_min=60,
        slurm_partition=slurm_partition,
        cpus_per_task=2,
        tasks_per_node=1,
        nodes=1,
    )
    
    logger.info(f"Submitting interface_job.py to partition '{slurm_partition}'")
    
    # submit the function
    job = executor.submit(run_script, args.env_file)
    logger.info(f"Submitted job with ID {job.job_id}")

if __name__ == "__main__":
    main()
