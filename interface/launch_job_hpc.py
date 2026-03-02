import os
import argparse
import submitit
import logging
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv


current_interpreter = sys.executable
print(f"The interpreter is: {current_interpreter}")

def main():
    parser = argparse.ArgumentParser(description="Launch HPC Job")
    parser.add_argument(
        "-e",
        "--env_file", 
        type=str, 
        default=".env_base",
        help="Path to the .env file",
        required=True)
    parser.add_argument(
        "-c",
        "--path_config", 
        type=str, 
        default="interface/config_local.toml",
        help="Path to the config file",
        required=True)
    parser.add_argument(
        "-n",
        "--n_samples", 
        type=int, 
        default=15,
        help="Number of samples to run from the HuggingFace dataset.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    assert Path(args.env_file).exists()
    assert Path(args.path_config).exists()

    load_dotenv(args.env_file)

    slurm_partition = os.environ.get("SLURM_PARTITION")
    if not slurm_partition:
        logger.warning("SLURM_PARTITION is not set. Using 'dev' as fallback.")
        slurm_partition = "dev"
    # end

    slurm_timeout_min = os.environ.get("SLURM_TIMEOUT_MIN", 270)
    _time = os.environ.get('SLURM_TIME', '20:00:00')

    slurm_log_folder = Path(os.environ.get("SUBMITIT_LOGS", "submitit_logs"))
    slurm_log_folder.mkdir(exist_ok=True)
    logger.info(f"Log files are at {slurm_log_folder}")

    lib_path = Path(os.environ.get("LIB_PATH", ""))
    
    executor = submitit.AutoExecutor(folder=slurm_log_folder.as_posix())
    
    executor.update_parameters(
        timeout_min=int(slurm_timeout_min),
        slurm_partition=slurm_partition,
        cpus_per_task=2,
        tasks_per_node=1,
        nodes=1,
    )
    
    logger.info(f"Submitting interface_job.py to partition '{slurm_partition}'")
    
    script_path = Path(__file__).resolve().parent
    target_script_path = (script_path / "interface_job.py").as_posix()

    # Build command to submit the ORCHESTRATOR
    cmd = [
        "sbatch",
        f"--job-name=ORCH_MASTER",
        f"--partition={slurm_partition}",
        f"--cpus-per-task={1}",
        f"--time={_time}",
        f"--output={slurm_log_folder}/orchestrator_%j.out", # Absolute path
        f"--error={slurm_log_folder}/orchestrator_%j.err",  # Separate error log
        # We pass the environment variable forward so the Master knows which config to use
        f"--export=ALL,LD_LIBRARY_PATH={lib_path}:$LD_LIBRARY_PATH",
        "--wrap", f"source /etc/profile.d/modules.sh && {current_interpreter} {target_script_path}"
    ]

    print(f"🚀 Submitting Orchestrator to {slurm_partition}...")
    subprocess.run(cmd, check=True)
    print("✅ Job submitted. You can now log out.")

if __name__ == "__main__":
    main()
