import os
import argparse
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
    parser.add_argument(
        "--env", 
        type=str, 
        default="standard",
        choices=['standard', 'jean-zay']
    )

    args = parser.parse_args()

    _exec_env = args.env

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    assert Path(args.env_file).exists()
    assert Path(args.path_config).exists()

    load_dotenv(args.env_file, override=True)

    slurm_partition = os.environ.get("SLURM_PARTITION")
    if not slurm_partition:
        logger.warning("SLURM_PARTITION is not set. Using 'dev' as fallback.")
        slurm_partition = "dev"
    # end

    slurm_timeout_min = os.environ.get("SLURM_TIMEOUT_MIN", 270)
    _time = os.environ.get('SLURM_TIME', '20:00:00')

    _mem_gb = os.environ.get('SLURM_MEM_GB', '16G')

    slurm_log_folder = Path(os.environ.get("SUBMITIT_LOGS", "submitit_logs"))
    slurm_log_folder.mkdir(exist_ok=True)
    logger.info(f"Log files are at {slurm_log_folder}")

    lib_path = os.environ.get("LIB_PATH", None)
        
    logger.info(f"Submitting interface_job.py to partition '{slurm_partition}'")
    
    script_path = Path(__file__).resolve().parent
    target_script_path = (script_path / "interface_job.py").as_posix()

    python_args = f"--n_samples {args.n_samples} --env_file {args.env_file} --path_config {args.path_config}"
    # Build command to submit the ORCHESTRATOR
    cmd = ["sbatch", f"--job-name=ORCH_MASTER"]

    if _exec_env == "standard":
        cmd.append(f"--mem={_mem_gb}")
    # end if

    cmd.append(f"--partition={slurm_partition}")
    cmd.append(f"--cpus-per-task=1")
    cmd.append(f"--time={_time}")
    cmd.append(f"--output={slurm_log_folder}/orchestrator_%j.out")
    cmd.append(f"--error={slurm_log_folder}/orchestrator_%j.err")
    
    if lib_path is not None:
        cmd.append(f"--export=ALL,LD_LIBRARY_PATH={lib_path}:$LD_LIBRARY_PATH")
    # end if
    
    cmd.append("--wrap")
    cmd.append(f"source /etc/profile.d/modules.sh && {current_interpreter} {target_script_path} {python_args}")
    
    print(f"🚀 Submitting Orchestrator to {slurm_partition}...")
    subprocess.run(cmd, check=True)
    print("✅ Job submitted. You can now log out.")

if __name__ == "__main__":
    main()
