#!/usr/bin/env python3
import os
import json
from pathlib import Path
from typing import Dict, Any, List

# We use tomli_w if available, otherwise fallback to a simple writer
try:
    import tomli_w
    HAS_TOMLI_W = True
except ImportError:
    HAS_TOMLI_W = False

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(text: str):
    print(f"\n{'='*60}")
    print(f" {text}")
    print(f"{'='*60}\n")

def ask(prompt: str, default: Any = None) -> str:
    default_str = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{default_str}: ").strip()
    return val if val else str(default)

def ask_bool(prompt: str, default: bool = False) -> bool:
    default_str = "Y/n" if default else "y/N"
    val = input(f"{prompt} ({default_str}): ").strip().lower()
    if not val:
        return default
    return val in ('y', 'yes', 'true', '1')

def ask_int(prompt: str, default: int) -> int:
    while True:
        try:
            return int(ask(prompt, default))
        except ValueError:
            print("Error: Please enter a valid integer.")

def ask_float(prompt: str, default: float) -> float:
    while True:
        try:
            return float(ask(prompt, default))
        except ValueError:
            print("Error: Please enter a valid float.")

def dump_toml(data: dict) -> str:
    """Fallback TOML writer if tomli_w is missing."""
    if HAS_TOMLI_W:
        return tomli_w.dumps(data)
    
    # Simple manual dump for basic types
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"\n[{key}]")
            for subkey, subval in value.items():
                if isinstance(subval, dict):
                    lines.append(f"\n[{key}.{subkey}]")
                    for ssubkey, ssval in subval.items():
                         v = json.dumps(ssval)
                         lines.append(f"{ssubkey} = {v}")
                else:
                    v = json.dumps(subval)
                    lines.append(f"{subkey} = {v}")
        else:
            v = json.dumps(value)
            lines.append(f"{key} = {v}")
    return "\n".join(lines)

def run_wizard():
    clear_screen()
    print_header("Agentic AI Pipeline - Configuration Wizard")
    
    config = {}

    # 1. Submitit System Config
    print_header("1. Submitit System Configuration")
    submitit = {}
    submitit["log_folder"] = ask("Log folder path", "submitit_logs")
    
    profiles = {}
    print("\nDefine at least one hardware profile (e.g., 'local', 'h100').")
    while True:
        p_name = ask("Profile name (or press Enter to finish)", "local" if not profiles else "")
        if not p_name:
            if not profiles:
                print("Error: You must define at least one profile.")
                continue
            break
        
        p_data = {}
        p_data["partition"] = ask(f"[{p_name}] Slurm partition", "dev")
        p_data["n_nodes_budget"] = ask_int(f"[{p_name}] Node budget", 1)
        p_data["n_tasks_per_node"] = ask_int(f"[{p_name}] Tasks per node", 1)
        p_data["n_cpus_per_task"] = ask_int(f"[{p_name}] CPUs per task", 4)
        p_data["time"] = ask(f"[{p_name}] Walltime (HH:MM:SS)", "02:00:00")
        
        has_gpu = ask_bool(f"[{p_name}] Does this profile use GPUs?", False)
        if has_gpu:
            p_data["gres"] = ask(f"[{p_name}] GRES string", "gpu:1")
            p_data["n_gpus_per_task"] = ask_int(f"[{p_name}] GPUs per task", 1)
        
        profiles[p_name] = p_data
        if not ask_bool("Add another profile?", False):
            break
            
    submitit["profiles"] = profiles
    submitit["default_profiles"] = [ask("Default profile name", list(profiles.keys())[0])]
    submitit["is_delete_worker_output"] = ask_bool("Delete worker output after completion?", True)
    config["submitit_system"] = submitit

    # 2. LLM Client Config
    print_header("2. LLM Client Configuration")
    llm_client = {}
    llm_client["openai_api_base"] = ask("OpenAI API Base URL", "http://localhost:8000/v1/")
    
    print("\nAcceptable Models: meta-llama/Meta-Llama-3.1-8B-Instruct, Qwen/Qwen2.5-7B-Instruct, etc.")
    llm_client["model"] = ask("Model name", "Qwen/Qwen2.5-7B-Instruct")
    config["llm_client"] = llm_client

    # 3. Local Server Config
    print_header("3. Local Server Configuration")
    local_server = {}
    start_server = ask_bool("Should the pipeline launch a local server?", False)
    if start_server:
        local_server["start"] = True
        local_server["model_id"] = ask("Model ID to load", llm_client["model"])
        local_server["host"] = ask("Server host", "127.0.0.1")
        local_server["port"] = ask_int("Server port", 8000)
        
        if ask_bool("Apply default 4-bit quantization?", True):
            local_server["quantization_config_dict"] = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_quant_type": "nf4"
            }
        config["local_server"] = local_server

    # 4. LLM Generation Parameters
    print_header("4. LLM Generation Parameters")
    gen_params = {}
    gen_params["temperature"] = ask_float("Temperature", 0.7)
    gen_params["max_tokens"] = ask_int("Max tokens", 999)
    gen_params["top_p"] = ask_float("Top P", 0.9)
    config["llm_generation_params"] = gen_params

    # 5. LLMOps Evaluation Config
    print_header("5. Evaluation Configuration")
    eval_cfg = {}
    eval_cfg["sampling_rate_evaluation"] = ask_float("Sampling rate (0.0 to 1.0)", 1.0)
    eval_cfg["compute_cka"] = ask_bool("Compute CKA metric?", False)
    config["llm_ops_evaluation"] = eval_cfg

    # Save to file
    print_header("Finalizing")
    output_name = ask("Save config as", "config_custom.toml")
    if not output_name.endswith(".toml"):
        output_name += ".toml"
    
    toml_str = dump_toml(config)
    
    with open(output_name, "w") as f:
        f.write(toml_str)
        
    print(f"\nSuccess! Configuration saved to: {output_name}")
    print(f"You can now run: python interface/interface_job.py --path_config {output_name}")

if __name__ == "__main__":
    try:
        run_wizard()
    except KeyboardInterrupt:
        print("\n\nWizard cancelled.")
