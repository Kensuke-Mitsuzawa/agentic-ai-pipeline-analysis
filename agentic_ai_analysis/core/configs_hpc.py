import typing as ty
from pathlib import Path
from pydantic import BaseModel, Field, model_validator


class SlurmProfile(BaseModel):
    """Defines the hardware resources for a specific type of job."""
    partition: str = Field(description="The Slurm partition name (e.g., 'gpu_p13', 'cpu_short')")
    n_nodes_budget: int = Field(default=1, description="The total number of nodes given. This pipeline automatically splits under this given condition.")
    n_tasks_per_node: int = Field(default=1)
    n_cpus_per_task: int = Field(default=4)
    time: str = Field(default="02:00:00", description="Walltime format HH:MM:SS")
    gres: ty.Optional[str] = Field(default=None, description="Generic resources, e.g., 'gpu:1'")


class EstimatorSlurmMap(BaseModel):
    """Maps a specific LLM/Estimator to a Slurm Profile."""
    profile_name: str = Field(description="Must match a key in SlurmSystemConfig.profiles")
    overrides: ty.Optional[ty.Dict[str, ty.Any]] = Field(default=None, description="Specific overrides for this model (e.g., longer time)")


class SlurmSystemConfig(BaseModel):
    """Root configuration for HPC Environment."""
    log_folder: Path = Field(default=Path("slurm_logs"), description="Directory to store submitit logs and worker logs")
    
    # Define available hardware profiles
    profiles: ty.Dict[str, SlurmProfile] = Field(description="Dictionary of available hardware profiles")
    
    # Map estimators to profiles
    # Key: estimator name (pet-name of algorithm config), Value: Mapping config
    estimator_dispatch_map: ty.Dict[str, EstimatorSlurmMap] = Field(
        default_factory=dict, 
        description="Configuration for dispatching specific estimators"
    )

    # Default profile to use if a model is not explicitly mapped
    default_profile: str = Field(description="Fallback profile name if model is not in map")

    is_delete_worker_output: bool = Field(description="True then the pipeline deletes the worker's temporary outcome.", default=True)

    # configurations about chunking at distributing jobs into worker functions (MAP).
    # n_max_inference_in_memory: int = Field(default=5, description="The max. number of inference tasks executed on GPU memory, held on RAM.")
    is_download_resources_prepost: bool = Field(default=True, description="Downloading the model resources before launching estimators' worker nodes.")
    is_download_resources_at_worker: bool = Field(default=False, description="if False, the worker does not download from the HuggingFace hub. Recommended to set False when the workers has no internet access.")

    max_io_workers: int = Field(default=1, description="The num. of max workers of writing outcome to a disk.")
    io_queue_size: int = Field(default=2, description="The num. of max LLM-outcome holding on a disk.")

    @model_validator(mode='after')
    def validate_profiles_exist(self) -> 'SlurmSystemConfig':
        """Ensure all mapped profiles actually exist."""
        if self.default_profile not in self.profiles:
            raise ValueError(f"Default profile '{self.default_profile}' not found in profiles.")
        # end
        for model, map_cfg in self.estimator_dispatch_map.items():
            if map_cfg.profile_name not in self.profiles:
                raise ValueError(f"Model {model} maps to non-existent profile '{map_cfg.profile_name}'")
        # end
        
        return self