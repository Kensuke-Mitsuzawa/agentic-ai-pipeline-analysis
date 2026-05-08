import typing as ty
from pathlib import Path
from pydantic import BaseModel, Field, model_validator, field_validator, AliasChoices


class SlurmProfile(BaseModel):
    """Defines the hardware resources for a specific type of job."""
    partition: str = Field(description="The Slurm partition name (e.g., 'gpu_p13', 'cpu_short')")
    n_nodes_budget: int = Field(default=1, description="The total number of nodes given. This pipeline automatically splits under this given condition.")
    n_tasks_per_node: int = Field(default=1)
    n_cpus_per_task: int = Field(default=4)
    # Backward-compatible fields used by tests/examples (even if not consumed by submitit kwargs).
    n_gpus_per_task: int = Field(default=0, description="Number of GPUs per task (optional for local/CPU).")
    mem_per_gpu: ty.Optional[str] = Field(default=None, description="Memory per GPU (optional).")
    time: str = Field(default="02:00:00", description="Walltime format HH:MM:SS")
    gres: ty.Optional[str] = Field(default=None, description="Generic resources, e.g., 'gpu:1'")


class EstimatorSlurmMap(BaseModel):
    """Maps a specific LLM/Estimator to one or multiple Slurm Profiles."""
    # CHANGED: Now accepts a list of profiles for heterogeneous dispatch
    profile_names: ty.Union[str, ty.List[str]] = Field(description="Must match keys in SlurmSystemConfig.profiles")
    overrides: ty.Optional[ty.Dict[str, ty.Any]] = Field(default=None, description="Specific overrides for this model")


class SlurmSystemConfig(BaseModel):
    """Root configuration for HPC Environment."""
    log_folder: Path = Field(default=Path("slurm_logs"), description="Directory to store submitit logs and worker logs")
    
    profiles: ty.Dict[str, SlurmProfile] = Field(description="Dictionary of available hardware profiles")
    
    estimator_dispatch_map: ty.Dict[str, EstimatorSlurmMap] = Field(
        default_factory=dict, 
        description="Configuration for dispatching specific estimators"
    )

    # Backward-compatible with earlier config shape used in tests/examples.
    # - Older code used `default_profile="local"` (singular string)
    # - Current code uses `default_profiles=["local"]` (list[str])
    default_profiles: ty.List[str] = Field(
        description="Fallback profile name(s) if model is not in map",
        default_factory=list,
        validation_alias=AliasChoices("default_profiles", "default_profile"),
    )

    is_delete_worker_output: bool = Field(description="True then the pipeline deletes the worker's temporary outcome.", default=True)
    is_download_resources_prepost: bool = Field(default=True, description="Downloading the model resources before launching estimators' worker nodes.")
    is_download_resources_at_worker: bool = Field(default=False, description="if False, the worker does not download from the HuggingFace hub.")
    max_io_workers: int = Field(default=1, description="The num. of max workers of writing outcome to a disk.")
    io_queue_size: int = Field(default=2, description="The num. of max LLM-outcome holding on a disk.")

    @field_validator("default_profiles", mode="before")
    @classmethod
    def coerce_default_profiles(cls, v: ty.Any) -> ty.List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    @model_validator(mode='after')
    def validate_profiles_exist(self) -> 'SlurmSystemConfig':
        """Ensure all mapped profiles actually exist."""
        # Normalize to list for validation
        defs = self.default_profiles
        if not defs:
            raise ValueError("default_profiles must not be empty.")
        for d in defs:
            if d not in self.profiles:
                raise ValueError(f"Default profile '{d}' not found in profiles.")
        
        for model, map_cfg in self.estimator_dispatch_map.items():
            p_names = [map_cfg.profile_names] if isinstance(map_cfg.profile_names, str) else map_cfg.profile_names
            for p in p_names:
                if p not in self.profiles:
                    raise ValueError(f"Model {model} maps to non-existent profile '{p}'")
        return self