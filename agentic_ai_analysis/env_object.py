from pydantic import BaseModel, SecretStr, Field
from typing import Optional
import os
from dotenv import load_dotenv

class EnvConfig(BaseModel):
    """Pydantic model for the environment variables defined in .env_base.
    
    Attributes:
        slurm_partition: The SLURM partition to use (default: dev).
        hf_home: Optional custom model cache location.
        submitit_logs: Optional path for submitit logs.
        langfuse_enabled: Whether to enable Langfuse tracing.
        langfuse_public_key: Public key for Langfuse (Masked).
        langfuse_secret_key: Secret key for Langfuse (Masked).
        langfuse_host: The Langfuse host URL.
    """
    slurm_partition: str = Field(default="dev", alias="SLURM_PARTITION")
    hf_home: Optional[str] = Field(default=None, alias="HF_HOME")
    submitit_logs: Optional[str] = Field(default=None, alias="SUBMITIT_LOGS")
    
    langfuse_enabled: bool = Field(default=False, alias="LANGFUSE_ENABLED")
    # Using SecretStr to mask sensitive keys in logs and repr()
    langfuse_public_key: Optional[SecretStr] = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[SecretStr] = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")

    model_config = {
        "populate_by_name": True,
        "frozen": True  # Configuration should generally be immutable after loading
    }

    @classmethod
    def load(cls, env_file: Optional[str] = None) -> "EnvConfig":
        """Loads configuration from environment variables and an optional .env file."""
        if env_file:
            load_dotenv(env_file, override=True)
        else:
            # Try to load default .env if it exists
            load_dotenv()
            
        return cls(
            SLURM_PARTITION=os.getenv("SLURM_PARTITION", "dev"),
            HF_HOME=os.getenv("HF_HOME"),
            SUBMITIT_LOGS=os.getenv("SUBMITIT_LOGS"),
            LANGFUSE_ENABLED=os.getenv("LANGFUSE_ENABLED", "false").lower() in ("true", "1", "t", "yes", "on"),
            LANGFUSE_PUBLIC_KEY=os.getenv("LANGFUSE_PUBLIC_KEY"),
            LANGFUSE_SECRET_KEY=os.getenv("LANGFUSE_SECRET_KEY"),
            LANGFUSE_HOST=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        ).validate_connection()

    def validate_connection(self) -> "EnvConfig":
        """Verifies connection to Langfuse if enabled."""
        if not self.langfuse_enabled:
            return self
            
        try:
            from langfuse import Langfuse
            if not self.langfuse_public_key or not self.langfuse_secret_key:
                raise ValueError("Langfuse is enabled but keys are missing in environment.")
                
            client = Langfuse(
                public_key=self.langfuse_public_key.get_secret_value(),
                secret_key=self.langfuse_secret_key.get_secret_value(),
                host=self.langfuse_host
            )
            
            # auth_check() is the official way to verify connection and keys
            if client.auth_check():
                return self
            else:
                raise ValueError(f"Authentication failed for Langfuse at {self.langfuse_host}")
        except ImportError:
            # If langfuse is not installed, we can't verify, but we shouldn't break 
            # unless the user explicitly wants strict enforcement.
            return self
        except Exception as e:
            # We raise a warning but return self to allow the app to potentially 
            # continue if tracing is not mission-critical.
            import logging
            logging.getLogger().warning(f"Langfuse connection check failed: {e}")
            return self

    def to_langfuse_dict(self) -> dict:
        """Returns a dictionary suitable for initializing Langfuse client."""
        return {
            "public_key": self.langfuse_public_key.get_secret_value() if self.langfuse_public_key else None,
            "secret_key": self.langfuse_secret_key.get_secret_value() if self.langfuse_secret_key else None,
            "host": self.langfuse_host
        }
