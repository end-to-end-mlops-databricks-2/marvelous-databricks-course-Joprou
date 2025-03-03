"""Configuration module."""

from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    """Project config."""

    experiment_name: Optional[str]
    experiment_name_fe: Optional[str]
    num_features: List[str]
    cat_features: List[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: Dict[str, Any]  # Dictionary to hold model-related parameters
    pipeline_id: Optional[str]

    @classmethod
    def from_yaml(cls, config_path: str, env: str = "dev"):
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # update config based on specified env
        env_catalog_config = {
            i: config_dict[env][i] for i in ["catalog_name", "schema_name", "pipeline_id"]
        }
        config_dict.update(env_catalog_config)

        return cls(**config_dict)


class Tags(BaseModel):
    """Model tags."""

    git_sha: str
    branch: str
    job_run_id: str
