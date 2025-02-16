from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    experiment_name: Optional[str]
    experiment_name_fe: Optional[str]
    num_features: List[str]
    cat_features: List[str]
    target: str
    drop_cols: List[str]
    catalog_name: str
    schema_name: str
    parameters: Dict[str, Any]  # Dictionary to hold model-related parameters

    @classmethod
    def from_yaml(cls, config_path: str):
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class Tags(BaseModel):
    git_sha: str
    branch: str
