from pathlib import Path

from core.utils import load_yaml
from core.logger import Logger

def train_config_validator(config_path: str | Path) -> None:
    """
    Validate the training configuration file.
    
    Args:
        config_path (str | Path): Path to the YAML configuration file.
    
    Raises:
        ValueError: If the configuration is invalid.
    """
    logger = Logger()
    config = load_yaml(config_path)
    
    
    