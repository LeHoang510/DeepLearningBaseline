import os
import time
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import torch

from utils.logger import Logger
from utils.check_hardware import check_hardware

def test(config_path: Path|str, device: str|torch.device):
    pass

if __name__ == "__main__":
    logger = Logger("test")
    device, is_cuda = check_hardware(verbose=False)
    config_path = Path("config/example_config.yaml")
    test(config_path, device=device)