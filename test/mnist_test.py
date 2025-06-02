import os
import shutil
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import torch
from utils.logger import Logger
from utils.check_hardware import check_hardware
from utils.utils import load_yaml
from utils.train_helper import get_model, prepare_dataset
from utils.evaluate_helper import evaluate


def test(config_path: Path|str, device: str|torch.device):
    logger = Logger()
    logger.info("Testing started...")
    logger.info(f"Loading configuration from {config_path}")

    # Load configuration
    config = load_yaml(config_path)
    path_config = config.get("path_config", {})
    model_config = config.get("model_config", {})
    dataset_config = config.get("dataset_config", {})
    evaluate_config = config.get("evaluate_config", {})

    model_checkpoint = Path(path_config["checkpoint_path"])
    output_dir = Path(path_config.get("output_dir", "outputs/test/experiment"))
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    logger.info("Loading model...")
    model = get_model(model_config)
    model.to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device)["model_state_dict"])
    logger.info("Model loaded successfully!")

    # Prepare dataset
    logger.info("Preparing dataset...")
    test_dataset, test_loader = prepare_dataset(dataset_config["test"])
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Evaluate the model
    test_loss, test_metric = evaluate(model, test_loader, device, evaluate_params=evaluate_config)

if __name__ == "__main__":
    logger = Logger("test")
    device, is_cuda = check_hardware(verbose=False)
    config_path = Path("config/mnist_config.yaml")
    test(config_path, device=device)