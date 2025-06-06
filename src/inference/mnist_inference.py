import os
import shutil
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn

from core.logger import Logger
from core.check_hardware import check_hardware
from core.utils import load_yaml, save_yaml
from utils.train_helper import get_model, prepare_dataset
from utils.visualize_helper import plot_image, denormalize

@torch.no_grad()
def inference_all(model, dataloader, output_dir, device, save: bool=False):
    """
    Function to handle inference for all data in the dataset.
    This function should iterate through the dataloader and perform inference on each batch.
    """
    logger = Logger()

    if save:
        image_dir = Path(output_dir) / "predictions"
        os.makedirs(image_dir, exist_ok=True)
        logger.info(f"Saving predictions to {image_dir}")

    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            predictions, _ = model(images, targets)
            probs = torch.exp(predictions)  
            max_probs, predicted_classes = torch.max(probs, dim=1)

            correct += (predicted_classes == targets).sum().item()
            total += targets.size(0)

            # Fix to add path or id
            for i, image in enumerate(images):
                label = targets[i].cpu().item()
                pred = f"{predicted_classes[i].cpu().item()}_{max_probs[i].cpu().item()}"
                logger.info(f"[Processing image] Label: {label}, Predicted: {pred}")
                if save:
                    image = denormalize(image, mean=(0.1307,), std=(0.3081,)).cpu()
                    save_path = image_dir / f"image_{i}_pred_{pred}.png"
                    # image = image.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
                    plot_image(image=image, 
                                label=label,
                                pred=pred,
                                save_path=save_path)
    
    logger.info(f"accuracy: {correct / total:.4f} ({correct}/{total})")

def inference_single():
    pass

def inference_random():
    pass

def inference_function(model: nn.Module, data: tuple, output_dir: str|Path, device: str|torch.device, config: dict):
    """
    Function to handle inference based on the mode and parameters provided.
    Args:
        mode (str): The mode of inference ('all' or 'single').
        params (dict): Parameters for the inference function.
    Returns:
        None
    """
    dataset, dataloader = data
    mode = config.get("mode", "all")
    save = config.get("save", False)
    params = config.get("params", {})
    #TODO: Add more parameters as needed
    
    if mode == "all":
        inference_all(model, dataloader, output_dir, device, **params)
    elif mode == "single":
        inference_single(model, dataset, output_dir, device, **params)
    elif mode == "random":
        inference_random(model, dataset, output_dir, device, **params)


def inference(config_path: str|Path, device: str|torch.device):
    """
    Main inference function to run a model on a dataset based on the provided configuration.
    Args:
        config_path (str|Path): Path to the YAML configuration file.
        device (str|torch.device): Device to run the inference on (e.g., 'cuda' or 'cpu').
    """
    logger = Logger()
    logger.info(f"{'=' * 20} Setting up inference {'=' * 20}")
    logger.info(f"Loading configuration from {config_path}")

    # Load configuration
    config = load_yaml(config_path)
    path_config = config.get("path_config", {})
    model_config = config.get("model_config", {})
    dataset_config = config.get("dataset_config", {})
    inference_config = config.get("inference_config", {})

    model_checkpoint = Path(path_config["checkpoint_path"]) if path_config.get("checkpoint_path") else None
    output_dir = Path(path_config.get("output_dir", "outputs/inference/experiment"))
    if output_dir.exists():
        shutil.rmtree(output_dir)   
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    save_yaml(config, output_dir / config_path.name)
    logger.info(f"Configuration saved to output directory")

    logger.info(f"Loading model from {model_checkpoint}")
    model = get_model(model_config)
    model.to(device)
    model.load_state_dict(torch.load(model_checkpoint, map_location=device)["model_state_dict"])
    logger.info(f"Model loaded successfully")

    # Prepare dataset
    logger.info("Preparing dataset")
    inference_dataset, inference_loader = prepare_dataset(dataset_config["inference"])
    logger.info(f"Inference dataset size: {len(inference_dataset)}")
    
    logger.info(f"{'=' * 20} Starting inference {'=' * 20}")
    inference_start_time = time.time()

    inference_function(model, 
                       (inference_dataset, inference_loader), 
                       output_dir, 
                       device,
                       inference_config)

    inference_end_time = time.time()
    logger.info(f"🕒 Inference completed in {inference_end_time - inference_start_time:.2f} seconds")
    logger.info(f"{'=' * 20} Inference finished {'=' * 20}")


if __name__ == "__main__":
    logger = Logger("inference")
    device, is_cuda = check_hardware(verbose=False)
    config_path = Path("src/configs/mnist/mnist_inference_config.yaml")

    try:
        inference(config_path, device=device)
    except Exception as e:
        logger.error(f"An error occurred during inference: {e}")
        logger.error(traceback.format_exc())