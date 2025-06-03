import os
import time
import shutil
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.logger import Logger
from utils.check_hardware import check_hardware
from utils.utils import load_yaml, save_yaml
from utils.train_helper import prepare_training, prepare_dataset
from utils.evaluate_helper import evaluate

def train(config_path: Path|str, device: str|torch.device):
    """
    Main training function to train a model based on the provided configuration.
    Args:
        config_path (Path|str): Path to the YAML configuration file.
        device (str|torch.device): Device to run the training on (e.g., 'cuda' or 'cpu').
    """
    logger = Logger()
    logger.info("Training started...")
    logger.info(f"Loading configuration from {config_path}")

    # Load configuration
    config = load_yaml(config_path)
    path_config = config.get("path_config", {})
    dataset_config = config.get("dataset_config", {})
    train_config = config.get("train_config", {})
    evaluate_config = config.get("evaluate_config", {})

    checkpoint_path = Path(path_config["checkpoint_path"]) if path_config.get("checkpoint_path") else None
    pretrained_path = Path(path_config["pretrained_path"]) if path_config.get("pretrained_path") else None
    output_dir = Path(path_config.get("output_dir", "outputs/train/experiment"))
    
    # Delete output directory if it exists
    if output_dir.exists():
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    save_yaml(config, output_dir / config_path.name)
    logger.info("Configuration saved to output directory.")

    # Prepare training
    model, optimizer, scheduler, early_stopper, tensorboard = prepare_training(config, output_dir)
    model.to(device)
    
    start_epoch = 0
    total_epochs = train_config["num_epochs"]

    # Load from checkpoint or pretrained model
    if checkpoint_path:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 
        if scheduler: scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if early_stopper: early_stopper.load_state_dict(checkpoint["early_stopper_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resuming training from epoch {start_epoch}")
    elif pretrained_path:
        logger.info(f"Loading pretrained model from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded pretrained model weights")

    # Set up datasets and dataloaders
    logger.info("Preparing datasets")
    train_dataset, train_loader = prepare_dataset(dataset_config["train"])
    if "val" in dataset_config:
        val_dataset, val_loader = prepare_dataset(dataset_config["val"])
    else:
        val_dataset, val_loader = None, None
        logger.warning("Validation dataset not provided, skipping validation.")

    logger.info(f"Training dataset size: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation dataset size: {len(val_dataset)}")

    # Training loop
    for epoch in range(start_epoch, total_epochs):
        epoch_start_time = time.time()
        logger.info(f"Epoch {epoch + 1}/{total_epochs}")
        model.train()
        # Reset loss
        all_losses = {}

        train_loader_tqdm = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{total_epochs} - Training",
            unit="batch",
            dynamic_ncols=True,
        )
        # Tranining step
        for batch_idx, (images, targets) in enumerate(train_loader_tqdm):
            try:
                images = images.to(device)
                targets = targets.to(device)

                predictions, loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                if train_config.get("gradient_clip", None):
                    nn.utils.clip_grad_norm_(model.parameters(), train_config["gradient_clip"])
                optimizer.step()

                # Initialize all_losses if not done yet (first batch)
                if not all_losses:
                    all_losses = {k: 0.0 for k in loss_dict.keys()}
                    all_losses["total_loss"] = 0.0
                # Accumulate all_losses
                for k, v in loss_dict.items():
                    all_losses[k] += v.item()
                all_losses["total_loss"] += losses.item()

                train_loader_tqdm.set_postfix({
                    "batch_loss": f"{losses.item():.4f}",
                    "avg_loss": f"{all_losses['total_loss'] / (batch_idx + 1):.4f}",
                })
            except Exception as e:
                logger.error(f"Error during training step {batch_idx}: {e}")
                continue
        
        # End of epoch
        epoch_duration = time.time() - epoch_start_time
        epoch_all_losses = {key: all_losses[key] / len(train_loader) for key in all_losses}

        scheduler.step() if scheduler else None
        checkpoint = {
            "config": config,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "early_stopper_state_dict": early_stopper.state_dict() if early_stopper else None,
        }
        # Save checkpoint, write to TensorBoard
        logger.info(
            f"Epoch {epoch + 1} completed in {epoch_duration:.2f}s, "
            f"Average Loss: {epoch_all_losses['total_loss']:.4f}, "
            f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}"
        )
        logger.info(f"Individual Losses: {', '.join([f'{k}: {v:.4f}' for k, v in epoch_all_losses.items()])}")
        tensorboard.write_scalars("Loss/train", epoch_all_losses, epoch + 1)
        tensorboard.write_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch + 1)

        # Save checkpoint
        torch.save(checkpoint, output_dir / f"checkpoint_epoch_{epoch + 1}.pth")
        logger.info(f"Checkpoint saved for epoch {epoch + 1}")
        # Delete old checkpoints
        if epoch > 0:
            prev_checkpoint = output_dir / f"checkpoint_epoch_{epoch}.pth"
            if prev_checkpoint.exists():
                prev_checkpoint.unlink()
                logger.debug(f"Deleted previous checkpoint: {prev_checkpoint}")
        
        # Validation
        if val_loader:
            logger.info("Starting validation")
            
            # val_metric_dict must be a dictionary with metric names as keys and the first metric as the main metric
            val_loss, val_metric = evaluate(model, val_loader, device, metrics=evaluate_config["metrics"])

            logger.info(f"Validation metric: {val_metric[evaluate_config['priority_metric']]:.4f}")
            tensorboard.write_scalars("Loss/val", val_loss, epoch + 1)
            for metric_name, metric_value in val_metric.items():
                tensorboard.write_scalar(f"Val/{metric_name}", metric_value, epoch + 1)
            logger.info(f"Validation metric: {', '.join([f'{k}: {v:.4f}' for k, v in val_metric.items()])}")

            # Early stopping check
            if early_stopper:
                if early_stopper(val_metric):
                    torch.save(checkpoint, output_dir / "best_model.pth")
                    logger.info(f"New best model at epoch {epoch + 1} saved!")
                if early_stopper.status():
                    logger.info("Early stopping triggered, stopping training!")
                    break
        
        tensorboard.flush()
        
    logger.info("Training completed successfully!")
    tensorboard.close()

if __name__ == "__main__":
    logger = Logger("train")
    device, is_cuda = check_hardware(verbose=False)
    config_path = Path("configs/mnist_train_config.yaml")
    train(config_path, device=device)
