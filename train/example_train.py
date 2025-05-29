import os
import time
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

import torch
import torch.nn as nn
from tqdm import tqdm

from utils.logger import Logger
from utils.check_hardware import check_hardware
from utils.utils import load_yaml
from utils.train_helper import prepare_training, prepare_dataset

def train(config_path: Path|str, device: str|torch.device):
    """
    """
    logger = Logger()
    logger.info("Training started...")
    logger.info(f"Loading configuration from {config_path}")

    # Load configuration
    config = load_yaml(config_path)
    path_config = config["path_config"]
    dataset_config = config["dataset_config"]
    train_config = config["train_config"]

    checkpoint_path = Path(path_config.get("checkpoint_path", None))
    pretrained_path = Path(path_config.get("pretrained_path", None))
    output_dir = Path(path_config.get("output_dir", "outputs/train/experiment"))
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

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
        total_loss = 0.0
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
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                predictions, loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                if train_config.get("gradient_clip", None):
                    nn.utils.clip_grad_norm_(model.parameters(), train_config["gradient_clip"])
                optimizer.step()

                total_loss += losses.item()

                train_loader_tqdm.set_postfix({
                    "batch_loss": f"{losses.item():.4f}",
                    "avg_loss": f"{total_loss / (batch_idx + 1):.4f}",
                })
            except Exception as e:
                logger.error(f"Error during training step {batch_idx}: {e}")
                continue
        
        # End of epoch
        epoch_duration = time.time() - epoch_start_time
        epoch_loss = total_loss / len(train_loader)
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
            f"Average Loss: {epoch_loss:.4f}, "
            f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}"
        )
        tensorboard.write_scalar("Loss/train", epoch_loss, epoch + 1)
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
            val_metric = evaluate(model, val_loader, device)

            logger.info(f"Validation metric: {val_metric:.4f}")
            tensorboard.write_scalar("Loss/Val", val_metric, epoch + 1)

            # Early stopping check
            if early_stopper:
                if early_stopper(val_metric):
                    torch.save(checkpoint, output_dir / "best_model.pth")
                    logger.info(f"New best model at epoch {epoch + 1} saved!")
                if early_stopper.status():
                    logger.info("Early stopping triggered, stopping training.")
                    break
        
        tensorboard.flush()
        
    logger.info("Training completed successfully!")
    tensorboard.close()

if __name__ == "__main__":
    # Check hardware compatibility
    logger = Logger("train")
    device, is_cuda = check_hardware(verbose=False)
    config_path = Path("config/example_config.yaml")
    train(config_path, device=device)
