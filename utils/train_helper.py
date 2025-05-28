from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau

from utils.utils import EarlyStopping, TensorBoard
from models.custom_model import CustomModel

def load_model(model_path: Path|str, device: torch.device):
    """
    Load a pre-trained model from the specified path.
    Args:
        model_path (Path|str): Path to the model file.
        device (torch.device): Device to load the model onto (e.g., 'cpu' or 'cuda').
    Returns:
        model (torch.nn.Module): The loaded model.
    """
    config = torch.load(model_path, map_location=device)["config"]
    model_config = config["model_config"]
    model = CustomModel(config=model_config)
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    return model

def prepare_training(config: dict, output_dir: Path|str):
    """
    Prepare the training environment, including model, optimizer, scheduler, and early stopping.
    Args:
        config (dict): Configuration dictionary containing model, optimizer, and training parameters.
        output_dir (Path|str): Directory where training outputs will be saved.
    Returns:
        tensorboard (TensorBoard): TensorBoard instance for logging.
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        early_stopper (EarlyStopping): Early stopping instance to monitor training progress.
    """
    output_dir = Path(output_dir)

    model_config = config["model_config"]
    train_config = config["train_config"]
    tensorboard_path = output_dir / "tensorboard"

    tensorboard = TensorBoard(log_dir=tensorboard_path)
    model = CustomModel(config=model_config)
    optimizer = get_optimizer(model, train_config["optimizer"])
    scheduler = get_scheduler(optimizer, train_config["scheduler"])
    early_stopper = EarlyStopping(**train_config["early_stopping"])
    return tensorboard, model, optimizer, scheduler, early_stopper

def get_optimizer(model: nn.Module, optimizer_config: dict):
    """
    Get the optimizer based on the provided configuration.
    Args:
        model (torch.nn.Module): The model for which the optimizer is to be created.
        optimizer_config (dict): Configuration dictionary for the optimizer, including type and parameters.
    Returns:
        optimizer (torch.optim.Optimizer): The configured optimizer.
    """
    optimizers = {
        "SGD": optim.SGD,
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "RMSprop": optim.RMSprop,
        # others can be added here
    }

    if optimizer_config["type"] not in optimizers:
        raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}. Available options: {list(optimizers.keys())}")

    return optimizers[optimizer_config["type"]](model.parameters(), **optimizer_config["params"])

def get_scheduler(optimizer: optim.Optimizer, scheduler_config: dict):
    """
    Get the learning rate scheduler based on the provided configuration.
    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which the scheduler is to be created.
        scheduler_config (dict): Configuration dictionary for the scheduler, including type and parameters.
    Returns:
        scheduler (torch.optim.lr_scheduler._LRScheduler): The configured learning rate scheduler.
    """
    schedulers = {
        "StepLR": StepLR,
        "MultiStepLR": MultiStepLR,
        "ExponentialLR": ExponentialLR,
        "CosineAnnealingLR": CosineAnnealingLR,
        "ReduceLROnPlateau": ReduceLROnPlateau,
        # others can be added here
    }

    if scheduler_config["type"] not in schedulers:
        raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}. Available options: {list(schedulers.keys())}")

    return schedulers[scheduler_config["type"]](optimizer, **scheduler_config["params"])