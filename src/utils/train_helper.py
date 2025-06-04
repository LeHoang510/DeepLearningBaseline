from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from utils.utils import EarlyStopping, TensorBoard

from models.example_model import ExampleModel
from models.MNIST.mnist_model import MnistModel
from datasets.example_dataset import ExampleDataset
from datasets.mnist_dataset import MnistDataset

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
    model = get_model(model_config)
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    return model

def prepare_training(config: dict, output_dir: Path|str):
    """
    Prepare the training environment, including model, optimizer, scheduler, and early stopping.
    Args:
        config (dict): Configuration dictionary containing model, optimizer, and training parameters.
        output_dir (Path|str): Directory where training outputs will be saved.
    Returns:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        early_stopper (EarlyStopping): Early stopping instance to monitor training progress.
        tensorboard (TensorBoard): TensorBoard instance for logging.
    """
    output_dir = Path(output_dir)

    model_config = config.get("model_config", {})
    train_config = config.get("train_config", {})
    tensorboard_path = output_dir / "tensorboard"

    tensorboard = TensorBoard(log_dir=tensorboard_path)
    model = get_model(model_config)
    optimizer = get_optimizer(model, train_config.get("optimizer", {}))
    scheduler = get_scheduler(optimizer, train_config.get("scheduler", {})) if "scheduler" in train_config else None
    early_stopper = EarlyStopping(**train_config.get("early_stopping", {})) if "early_stopping" in train_config else None
    return model, optimizer, scheduler, early_stopper, tensorboard

def prepare_dataset(dataset_config: dict):
    """
    Prepare the dataset and dataloader based on the provided configuration.
    Args:
        dataset_config (dict): Configuration dictionary for the dataset, including type and parameters.
    Returns:
        dataset (torch.utils.data.Dataset): The instantiated dataset.
        dataloader (torch.utils.data.DataLoader): The dataloader for the dataset.
    """
    config = dataset_config
    datasets = {
        "ExampleDataset": ExampleDataset,
        "MnistDataset": MnistDataset,
        # "VOC": VOCDataset,
        # other datasets can be added here
    }
    if "type" not in config:
        raise ValueError("Dataset type must be specified in the configuration.")
    if config["type"] not in datasets:
        raise ValueError(f"Unsupported dataset type: {config['type']}. Available options: {list(datasets.keys())}")

    dataset = datasets[config["type"]](**config.get("dataset_params", {}))
    dataloader = DataLoader(dataset, **config.get("dataloader_params", {}), collate_fn=dataset.collate_fn) if "dataloader_params" in config else None

    return dataset, dataloader

def get_model(model_config: dict):
    """
    Get the model based on the provided configuration.
    Args:
        model_config (dict): Configuration dictionary for the model, including type and parameters.
    Returns:
        model (torch.nn.Module): The instantiated model.
    """
    models = {
        "ExampleModel": ExampleModel,
        "MnistModel": MnistModel, 
        # other models can be added here
    }
    if "type" not in model_config:
        raise ValueError("Model type must be specified in the configuration.")
    if model_config["type"] not in models:
        raise ValueError(f"Unsupported model type: {model_config['type']}. Available options: {list(models.keys())}")
    
    return models[model_config["type"]](**model_config.get("params", {}))

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
    if "type" not in optimizer_config:
        raise ValueError("Optimizer type must be specified in the configuration.")
    if optimizer_config["type"] not in optimizers:
        raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}. Available options: {list(optimizers.keys())}")

    return optimizers[optimizer_config["type"]](model.parameters(), **optimizer_config.get("params", {}))

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
        "StepLR": lr_scheduler.StepLR,
        "LinearLR": lr_scheduler.LinearLR,
        "MultiStepLR": lr_scheduler.MultiStepLR,
        "ExponentialLR": lr_scheduler.ExponentialLR,
        "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
        "ReduceLROnPlateau": lr_scheduler.ReduceLROnPlateau, # need to modify step in training loop
        "OneCycleLR": lr_scheduler.OneCycleLR,
        # others can be added here
    }
    if "type" not in scheduler_config:
        raise ValueError("Scheduler type must be specified in the configuration.")
    if scheduler_config["type"] not in schedulers:
        raise ValueError(f"Unsupported scheduler type: {scheduler_config['type']}. Available options: {list(schedulers.keys())}")

    return schedulers[scheduler_config["type"]](optimizer, **scheduler_config.get("params", {}))