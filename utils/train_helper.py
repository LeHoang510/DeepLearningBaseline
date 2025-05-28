from pathlib import Path

import torch
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
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_config["learning_rate"],
        momentum=train_config["momentum"],
        weight_decay=train_config["weight_decay"]
    )
    scheduler = MultiStepLR(optimizer, **train_config["scheduler"])
    early_stopper = EarlyStopping(**train_config["early_stopping"])
    return tensorboard, model, optimizer, scheduler, early_stopper

