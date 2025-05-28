from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import torch

from models.CustomFasterRCNN.faster_rcnn_fpn import CustomFasterRCNN

def load_model(model_path, device):
    config = torch.load(model_path, map_location=device)["config"]
    model_config = config["model_config"]
    model = CustomFasterRCNN(config=model_config)
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    return model
    
def prepare_training(config, output_dir):
    dataset_config = config["dataset_config"]
    model_config = config["model_config"]
    train_config = config["training_config"]
    tensorboard_path = output_dir / "tensorboard"

    tensorboard = TensorBoard(log_dir=tensorboard_path)
    model = CustomFasterRCNN(config=model_config)
    optimizer = optim.SGD(
        model.parameters(),
        lr=train_config["learning_rate"],
        momentum=train_config["momentum"],
        weight_decay=train_config["weight_decay"]
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=train_config["milestones"],
        gamma=train_config["gamma"]
    )
    early_stopper = EarlyStopping(
        patience=train_config["early_stopping_patience"],
        min_delta=train_config["early_stopping_min_delta"],
        mode=train_config["early_stopping_mode"]
    )
    return tensorboard, model, optimizer, scheduler, early_stopper

