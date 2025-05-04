from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from models.CustomFasterRCNN.faster_rcnn_fpn import CustomFasterRCNN

def prepare_training(config, output_dir):
    dataset_config = config["dataset_config"]
    model_config = config["model_config"]
    train_config = config["training_config"]
    tensorboard_path = output_dir / "tensorboard"

    tensorboard = TensorBoard(log_dir=tensorboard_path)
    model = CustomFasterRCNN(
        backbone=model_config["backbone"],
        num_classes_type=model_config["num_classes_type"],
        num_classes_patho=model_config["num_classes_patho"]
    )
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

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.mode = mode
        if mode == 'max':
            self.best_metric = -float('inf')
            self.compare_op = lambda x, y: x > y + min_delta
        elif mode == 'min':
            self.best_metric = float('inf')
            self.compare_op = lambda x, y: x < y - min_delta
    
    def __call__(self, val_metric):
        if self.compare_op(val_metric, self.best_metric):
            self.best_metric = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def state_dict(self):
        return {
            'counter': self.counter,
            'best_metric': self.best_metric,
            'mode': self.mode,
        }
    
    def load_state_dict(self, state_dict):
        self.counter = state_dict['counter']
        self.best_loss = state_dict['best_metric']
        self.mode = state_dict.get('mode', 'max')

class TensorBoard:
    def __init__(self, log_dir=Path("outputs/logs")):
        self.log_dir = log_dir
        self.writer = None
    
    def create_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
    
    def write_scalar(self, title, value, epoch):
        self.create_writer()
        self.writer.add_scalar(title, value, epoch)
    
    def write_scalars(self, title, values, epoch):
        self.create_writer()
        self.writer.add_scalars(title, values, epoch)


    def close(self):
        self.writer.close()