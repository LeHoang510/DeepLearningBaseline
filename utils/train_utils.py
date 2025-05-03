from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
    
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
    def state_dict(self):
        return {
            'counter': self.counter,
            'best_loss': self.best_loss
        }
    
    def load_state_dict(self, state_dict):
        self.counter = state_dict['counter']
        self.best_loss = state_dict['best_loss']


class TensorBoard:
    def __init__(self, log_dir=Path("outputs/logs")):
        self.log_dir = log_dir
        self.writer = None
    
    def create_writer(self):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
    
    def write_train(self, epoch, train_loss):
        self.create_writer()
        self.writer.add_scalar("Loss/Train", train_loss, epoch)
      
    def write_train_val(self, epoch, train_loss, val_loss, val_acc):
        self.create_writer()
        self.writer.add_scalars("Loss", {
            "Train": train_loss,
            "Val": val_loss
        }, epoch)
        self.writer.add_scalar("Accuracy/Val", val_acc, epoch)
       
    def write_scheduler(self, epoch, scheduler):
        self.create_writer()
        self.writer.add_scalar("Scheduler/Learning Rate", scheduler.get_last_lr()[0], epoch)

    def close(self):
        self.writer.close()