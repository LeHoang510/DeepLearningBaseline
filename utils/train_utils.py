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
    def __init__(self, log_dir=Path("output/logs")):
        self.log_dir = log_dir
        self.writer = None
    
    def write(self, epoch, total_epoch, train_loss, val_loss=None, val_acc=None):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        if val_loss is not None and val_acc is not None:
            self.writer.add_scalars("Loss", {
                "Train": train_loss,
                "Val": val_loss
            }, epoch)
            self.writer.add_scalar("Accuracy/Val", val_acc, epoch)
            print(f"Epoch {epoch}/{total_epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Acc: {val_acc}")
        else:
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            print(f"Epoch {epoch}/{total_epoch}, Train Loss: {train_loss}")

    def close(self):
        self.writer.close()