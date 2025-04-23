import os.path as osp
from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, model, scheduler, log_dir=osp.join("output", "logs")):
        self.model = model
        self.scheduler = scheduler
        self.writer = None
        self.log_dir = log_dir

    def write_dict(self, epoch, total_epoch, train_loss, val_loss=None, val_acc=None, test_loss=None):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)
        
        if val_loss is not None and val_acc is not None:
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Val", val_acc, epoch)
            print(f"Epoch {epoch}/{total_epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Acc: {val_acc}")
        elif test_loss is not None:
            self.writer.add_scalar("Loss/Test", test_loss, epoch)
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            print(f"Epoch {epoch}/{total_epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}")
        else:
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            print(f"Epoch {epoch}/{total_epoch}, Train Loss: {train_loss}")
    
    def close(self):
        self.writer.close()