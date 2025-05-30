import torch.nn as nn
import torch.nn.functional as F

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10) 

        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, image, target=None):
        """        Forward pass of the model.
        Args:
            image (torch.Tensor): Input images.
            target (torch.Tensor, optional): Target labels for training. Defaults to None.
        Returns:
            torch.Tensor: The output logits.
            If target is provided, also returns the loss.
        """
        x = self.relu(self.conv1(image))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)

        if target is not None:
            loss_fn = nn.NLLLoss()
            loss = loss_fn(x, target)
            return x, {'loss': loss}
        
        return x

