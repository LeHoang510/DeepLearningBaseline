import torch
import torch.nn as nn

class ExampleModel(nn.Module):
    """
    A custom PyTorch model that extends nn.Module.
    This is a placeholder for your specific model architecture.
    """

    def __init__(self, input_size, output_size, dropout_rate=0):
        """
        Initializes the ExampleModel with the given configuration.
        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super(ExampleModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        self.fc_1 = nn.Linear(input_size, 512)
        self.fc_2 = nn.Linear(512, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, images, targets=None):
        """
        Forward pass of the model.
        Args:
            images (torch.Tensor): Input images.
            targets (torch.Tensor, optional): Target labels for training. Defaults to None.
        Returns:
            torch.Tensor: Output predictions.
        """
        x = self.fc_1(images)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)

        if targets is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(x, targets)
            return x, loss
        
        return x

        
    