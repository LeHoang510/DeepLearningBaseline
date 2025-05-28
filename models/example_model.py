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

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_2(x)
        return x
    