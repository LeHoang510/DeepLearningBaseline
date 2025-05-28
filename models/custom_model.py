import torch
import torch.nn as nn

class CustomModel(nn.Module):
    """
    A custom PyTorch model that extends nn.Module.
    This is a placeholder for your specific model architecture.
    """
    
    def __init__(self, config: dict):
        """
        Initializes the CustomModel with the given configuration.
        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        super(CustomModel, self).__init__()
        self.config = config
        input_size = config["input_size"] 
        output_size = config["output_size"]
        dropout_rate = config["dropout_rate"] if "dropout_rate" in config else 0

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
    