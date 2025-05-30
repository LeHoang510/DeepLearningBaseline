import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

def calculate_mnist_accuracy(predictions, targets):
    """
    Calculate accuracy for MNIST classification.
    
    Args:
        predictions: List of model outputs (log probabilities from log_softmax)
        targets: List of target labels
        
    Returns:
        float: Accuracy percentage (0-100)
    """
    # Convert list of predictions to tensor
    preds_tensor = torch.stack(predictions)
    
    # Convert list of target dictionaries to tensor
    # Assuming targets is list of dicts like [{'labels': tensor(3)}, ...]
    targets_tensor = torch.stack([t['labels'] for t in targets])
    
    # Get predicted classes (argmax of log probabilities)
    _, pred_classes = preds_tensor.max(dim=1)
    
    # Calculate accuracy
    correct = (pred_classes == targets_tensor).sum().item()
    total = targets_tensor.size(0)
    accuracy = 100.0 * correct / total
    
    return accuracy