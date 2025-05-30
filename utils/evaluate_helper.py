import torch
from tqdm import tqdm

from models.example_model import ExampleModel
from models.mnist_model import MnistModel
from evaluate.example_evaluate import calculate_map
from evaluate.mnist_evaluate import calculate_mnist_accuracy

evaluate_function = {
    ExampleModel: calculate_map,
    MnistModel: calculate_mnist_accuracy,
}

@torch.no_grad()
def evaluate(model, dataloader, device, evaluate_params={}):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)
            
            predictions, loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            all_preds.extend(predictions)
            all_targets.extend(targets)
            total_loss += losses.item()
    
    # Calculate average loss and metric
    avg_loss = total_loss / len(dataloader)
    metric = evaluate_function[type(model)](all_preds, all_targets, **evaluate_params)

    return avg_loss, metric