import torch
from tqdm import tqdm

from models.example_model import ExampleModel
from models.mnist_model import MnistModel
from evaluate.example_evaluate import calculate_map
from evaluate.mnist_evaluate import mnist_accuracy, mnist_f1_score

EVALUATE_FUNCTIONS = {
    "accuracy_mnist": mnist_accuracy,
    "f1_score_mnist": mnist_f1_score,
    "map": calculate_map,
}

@torch.no_grad()
def evaluate(model, dataloader, device, metrics={}):
    model.eval()
    all_preds = []
    all_targets = []
    all_losses = {}

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)
            
            predictions, loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            all_preds.extend(predictions)
            all_targets.extend(targets)

            all_losses["total_loss"] = all_losses.get("total_loss", 0.0) + losses.item()
            for k, v in loss_dict.items():
                all_losses[k] = all_losses.get(k, 0.0) + v.item()

    # Calculate average loss and metric
    avg_all_losses = {k: v / len(dataloader) for k, v in all_losses.items()}

    metric_results = {}
    # Calculate metrics
    for metric in metrics:
        metric_fn = metric.get("function")
        if metric_fn in EVALUATE_FUNCTIONS:
            metric_results.update(EVALUATE_FUNCTIONS[metric_fn](all_preds, all_targets, **metric.get("params", {})))
        else:
            raise ValueError(f"Unsupported metric function: {metric_fn}. Available options: {list(EVALUATE_FUNCTIONS.keys())}")

    return avg_all_losses, metric_results