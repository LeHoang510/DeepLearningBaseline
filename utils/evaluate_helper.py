import torch
from tqdm import tqdm

from models.example_model import ExampleModel
from evaluate.example_evaluate import calculate_map

evaluate_function = {
    ExampleModel: calculate_map,
}

@torch.no_grad()
def evaluate(model, dataloader, device, evaluate_params=None):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions, loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            all_preds.extend(predictions)
            all_targets.extend(targets)
            total_loss += losses.item()
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)

    if evaluate_params is not None:
        metric = evaluate_function[type(model)](all_preds, all_targets, **evaluate_params)
    else:
        metric = evaluate_function[type(model)](all_preds, all_targets)

    return avg_loss, metric