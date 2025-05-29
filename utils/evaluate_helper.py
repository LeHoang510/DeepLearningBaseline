import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

def calculate_metric(predictions, targets):
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    
    for pred, target in zip(predictions, targets):
        metric.update([{
            "boxes": pred["boxes"],
            "scores": pred["scores_type"],
            "labels": pred["labels_type"]
        }], [{
            "boxes": target["boxes"],
            "labels": target["labels_type"]
        }])
    
    results = metric.compute()

    return {
            "map": results["map"].item(),
            "map_50": results["map_50"].item(),
            "map_75": results["map_75"].item(),
            "ap_per_class": results["map_per_class"].tolist()
        }

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            all_preds.extend(outputs)
            all_targets.extend(targets)
    
    metric = calculate_metric(all_preds, all_targets)

    return metric