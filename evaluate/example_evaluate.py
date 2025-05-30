import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision

def calculate_map(predictions, targets):
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
