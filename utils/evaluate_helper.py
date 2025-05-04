import torch
from tqdm import tqdm
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou


def calculate_map(predictions, targets):
    metric_type = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    metric_patho = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    
    for pred, target in zip(predictions, targets):
        metric_type.update([{
            "boxes": pred["boxes"],
            "scores": pred["scores_type"],
            "labels": pred["labels_type"]
        }], [{
            "boxes": target["boxes"],
            "labels": target["labels_type"]
        }])
        
        metric_patho.update([{
            "boxes": pred["boxes"],
            "scores": pred["scores_patho"],
            "labels": pred["labels_patho"]
        }], [{
            "boxes": target["boxes"],
            "labels": target["labels_patho"]
        }])
    
    results_type = metric_type.compute()
    results_patho = metric_patho.compute()
    
    return {
        "type_map": {
            "map": results_type["map"].item(),
            "map_50": results_type["map_50"].item(),
            "map_75": results_type["map_75"].item(),
            "ap_per_class": results_type["map_per_class"].tolist()
        },
        "patho_map": {
            "map": results_patho["map"].item(),
            "map_50": results_patho["map_50"].item(),
            "map_75": results_patho["map_75"].item(),
            "ap_per_class": results_patho["map_per_class"].tolist()
        }
    }

def calculate_accuracy(preds, targets, iou_threshold=0.5):
    type_correct = 0
    patho_correct = 0
    total_objects = 0
    
    for pred, target in zip(preds, targets):
        pred_boxes = pred['boxes']
        gt_boxes = target['boxes']
        
        if len(gt_boxes) == 0:
            continue
            
        # Matching predictions with ground truth
        ious = box_iou(pred_boxes, gt_boxes)
        best_gt_idx = ious.argmax(dim=1)
        best_iou = ious.max(dim=1).values
        valid_matches = best_iou > iou_threshold
        
        if valid_matches.any():
            matched_gt = best_gt_idx[valid_matches]
            
            # Accuracy for each label
            pred_type = pred['labels_type'][valid_matches]
            gt_type = target['labels_type'][matched_gt]
            type_correct += (pred_type == gt_type).sum().item()
            
            pred_patho = pred['labels_patho'][valid_matches]
            gt_patho = target['labels_patho'][matched_gt]
            patho_correct += (pred_patho == gt_patho).sum().item()
        
        total_objects += len(gt_boxes)
    
    return {
        'type_accuracy': type_correct / max(1, total_objects),
        'patho_accuracy': patho_correct / max(1, total_objects)
    }

@torch.no_grad()
def evaluate(model, dataloader, device):
    results = {}
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            all_preds.extend(outputs)
            all_targets.extend(targets)
    
    accuracy = calculate_accuracy(all_preds, all_targets)
    map = calculate_map(all_preds, all_targets)

    results.update(accuracy)
    results.update(map)
    return results