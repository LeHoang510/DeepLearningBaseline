import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent  
sys.path.append(str(root_dir))

import torch
from torch.utils.data import DataLoader
from dataset.birads_dataset import BIRADSDataset

def collate_fn(batch):
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    return images, targets

def test_dataset(dataset, batch_size=4, num_workers=8):
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=num_workers,
                            collate_fn=collate_fn)

    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}")
        for i in range(len(images)):
            print(f" - Image {i}: {images[i]}")
            print(f" - Label {i}: {labels[i]}")

        break  # Test 1 batch là đủ

if __name__ == "__main__":
    dataset = BIRADSDataset(data_dir="data/birads_data", transform=None)
    test_dataset(dataset, batch_size=4, num_workers=8)