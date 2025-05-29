import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from PIL import Image

from utils.utils import load_json

class ExampleTransform:
    """
    Example transform class that can be used to apply transformations to the dataset.
    This is a placeholder for demonstration purposes.
    """
    def __init__(self):
        self.transforms = [
            self.resize,
            self.to_tensor,
            self.normalize
        ]

    def __call__(self, image, label=None):
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label

    def resize(self, image, label, size=(512, 416)):
        image = F.resize(image, size)
        if label is not None:
            label['boxes'] = [
                [int(bbox[0] * size[0] / image.size[0]), int(bbox[1] * size[1] / image.size[1]),
                 int(bbox[2] * size[0] / image.size[0]), int(bbox[3] * size[1] / image.size[1])]
                for bbox in label['boxes']
            ]
        return image, label

    def to_tensor(self, image, label):
        image = F.to_tensor(image)
        return image, label

    def normalize(self, image, label, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        image = F.normalize(image, mean=mean, std=std)
        return image, label

class ExampleDataset(Dataset):
    """
    Example dataset class that can be used to load and process data.
    This is a placeholder for demonstration purposes.
    """

    def __init__(self, data_source, transform=ExampleTransform()):
        self.data_source = data_source
        self.data = self.load_data()
        self.transform = transform
    
    def load_data(self):
        data = load_json(self.data_source)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image_path = sample['image_path']
        label = {
            'boxes':  torch.as_tensor(sample['boxes'], dtype=torch.float32),
            'labels': torch.as_tensor(sample['labels'], dtype=torch.int64)
        }

        image = Image.open(image_path).convert('RGB')

        if self.transform is None:
            image, label = self.transform(image, label)

        return image, label

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle batching of data.
        Args:
            batch (list): List of samples from the dataset.
        Returns:
            tuple: A tuple containing a batch of images and their corresponding targets.
        """
        images = []
        targets = []

        for img, target in batch:
            images.append(img)
            targets.append(target)
        images = torch.stack(images, dim=0)
        
        return images, targets 