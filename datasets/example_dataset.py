import torch
from torch.utils.data import Dataset
from torchvision import transforms

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

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def resize(self, image, size=(512, 416)):
        image = transforms.Resize(size)(image)
        return image
    
    def to_tensor(self, image):
        image = transforms.ToTensor()(image)
        return image
    
    def normalize(self, image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        image = transforms.Normalize(mean=mean, std=std)(image)
        return image

    
class ExampleDataset(Dataset):
    """
    Example dataset class that can be used to load and process data.
    This is a placeholder for demonstration purposes.
    """

    def __init__(self, data_source, transform=ExampleTransform()):
        self.data_source = data_source
        self.data = self.load_data()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return self.transform(sample)

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
            targets.append({
                'boxes': torch.as_tensor(target['boxes'], dtype=torch.float32),
                'labels': torch.as_tensor(target['labels'], dtype=torch.int64),
            })
        images = torch.stack(images, dim=0)
        
        return images, targets 