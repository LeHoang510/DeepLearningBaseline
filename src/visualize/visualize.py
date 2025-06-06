import random

from torch.utils.data import DataLoader

from datasets.mnist_dataset import MnistDataset
from utils.visualize_helper import plot_multiple_gray_images, plot_gray_image

def visualize_images():

    dataset = MnistDataset()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    first_batch = next(iter(dataloader))
    images, labels = first_batch

    plot_multiple_gray_images(images, labels=labels, display=True)

def visualize_image(id: int):
    dataset = MnistDataset()
    image, label = dataset[id]
    plot_gray_image(image=image, label=label)

if __name__ == "__main__":
    # visualize_images()
    visualize_image(0)
