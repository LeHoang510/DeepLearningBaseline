import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.mnist_dataset import MnistTransform

TRANSFORMS = {
    "MnistTransform": MnistTransform(),
    # Add other transforms here as needed
}

def plot_gray_image(image, label=None, pred=None, display=True, save_path=None):
    """
    Plots an image with optional label and prediction.
    Args:
        image (torch.Tensor): The image tensor to plot.
        label (int, optional): The true label of the image.
        pred (int, optional): The predicted label of the image.
        save_path (str, optional): Path to save the plotted image. If None, it will display the image.
    """
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    if label is not None:
        plt.title(f"Label: {label}, Pred: {pred}" if pred is not None else f"Label: {label}")
    else:
        plt.title("Image")
    plt.imshow(image.squeeze(), cmap='gray')
    plt.savefig(save_path, dpi=300, bbox_inches='tight') if save_path else None
    plt.show() if display else None
    plt.close()

def plot_color_image(image, label=None, pred=None, display=True, save_path=None):
    """
    Plots a color image with optional label and prediction.

    Args:
        image (torch.Tensor): The image tensor to plot.
        label (int, optional): The true label of the image.
        pred (int, optional): The predicted label of the image.
        save_path (str, optional): Path to save the plotted image. If None, it will display the image.
    """
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    if label is not None:
        plt.title(f"Label: {label}, Pred: {pred}" if pred is not None else f"Label: {label}")
    else:
        plt.title("Image")
    plt.imshow(image.permute(1, 2, 0).numpy())
    plt.savefig(save_path, dpi=300, bbox_inches='tight') if save_path else None
    plt.show() if display else None
    plt.close()

def denormalize(tensor, mean, std):
    """
    Denormalizes a tensor using the provided mean and standard deviation.

    Args:
        tensor (torch.Tensor): The input tensor to denormalize.
        mean (float or list): The mean value(s) for denormalization.
        std (float or list): The standard deviation value(s) for denormalization.

    Returns:
        torch.Tensor: The denormalized tensor.
    """
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)

    return tensor * std + mean

def apply_transform(image, transform_name):
    """
    Applies a specified transform to an image.

    Args:
        image (torch.Tensor): The input image tensor.
        transform_name (str): The name of the transform to apply.

    Returns:
        torch.Tensor: The transformed image tensor.
    """
    if transform_name not in TRANSFORMS:
        raise ValueError(f"Transform '{transform_name}' is not defined. Available transforms: {list(TRANSFORMS.keys())}")

    transform = TRANSFORMS[transform_name]
    return transform(image)
