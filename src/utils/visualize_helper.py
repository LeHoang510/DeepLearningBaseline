import math

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

def plot_multiple_gray_images(images, labels=None, preds=None, display=True, save_path=None):
    """
    Plots multiple gray images with optional labels and predictions.

    Args:
        images (torch.Tensor): List of image tensors to plot.
        labels (torch.Tensor, optional): List of true labels for the images.
        preds (torch.Tensor, optional): List of predicted labels for the images.
        save_path (str, optional): Path to save the plotted images. If None, it will display the images.
    .. note::
        loads images, labels from dataloader and plots them in a grid."""
    nb_img = images.size(0)
    ncols = 5
    nrows = math.ceil(nb_img / ncols)

    plt.figure(figsize=(ncols * 2, nrows * 2))

    for i in range(nb_img):
        img = images[i].squeeze()
        label = labels[i] if labels is not None else None
        pred = preds[i] if preds is not None else None

        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')

        title = ""
        if label is not None:
            title += f"Label: {label}"
        if pred is not None:
            title += f", Pred: {pred}" if title else f"Pred: {pred}"
        if title:
            plt.title(title)

    plt.tight_layout()
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

def plot_multiple_color_images(images, labels=None, preds=None, display=True, save_path=None):
    """
    Plots multiple color images with optional labels and predictions.
    Args:
        images (torch.Tensor): List of image tensors to plot.
        labels (torch.Tensor, optional): List of true labels for the images.
        preds (torch.Tensor, optional): List of predicted labels for the images.
        save_path (str, optional): Path to save the plotted images. If None, it will display the images.
    .. note::
        loads images, labels from dataloader and plots them in a grid.
    """
    nb_img = images.size(0)
    ncols = 5
    nrows = math.ceil(nb_img / ncols)

    plt.figure(figsize=(ncols * 2, nrows * 2))

    for i in range(nb_img):
        img = images[i].permute(1, 2, 0).numpy()
        label = labels[i] if labels is not None else None
        pred = preds[i] if preds is not None else None

        plt.subplot(nrows, ncols, i + 1)
        plt.imshow(img)
        plt.axis('off')

        title = ""
        if label is not None:
            title += f"Label: {label}"
        if pred is not None:
            title += f", Pred: {pred}" if title else f"Pred: {pred}"
        if title:
            plt.title(title)

    plt.tight_layout()
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
