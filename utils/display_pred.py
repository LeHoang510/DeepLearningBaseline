import torch
import matplotlib.pyplot as plt

from utils.utils import *

@torch.inference_mode()
def display_prediction(model, image, target, device):
    model.eval()
    img = image[None, ...].to(device)
    output = model(img)
    pred = torch.argmax(output, axis=1)

    plt.figure(figsize=(10, 5))

    plt.subplot(1,3,1)
    plt.axis("off")
    plt.title("Input Image")
    plt.imshow(de_normalize(image.numpy().transpose(1, 2, 0)))

    plt.subplot(1,3,2)
    plt.axis("off")
    plt.title("Ground Truth")
    plt.imshow(target)

    plt.subplot(1,3,3)
    plt.axis("off")
    plt.title("Prediction")
    plt.imshow(pred.cpu().squeeze())

    plt.show()