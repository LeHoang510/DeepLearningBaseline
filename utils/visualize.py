from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = BiradsTransform()
    image, _ = transform(image)  
    return image.unsqueeze(0)  

def visualize_image(image_tensor, predictions, targets=None, save_path=None):
    image_tensor = denormalize(image_tensor.squeeze(0))  
    image = transforms.ToPILImage()(image_tensor.squeeze(0))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    
    for box, label_type, label_patho, score_type, score_patho in zip(
        predictions["boxes"],
        predictions["labels_type"],
        predictions["labels_patho"],
        predictions["scores_type"],
        predictions["scores_patho"],
    ):  
        if score_type > 0.0 or score_patho > 0.0:  
            box = box.detach().cpu().numpy().astype(int)
            x_min, y_min, x_max, y_max = box
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)
            text_type = f"{LABEL_TYPE_MAP[label_type.item()]}: {score_type:.2f}"
            text_patho = f"{LABEL_PATHO_MAP[label_patho.item()]}: {score_patho:.2f}"
            text = f"{text_type}|{text_patho}"
            
            draw.text((box[0], box[1]), text, fill="blue", font=font)

    if targets:
        for box, label_type, label_patho in zip(
            targets["boxes"],
            targets["labels_type"],
            targets["labels_patho"],
        ):
            x_min, y_min, x_max, y_max = box
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="green", width=2)
            text_type = f"{label_type}"
            text_patho = f"{label_patho}"
            text = f"{text_type}|{text_patho}"
            
            draw.text((box[0], box[1]), text, fill="blue", font=font)
    
    plt.imshow(np.array(image))

    if not save_path:
        plt.axis("off")
        plt.show()
    else:
        image.save(save_path)
        print(f"Saved image with predictions to {save_path}")