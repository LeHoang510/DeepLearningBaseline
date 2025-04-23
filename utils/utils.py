import random
import os.path as osp

import numpy as np
import torch
from torchvision import transforms

from tqdm import tqdm

def set_seed(seed: int = 555):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministc = True
	torch.backends.cudnn.benchmark = False

def de_normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
	result = img*std+mean
	result = np.clip(result, 0, 1)
	return result
	
def target_transform(target, img_size=(128, 128)):
	img = transforms.Resize(img_size)(target)
	img = transforms.functional.pil_to_tensor(img).squeeze()
	img = img-1
	img = img.to(torch.long)
	return img

def evaluate(model, test_loader, criterion, device):
	model.eval()
	test_loss = 0.0
	with torch.no_grad():
		for i, (inputs, labels) in enumerate(tqdm(test_loader)):
			inputs, labels = inputs.to(device), labels.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			test_loss += loss.item()
	
	test_loss = test_loss / len(test_loader)
	return test_loss

