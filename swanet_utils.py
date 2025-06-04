import sys
import os

sys.path.append(os.path.abspath("SWANet"))

from model.SWANet import SWANet

import torch
import cv2
import numpy as np
from torchvision import transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SWANet(weight1_path='SWanet/checkpoints/stage1.pth', weight2_path='SWanet/checkpoints/stage2.pth')
model.to(device).eval()

def swanet(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced_tensor = model(input_tensor)

    enhanced_img = enhanced_tensor.squeeze().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    enhanced_img = (enhanced_img * 255).astype(np.uint8)
    enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
    return enhanced_img_bgr
