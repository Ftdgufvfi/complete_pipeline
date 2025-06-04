import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys
import cv2

# Add the folder path where model.py lives (make sure path is correct)
sys.path.append(r"C:\Users\shiva\OneDrive\Desktop\DualDegreeProject\Complete_pipleine\Zero-DCE\code")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from model import DCENet

# Initialize model once, load checkpoint once
model = DCENet()
ckpt_path = r"C:\Users\shiva\OneDrive\Desktop\DualDegreeProject\Complete_pipleine\Zero-DCE\train-jobs\ckpt\8LE-color-loss2_best_model.pth"
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Predefine transform globally (avoid re-creating each call)
transform = transforms.ToTensor()

def enhance_image(img):
    """
    img: input BGR image as numpy array (OpenCV format)
    returns: enhanced BGR image as numpy array (OpenCV format)
    """
    # Convert BGR to RGB once
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert numpy array to PIL image (no extra convert needed)
    pil_img = Image.fromarray(img_rgb)

    # Transform once per image (no need to call convert('RGB') again)
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Model inference
        results, _ = model(input_tensor)
        enhanced = results[1]

    # Convert tensor to numpy image efficiently
    enhanced_img = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
    # Clamp and convert to uint8
    enhanced_img = np.clip(enhanced_img * 255, 0, 255).astype(np.uint8)

    # Convert RGB back to BGR for OpenCV usage
    enhanced_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)

    return enhanced_bgr
