import cv2
import numpy as np
import cv2

def clahe_contrast_rgb(img):
    # Convert BGR image to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Split into channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    
    # Merge back the channels
    lab_clahe = cv2.merge((l_clahe, a, b))
    
    # Convert LAB back to BGR
    final = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return final



def histogram_equalization_rgb(img):
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    
    # Split into channels
    y, cr, cb = cv2.split(ycrcb)
    
    # Equalize only the Y channel
    y_eq = cv2.equalizeHist(y)
    
    # Merge back
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    
    # Convert back to BGR
    img_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)
    
    return img_eq



def gamma_correction(img, gamma=1.2):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def contrast_stretching(img):
    min_val = np.min(img)
    max_val = np.max(img)
    stretched = (img - min_val) * (255 / (max_val - min_val))
    return stretched.astype(np.uint8)