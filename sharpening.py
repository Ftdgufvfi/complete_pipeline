import cv2
import numpy as np

def unsharp_mask(img, ksize=(5,5), amount=2, threshold=0):
    blurred = cv2.GaussianBlur(img, ksize, 0)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    return sharpened

def laplacian_sharpen(img):
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])  # Standard Laplacian Kernel
    sharpened = cv2.filter2D(img, -1, kernel)
    return sharpened

def high_pass_filter(img, ksize=5):
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    high_pass = cv2.subtract(img, blurred)
    sharpened = cv2.add(img, high_pass)
    return sharpened