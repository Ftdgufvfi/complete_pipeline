# noise_reduction.py
import cv2

def noise_reduction_bilateral(img, d=5, sigmaColor=25, sigmaSpace=25):
    """
    Apply Bilateral Filtering to reduce noise while preserving edges.
    """
    return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

def noise_reduction_median(img, ksize=5):
    """
    Apply Median Filtering to remove salt-and-pepper noise.
    """
    return cv2.medianBlur(img, ksize)

def noise_reduction_gaussian(img, ksize=(5, 5), sigmaX=1):
    """
    Apply Gaussian Filtering to reduce general noise.
    """
    return cv2.GaussianBlur(img, ksize, sigmaX)

def noise_reduction_fastNL(img):
    return cv2.fastNlMeansDenoising(img, None, h=10)
