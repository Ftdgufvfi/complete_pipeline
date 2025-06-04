import cv2
import numpy as np
from ultralytics import YOLO

# Frame differencing method
def frame_difference(frame2, frame1):
    return cv2.absdiff(frame2, frame1)

# gausiiaan of mixtures modelling
fgbg=cv2.createBackgroundSubtractorMOG2(history=50,varThreshold=20, detectShadows=False)
method="guassian_mix"
def bgsub_guassian(present_frame):
    return fgbg.apply(present_frame)



# Function to get motion mask
def get_mask(frame1, frame2):

    """ Obtains image mask for moving objects """
    # Step 1: Smooth with Gaussian Blur
    #blurred = cv2.GaussianBlur(frame2, (5, 5), sigmaX=0)

    # Step 2: Apply Laplacian (edge detector)
    # laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = frame1
    if method=="frame_dif":
        frame_diff = frame_difference(laplacian, frame1)
    elif method=="guassian_mix":
        frame_diff=bgsub_guassian(laplacian)


    # Blur and threshold the frame difference

    mask = cv2.adaptiveThreshold(frame_diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 3)
    mask = cv2.medianBlur(mask, 3)

    # Morphological operations
    kernel = np.ones((9, 9), dtype=np.uint8)  # Fixed kernel definition
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return mask

# Function to extract contours (bounding boxes)
def get_contour_detections(mask, thresh=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = [
                    [x, y, x + w, y + h, w * h]
                    for cnt in contours
                    for (x, y, w, h) in [cv2.boundingRect(cnt)]
                    if (w * h) > thresh
                ]
    return np.array(detections)