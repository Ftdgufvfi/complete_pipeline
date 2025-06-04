# main.py

import cv2
import noise_reduction
import contrast_adjustments
import sharpening
import static_background_sub as bg_sub
import yolo_utilities as yolo_utils
import numpy as np
#import swanet_utils
import zero_dce_utils


# Open the video file or webcam
#video_path = r"C:\Users\shiva\Downloads\WhatsApp Video 2025-05-30 at 18.19.41.mp4"
video_path = r"C:\Users\shiva\Downloads\Unknown Camera1_20250312120150.avi"  # Replace with your video file path or '0' for webcam
#video_path = r"C:\Users\shiva\Downloads\Unknown Camera1_20250319200247.avi"
#video_path = r"C:\Users\shiva\OneDrive\Desktop\DualDegreeProject\back_ground_subtraction\Unknown Camera1_20250312170000.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video!")
    exit()

count = 1
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    #frame_enhanced = swanet_utils.swanet(frame)
    frame_enhanced = zero_dce_utils.enhance_image(frame)
    #frame_enhanced = frame
    frame_grey = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2GRAY)
    
    # Convert the frame to grayscale
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not ret:
        print("Error: Failed to read frame or video ended!")
        break

    # (Optional) Convert the frame to grayscale if you want
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # For now we keep it color (BGR)

    # --- Step 1: Noise Reduction ---
    #frame_denoised = noise_reduction.noise_reduction_bilateral(frame)
    # You can choose Bilateral, Median, or Gaussian

    # --- Step 2: Contrast Enhancement ---
    #frame_contrast = contrast_adjustments.clahe_contrast_rgb(frame_denoised)
    # OR you can test: histogram_equalization_rgb, gamma_correction, etc.

    # --- Step 3: Sharpening (Optional) ---
    #frame_sharp = sharpening.laplacian_sharpen(frame_denoised)

    # ---- Step 4: Static background Step ------
    #frame_sharp_grey = cv2.cvtColor(frame_sharp, cv2.COLOR_BGR2GRAY)

    # Background mask and detections
    mask = bg_sub.bgsub_guassian(frame_grey)
    cv2.imshow('Motion Mask', mask)

    count += 1
    if count % 5 != 0:
        continue
    else:
        count = 1

    detections = bg_sub.get_contour_detections(mask)
    # Run YOLO on full frame
    yolo_bboxes_full, yolo_scores_full = yolo_utils.get_yolo_detections(frame_enhanced, conf_thresh=0.15)

    # fusion
    bg_boxes_coords = np.array([det[:4] for det in detections], dtype=np.float32)
    all_bboxes, all_scores = yolo_utils.select_yolo_with_bg_iou(yolo_bboxes_full, yolo_scores_full, bg_boxes_coords, conf_thresh=0.38, iou_thresh=0.7)  #frame, forced_threshold=0.3
    
    if len(all_bboxes):
        final_bboxes = yolo_utils.non_max_suppression(all_bboxes, all_scores, threshold=0.1)
        i = 0
        for box in final_bboxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.putText(frame, "Object", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        #0.5, (0, 255, 0), 2)
            label = f"{all_scores[i]:.2f}"  # Formats the score to 2 decimal places
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 255, 0), 2)
            i = i + 1

    cv2.imshow('Detections', frame)
    cv2.imshow('enhanced_frame', frame_enhanced)
    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
