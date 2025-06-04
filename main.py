# main.py

import cv2
import noise_reduction
import contrast_adjustments
import sharpening
import static_background_sub as bg_sub
import yolo_utilities as yolo_utils
import numpy as np

# Open the video file or webcam
video_path = r"C:\Users\shiva\Downloads\Unknown Camera1_20250312120150.avi"  # Replace with your video file path or '0' for webcam
#video_path = r"C:\Users\shiva\OneDrive\Desktop\DualDegreeProject\back_ground_subtraction\Unknown Camera1_20250312170000.avi"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video!")
    exit()

count = 1
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Convert the frame to grayscale
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if not ret:
        print("Error: Failed to read frame or video ended!")
        break

    count += 1
    if count % 5 != 0:
        continue
    else:
        count = 1

    # (Optional) Convert the frame to grayscale if you want
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # For now we keep it color (BGR)

    # --- Step 1: Noise Reduction ---
    frame_denoised = noise_reduction.noise_reduction_bilateral(frame)
    # You can choose Bilateral, Median, or Gaussian

    # --- Step 2: Contrast Enhancement ---
    #frame_contrast = contrast_adjustments.clahe_contrast_rgb(frame_denoised)
    # OR you can test: histogram_equalization_rgb, gamma_correction, etc.

    # --- Step 3: Sharpening (Optional) ---
    frame_sharp = sharpening.laplacian_sharpen(frame_denoised)

    # ---- Step 4: Static background Step ------
    frame_sharp_grey = cv2.cvtColor(frame_sharp, cv2.COLOR_BGR2GRAY)

    frame_sharp_grey = frame_grey

    # Background mask and detections
    mask = bg_sub.bgsub_guassian(frame_sharp_grey)
    detections = bg_sub.get_contour_detections(mask)
    cv2.imshow('Motion Mask', mask)

    # Run YOLO on full frame
    yolo_bboxes_full, yolo_scores_full = yolo_utils.get_yolo_detections(frame, conf_thresh=0.45)
    yolo_bboxes_roi, yolo_scores_roi = [], []
    detections = yolo_utils.filter_overlapping_detections(detections, overlap_thresh=0.6)
    detections = yolo_utils.filter_detections_with_yolo(detections, yolo_bboxes_full, yolo_scores_full, 0.35, 0.45)

    if len(detections) != 0:
        detections = np.array(detections, dtype=int)  # shape: [N, 5]
        frame_h, frame_w = frame.shape[:2]
        pad = 10

        # Vectorized padding and clipping
        x1s = np.clip(detections[:, 0] - pad, 0, frame_w)
        y1s = np.clip(detections[:, 1] - pad, 0, frame_h)
        x2s = np.clip(detections[:, 2] + pad, 0, frame_w)
        y2s = np.clip(detections[:, 3] + pad, 0, frame_h)

        for x1, y1, x2, y2 in zip(x1s, y1s, x2s, y2s):
            roi = frame[y1:y2, x1:x2]
            bbs, scores = yolo_utils.get_yolo_detections(roi, conf_thresh=0.15)

            if len(bbs):
                bbs = np.array(bbs)
                # Offset bbox coordinates to full frame
                bbs[:, [0, 2]] += x1
                bbs[:, [1, 3]] += y1

                yolo_bboxes_roi.extend(bbs.tolist())
                yolo_scores_roi.extend(scores)

    # Combine full + ROI detections
    if len(yolo_bboxes_full) != 0 and len(yolo_bboxes_roi) != 0:
        all_bboxes = np.vstack([yolo_bboxes_full, yolo_bboxes_roi])
        all_scores = np.concatenate([yolo_scores_full, yolo_scores_roi])
    elif len(yolo_bboxes_full) != 0:
        all_bboxes = yolo_bboxes_full
        all_scores = yolo_scores_full
    elif len(yolo_bboxes_roi)!= 0:
        all_bboxes = yolo_bboxes_roi
        all_scores = yolo_scores_roi
    else:
        all_bboxes = np.empty((0, 4))
        all_scores = np.empty((0,))

    
    # Convert lists to arrays for uniform handling
    yolo_bboxes_full = np.array(yolo_bboxes_full).reshape(-1, 4)
    yolo_scores_full = np.array(yolo_scores_full).reshape(-1)

    yolo_bboxes_roi = np.array(yolo_bboxes_roi).reshape(-1, 4)
    yolo_scores_roi = np.array(yolo_scores_roi).reshape(-1)

    # Stack if any detections exist
    all_bboxes = np.vstack([yolo_bboxes_full, yolo_bboxes_roi]) if yolo_bboxes_full.size or yolo_bboxes_roi.size else np.empty((0, 4))
    all_scores = np.concatenate([yolo_scores_full, yolo_scores_roi]) if yolo_scores_full.size or yolo_scores_roi.size else np.empty((0,))

    # Final NMS
    if len(all_bboxes):
        final_bboxes = yolo_utils.non_max_suppression(all_bboxes, all_scores, threshold=0.1)
        for box in final_bboxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Object", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    cv2.imshow('Detections', frame)
    # Break loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
