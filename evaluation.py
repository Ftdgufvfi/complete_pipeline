import os
from PIL import Image

# Define your directories
images_dir = "Datasets/train/images"
labels_dir = "Datasets/train/labels"  # YOLO labels
image_extensions = [".jpg", ".jpeg", ".png"]

# Function to load annotations for one image
def load_yolo_labels(label_path, img_width, img_height):
    boxes = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(parts[0])
            x_min = float(parts[1]) * img_width
            y_min = float(parts[2]) * img_height
            w = float(parts[3]) * img_width
            h = float(parts[4]) * img_height
            x_max = x_min + w
            y_max =
            boxes.append({
                "class_id": class_id,
                "bbox": [x_min, y_min, w, h],
            })
    return boxes

# Load data
data = []

for file in os.listdir(images_dir):
    image_path = os.path.join(images_dir, file)
    label_path = os.path.join(labels_dir, os.path.splitext(file)[0] + ".txt")

    img = Image.open(image_path)
    img_width, img_height = img.size

    # If label exists, load it
    if os.path.exists(label_path):
        boxes = load_yolo_labels(label_path, 640, 480)
    else:
        boxes = []

    data.append({
        "image_path": image_path,
        "image_size": (img_width, img_height),
        "boxes": boxes  # list of dicts with class_id and bbox
    })

# Example: Print one entry
import pprint
pprint.pprint(data[0])


import cv2
import noise_reduction
import contrast_adjustments
import sharpening
import static_background_sub as bg_sub
import yolo_utilities as yolo_utils
import numpy as np
#import swanet_utils
import zero_dce_utils



import os
import cv2
import numpy as np
from PIL import Image

# Set directories
images_dir = "Datasets/train/images"
image_files = [f for f in os.listdir(images_dir)]

predictions = []

for img_file in image_files:
    frame = cv2.imread(os.path.join(images_dir, img_file))
    if frame is None:
        print(f"Failed to load {img_file}")
        continue

    frame = cv2.resize(frame, (640, 480))
    
    # === Preprocessing ===
    frame_enhanced = zero_dce_utils.enhance_image(frame)
    frame_grey = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2GRAY)

    # === Background Subtraction ===
    mask = bg_sub.bgsub_guassian(frame_grey)
    detections = bg_sub.get_contour_detections(mask)

    # === YOLO Detections ===
    yolo_bboxes_full, yolo_scores_full = yolo_utils.get_yolo_detections(frame_enhanced, conf_thresh=0.15)

    # === Fusion ===
    bg_boxes_coords = np.array([det[:4] for det in detections], dtype=np.float32)
    all_bboxes, all_scores = yolo_utils.select_yolo_with_bg_iou(
        yolo_bboxes_full, yolo_scores_full, bg_boxes_coords,
        conf_thresh=0.38, iou_thresh=0.7
    )

    final_results = []
    if len(all_bboxes):
        final_bboxes = yolo_utils.non_max_suppression(all_bboxes, all_scores, threshold=0.1)
        for i, box in enumerate(final_bboxes):
            x1, y1, x2, y2 = map(int, box)
            score = float(all_scores[i])
            final_results.append({
                "bbox": [x1, y1, x2, y2],
                "score": score
            })

    predictions.append({
        "image_name": img_file,
        "detections": final_results
    })

print("Prediction complete for all images.")

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2

    xi1 = max(x1, x1g)
    yi1 = max(y1, y1g)
    xi2 = min(x2, x2g)
    yi2 = min(y2, y2g)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

total_tp = 0
total_fp = 0
total_fn = 0

for gt_entry, pred_entry in zip(data, predictions):
    gt_boxes = [box["bbox"] for box in gt_entry["boxes"]]
    pred_boxes = [det["bbox"] for det in pred_entry["detections"]]

    matched = set()
    for pb in pred_boxes:
        ious = [compute_iou(pb, gt_box) for gt_box in gt_boxes]
        max_iou = max(ious) if ious else 0
        if max_iou >= 0.5:
            gt_idx = ious.index(max_iou)
            if gt_idx not in matched:
                matched.add(gt_idx)
                total_tp += 1
            else:
                total_fp += 1  # duplicate detection
        else:
            total_fp += 1  # no matching GT

    total_fn += len(gt_boxes) - len(matched)

precision = total_tp / (total_tp + total_fp + 1e-6)
recall = total_tp / (total_tp + total_fn + 1e-6)

print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")

