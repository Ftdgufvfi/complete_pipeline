import cv2
import numpy as np
from ultralytics import YOLO
import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Count:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

model = YOLO("yolov10l.pt").to("cuda")

def get_yolo_detections(frame, conf_thresh=0.35):
    results = model.predict(frame, show=False, save=False, conf=conf_thresh)
    bboxes, scores = [], []
    allowed_classes = {0, 1, 2, 3, 5, 6, 7}  # Person & Vehicle classes

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])  # Class ID
            if cls in allowed_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0]) * 100
                bboxes.append([x1, y1, x2, y2])
                scores.append(confidence)
    return np.array(bboxes, dtype=np.float32), np.array(scores, dtype=np.float32)


#compute iou
def compute_iou(box1, box2):
    if box1 is None or len(box1)==0 or box2 is None or len(box2)==0 :
        return 0.0

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area != 0 else 0.0

# filter the detections
def compute_intersection_area(box1, box2):
    if box1 is None or len(box1) == 0 or box2 is None or len(box2) == 0:
        return 0
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def filter_overlapping_detections(detections, overlap_thresh=0.5):
    if detections is None or len(detections) == 0:
        return []

    # Compute area of a box
    def box_area(box):
        return max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    # Sort detections by area (largest first)
    detections = sorted(detections, key=lambda d: box_area(d[:4]), reverse=True)
    keep = []

    for i in range(len(detections)):
        current = detections[i]
        keep_current = True

        for kept in keep:
            inter_area = compute_intersection_area(current[:4], kept[:4])
            small_area = min(box_area(current[:4]), box_area(kept[:4]))
            if small_area == 0:
                continue  # avoid divide-by-zero
            if inter_area / small_area > overlap_thresh:
                keep_current = False
                break

        if keep_current:
            keep.append(current)

    return keep


# filter the detections
def filter_detections_with_yolo(detections, yolo_boxes, yolo_scores, iou_thresh=0.4, score_thresh=0.4):
    if len(detections) == 0 or len(yolo_boxes) == 0:
        return detections  # nothing to filter

    filtered = []
    for det in detections:
        det_box = det[:4]
        keep = True
        for yolo_box, score in zip(yolo_boxes, yolo_scores):
            if score >= score_thresh and compute_iou(det_box, yolo_box) > iou_thresh:
                keep = False
                break
        if keep:
            filtered.append(det)
    return filtered


def non_max_suppression(boxes, scores, threshold=0.1):
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)

    if len(boxes) == 0:
        return np.empty((0, 4), dtype=np.float32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        # Compute intersection
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        intersection = w * h

        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / (union + 1e-6)  # small epsilon to avoid div by 0

        # Keep boxes with IoU below threshold
        inds = np.where(iou < threshold)[0]
        order = order[inds + 1]  # +1 to skip i (current best box)

    return boxes[keep]

# utilites for optimized main.py
def compute_iou_matrix(boxes1, boxes2):
    boxes1 = torch.tensor(boxes1).float().unsqueeze(1)  # (N,1,4)
    boxes2 = torch.tensor(boxes2).float().unsqueeze(0)  # (1,M,4)

    xa = torch.max(boxes1[..., 0], boxes2[..., 0])
    ya = torch.max(boxes1[..., 1], boxes2[..., 1])
    xb = torch.min(boxes1[..., 2], boxes2[..., 2])
    yb = torch.min(boxes1[..., 3], boxes2[..., 3])

    inter = (xb - xa).clamp(0) * (yb - ya).clamp(0)
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union = area1 + area2 - inter + 1e-6

    iou = inter / union
    return iou.cpu().numpy()


def select_yolo_with_bg_iou_forced(yolo_boxes, yolo_scores, bg_boxes, frame,
                            forced_threshold=0.25, conf_thresh=0.45, iou_thresh=0.3):
    import numpy as np

    yolo_boxes = np.array(yolo_boxes, dtype=np.float32).reshape(-1, 4)
    yolo_scores = np.array(yolo_scores, dtype=np.float32).reshape(-1)
    bg_boxes = np.array(bg_boxes, dtype=np.float32).reshape(-1, 4)

    if yolo_boxes.shape[0] == 0:
        return np.empty((0, 4)), np.empty((0,))

    selected_boxes = []
    selected_scores = []

    # Step 1: Select high confidence boxes directly
    high_conf_mask = yolo_scores > conf_thresh
    selected_boxes.append(yolo_boxes[high_conf_mask])
    selected_scores.append(yolo_scores[high_conf_mask])

    # Step 2: For boxes between forced_threshold and conf_thresh, apply IoU check
    iou_mask = (yolo_scores <= conf_thresh) & (yolo_scores >= forced_threshold)
    boxes_iou = yolo_boxes[iou_mask]
    scores_iou = yolo_scores[iou_mask]

    if bg_boxes.shape[0] > 0 and boxes_iou.shape[0] > 0:
        iou_matrix = compute_iou_matrix(boxes_iou, bg_boxes)
        iou_max = iou_matrix.max(axis=1)
        iou_pass_mask = iou_max > iou_thresh

        selected_boxes.append(boxes_iou[iou_pass_mask])
        selected_scores.append(scores_iou[iou_pass_mask])

    # Step 3: For boxes below forced_threshold, re-run YOLO on ROI with low conf
    low_conf_mask = yolo_scores < forced_threshold
    low_conf_boxes = yolo_boxes[low_conf_mask]

    for box in low_conf_boxes:
        x1, y1, x2, y2 = box.astype(int)
        pad = 10
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)

        roi = frame[y1:y2, x1:x2]
        bbs, scores = get_yolo_detections(roi, conf_thresh=0.1)

        for i in range(len(bbs)):
            bbs[i][0] += x1
            bbs[i][1] += y1
            bbs[i][2] += x1
            bbs[i][3] += y1

        if len(bbs):
            selected_boxes.append(np.array(bbs))
            selected_scores.append(np.array(scores))

    # Final result
    if selected_boxes:
        final_boxes = np.vstack(selected_boxes)
        final_scores = np.concatenate(selected_scores)
    else:
        final_boxes = np.empty((0, 4))
        final_scores = np.empty((0,))

    return final_boxes, final_scores

def select_yolo_with_bg_iou(yolo_boxes, yolo_scores, bg_boxes, conf_thresh=0.45, iou_thresh=0.3):
    conf_thresh = 100*conf_thresh
    yolo_boxes = np.array(yolo_boxes, dtype=np.float32).reshape(-1, 4)
    yolo_scores = np.array(yolo_scores, dtype=np.float32).reshape(-1)
    bg_boxes = np.array(bg_boxes, dtype=np.float32).reshape(-1, 4)

    if yolo_boxes.shape[0] == 0:
        return np.empty((0, 4)), np.empty((0,))

    # Step 1: Pick high-confidence YOLO predictions directly
    high_conf_mask = yolo_scores > conf_thresh
    selected_boxes = yolo_boxes[high_conf_mask]
    selected_scores = yolo_scores[high_conf_mask]

    #Step 2: For remaining YOLO boxes, check IoU with bg_boxes
    remaining_mask = ~high_conf_mask
    remaining_boxes = yolo_boxes[remaining_mask]
    remaining_scores = yolo_scores[remaining_mask]

    if bg_boxes.shape[0] > 0 and remaining_boxes.shape[0] > 0:
        iou_matrix = compute_iou_matrix(remaining_boxes, bg_boxes)
        iou_max = iou_matrix.max(axis=1)

        iou_pass_mask = iou_max > iou_thresh
        selected_boxes = np.vstack([selected_boxes, remaining_boxes[iou_pass_mask]])
        selected_scores = np.concatenate([selected_scores, remaining_scores[iou_pass_mask]])

    return selected_boxes, selected_scores


# filter the detections
def filter_detections(detections, frame_shape=(480, 640)):

    if len(detections) == 0:
        return np.empty((0, 4), dtype=np.float32)

    detections = np.array(detections, dtype=np.float32)

    # Handle both (N, 5) and (N, 4) shapes
    if detections.shape[1] == 5:
        detections = detections[:, :4]  # drop score column

    height, width = frame_shape
    zone_h, zone_w = height // 3, width // 2

    centers_x = (detections[:, 0] + detections[:, 2]) / 2
    centers_y = (detections[:, 1] + detections[:, 3]) / 2

    zone_rows = (centers_y // zone_h).astype(int)
    zone_cols = (centers_x // zone_w).astype(int)
    zone_ids = zone_rows * 3 + zone_cols
    zone_ids = np.clip(zone_ids, 0, 8)

    selected_zones = {}
    for i, zone in enumerate(zone_ids):
        if zone not in selected_zones:
            selected_zones[zone] = detections[i]

    return np.array(list(selected_zones.values()), dtype=np.float32)


