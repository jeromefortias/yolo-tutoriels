import cv2
import numpy as np
import json
import requests
from datetime import datetime
from ultralytics import YOLO

ES_URL = "http://localhost:9200"
ES_INDEX = "visiondetect"

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def check_person_overlap(person_boxes):
    """Check if any person boxes overlap"""
    boxes_list = []
    for box in person_boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        boxes_list.append((x1, y1, x2, y2))
    
    for i in range(len(boxes_list)):
        for j in range(i + 1, len(boxes_list)):
            iou = calculate_iou(boxes_list[i], boxes_list[j])
            if iou > 0:
                return True
    return False

def ensure_index():
    """Ensure Elasticsearch index exists"""
    try:
        resp = requests.put(f"{ES_URL}/{ES_INDEX}", timeout=2)
        if resp.status_code not in (200, 400):
            resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[ES] failed to ensure index: {exc}")

def save_to_elasticsearch(doc):
    """Index detection doc into Elasticsearch"""
    try:
        resp = requests.post(
            f"{ES_URL}/{ES_INDEX}/_doc",
            json=doc,
            timeout=2,
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[ES] failed to index doc: {exc}")

def save_detection_to_json(box, model, detection_counter):
    """Save a single detection to a JSON file and Elasticsearch"""
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    class_id = int(box.cls[0])
    object_name = model.names[class_id]
    surface = float((x2 - x1) * (y2 - y1))
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    detection_data = {
        "objectname": object_name,
        "xstart": float(x1),
        "ystart": float(y1),
        "xend": float(x2),
        "yend": float(y2),
        "surface": surface,
        "date": date_now,
    }
    
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{datetime_str}_{detection_counter:04d}.json"
    
    
    save_to_elasticsearch(detection_data)
    return filename

def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    ensure_index()
    detection_counter = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, imgsz=640, conf=0.5)
            boxes = results[0].boxes
            person_boxes = boxes[boxes.cls == 0]
            
            for box in boxes:
                detection_counter += 1
                filename = save_detection_to_json(box, model, detection_counter)
                print(f"Saved detection to {filename}")
            
            annotated = results[0].plot()
            has_overlap = check_person_overlap(person_boxes)
            
            if has_overlap:
                h, w = annotated.shape[:2]
                text = "ALERT ASPERGER"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = (w - text_width) // 2
                text_y = (h + text_height) // 2
                cv2.putText(annotated, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
            
            for box in person_boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.circle(annotated, (center_x, center_y), 5, (0, 255, 0), -1)
                text = f"({center_x}, {center_y})"
                cv2.putText(annotated, text, (center_x + 10, center_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Webcam", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()