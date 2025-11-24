import cv2
import numpy as np
import json
from datetime import datetime
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    # box format: [x1, y1, x2, y2]
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
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
    
    # Check all pairs of boxes
    for i in range(len(boxes_list)):
        for j in range(i + 1, len(boxes_list)):
            iou = calculate_iou(boxes_list[i], boxes_list[j])
            if iou > 0:  # Any overlap detected
                return True
    return False

def save_detection_to_json(box, model, detection_counter):
    """Save a single detection to a JSON file"""
    # Get bounding box coordinates
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    
    # Get class ID and name
    class_id = int(box.cls[0])
    object_name = model.names[class_id]
    
    # Calculate surface (area of the box)
    surface = float((x2 - x1) * (y2 - y1))
    
    # Get current datetime
    date_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    # Create JSON data
    detection_data = {
        "objectname": object_name,
        "xstart": float(x1),
        "ystart": float(y1),
        "xend": float(x2),
        "yend": float(y2),
        "surface": surface,
        "date": date_now
    }
    
    # Create filename with datetime and increment
    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{datetime_str}_{detection_counter:04d}.json"
    
    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(detection_data, f, indent=2)
    
    return filename

def main():
    # Use standard YOLOv8 model for person detection (class 0 in COCO dataset)
    model = YOLO("yolov8n.pt")  # Change to yolov8s.pt or yolov8m.pt for better accuracy
    cap = cv2.VideoCapture(0)  # number of the webcam
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")  # if the webcam is not opened, raise an error

    detection_counter = 0  # Counter for increment in filename

    try:
        while True:  # while the webcam is opened, read the frame
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, imgsz=640, conf=0.5)  # detect the objects in the frame
            
            # Get boxes and filter for person class (class ID 0)
            boxes = results[0].boxes
            person_boxes = boxes[boxes.cls == 0]  # Filter for person class
            
            # Save all detections to individual JSON files
            for box in boxes:
                detection_counter += 1
                filename = save_detection_to_json(box, model, detection_counter)
                print(f"Saved detection to {filename}")
            
            # Draw detections (this will draw all classes, but we'll overlay person centers)
            annotated = results[0].plot()
            
            # Check for overlapping person boxes
            has_overlap = check_person_overlap(person_boxes)
            
            # Display alert if overlap detected
            if has_overlap:
                # Get frame dimensions for text positioning
                h, w = annotated.shape[:2]
                text = "ALERT ASPERGER"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                
                # Get text size for centering
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = (w - text_width) // 2
                text_y = (h + text_height) // 2
                
                # Draw text with red color (BGR format: (0, 0, 255))
                cv2.putText(annotated, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness)
            
            # Draw center points and coordinates for each person
            for box in person_boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate center coordinates
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Draw center point
                cv2.circle(annotated, (center_x, center_y), 5, (0, 255, 0), -1)
                
                # Display center coordinates as text
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