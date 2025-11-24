import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO

def calculate_average_direction(positions):
    """Calculate average direction vector from last 4 positions"""
    if len(positions) < 2:
        return None, None
    
    # Calculate direction vectors between consecutive positions
    directions = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        directions.append((dx, dy))
    
    # Calculate average direction
    if directions:
        avg_dx = sum(d[0] for d in directions) / len(directions)
        avg_dy = sum(d[1] for d in directions) / len(directions)
        return avg_dx, avg_dy
    return None, None

def match_person_to_track(current_pos, tracked_persons, max_distance=100):
    """Match current detection to existing tracked person"""
    if not tracked_persons:
        return None
    
    min_dist = float('inf')
    matched_id = None
    
    for person_id, positions in tracked_persons.items():
        if positions:
            last_pos = positions[-1]
            dist = np.sqrt((current_pos[0] - last_pos[0])**2 + (current_pos[1] - last_pos[1])**2)
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                matched_id = person_id
    
    return matched_id

def main():
    # Use standard YOLOv8 model for person detection (class 0 in COCO dataset)
    model = YOLO("yolov8m.pt")  # Change to yolov8s.pt or yolov8m.pt for better accuracy
    cap = cv2.VideoCapture(0)  # number of the webcam
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")  # if the webcam is not opened, raise an error

    # Dictionary to store position history for each person (max 4 positions)
    # Key: person_id, Value: deque of (x, y) positions
    tracked_persons = {}
    next_person_id = 0

    try:
        while True:  # while the webcam is opened, read the frame
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, imgsz=640, conf=0.5)  # detect the objects in the frame
            
            # Get boxes and filter for person class (class ID 0)
            boxes = results[0].boxes
            person_boxes = boxes[boxes.cls == 0]  # Filter for person class
            
            # Draw detections (this will draw all classes, but we'll overlay person centers)
            annotated = results[0].plot()
            
            # Current frame detections
            current_detections = []
            for box in person_boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate center coordinates
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                current_detections.append((center_x, center_y))
            
            # Match detections to tracked persons
            matched_ids = set()
            for center_x, center_y in current_detections:
                person_id = match_person_to_track((center_x, center_y), tracked_persons)
                
                if person_id is None:
                    # New person detected
                    person_id = next_person_id
                    next_person_id += 1
                    tracked_persons[person_id] = deque(maxlen=4)
                
                # Add current position to history
                tracked_persons[person_id].append((center_x, center_y))
                matched_ids.add(person_id)
            
            # Remove persons that are no longer detected
            tracked_persons = {pid: pos for pid, pos in tracked_persons.items() if pid in matched_ids}
            
            # Draw center points, coordinates, and direction vectors
            for person_id, positions in tracked_persons.items():
                if not positions:
                    continue
                
                current_x, current_y = positions[-1]
                
                # Draw center point
                cv2.circle(annotated, (current_x, current_y), 5, (0, 255, 0), -1)
                
                # Display center coordinates as text
                text = f"({current_x}, {current_y})"
                cv2.putText(annotated, text, (current_x + 10, current_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Calculate and draw average direction vector
                if len(positions) >= 2:
                    avg_dx, avg_dy = calculate_average_direction(list(positions))
                    
                    if avg_dx is not None and avg_dy is not None:
                        # Scale the vector for better visibility
                        scale = 50
                        end_x = int(current_x + avg_dx * scale)
                        end_y = int(current_y + avg_dy * scale)
                        
                        # Draw direction vector as arrow
                        cv2.arrowedLine(annotated, (current_x, current_y), 
                                       (end_x, end_y), (255, 0, 0), 3, tipLength=0.3)
                        
                        # Draw position history trail
                        for i in range(1, len(positions)):
                            cv2.line(annotated, positions[i-1], positions[i], 
                                    (0, 255, 255), 2)

            cv2.imshow("YOLOv8 Webcam", annotated)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()