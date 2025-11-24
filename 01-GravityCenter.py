import cv2
from ultralytics import YOLO

def main():
    # Use standard YOLOv8 model for person detection (class 0 in COCO dataset)
    model = YOLO("yolov8n.pt")  # Change to yolov8s.pt or yolov8m.pt for better accuracy
    cap = cv2.VideoCapture(0)  # number of the webcam
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")  # if the webcam is not opened, raise an error

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