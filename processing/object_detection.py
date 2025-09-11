from ultralytics import YOLO
import cv2

class ObjectDetector:
    def __init__(self, model_path="models/yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]
        objects = []
        for result in results.boxes:
            x1, y1, x2, y2 = result.xyxy[0]
            label = self.model.names[int(result.cls[0])]
            objects.append({"bbox": (int(x1), int(y1), int(x2), int(y2)), "label": label})
        return objects

    def draw_boxes(self, frame, objects):
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, obj["label"], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame
