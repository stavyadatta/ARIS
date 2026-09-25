import cv2
from ultralytics import YOLOWorld

class _YOLODetector:
    def __init__(self, model_path='yolov8s-world.pt'):
        # Initialize the YOLO-World model
        self.model = YOLOWorld(model_path)
        self.classes = []

    def set_classes(self, classes):
        # Set the classes to detect
        self.classes = classes
        self.model.set_classes(classes)

    def detect_objects(self, image):
        # Read the image
        if image is None:
            raise ValueError(f"The image is empty")

        # Perform inference
        results = self.model.predict(image)

        # Extract bounding boxes for the specified classes
        bounding_boxes = []
        for result in results:
            for obj in result.boxes:
                class_id = int(obj.cls[0])
                class_name = self.classes[class_id]
                confidence = obj.conf[0]
                if confidence > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = map(int, obj.xyxy[0])
                    bounding_boxes.append({
                        'class': class_name,
                        'confidence': float(confidence),
                        'bounding_box': (x1, y1, x2, y2)
                    })

        return bounding_boxes
