import cv2
import numpy as np
import torch
from ultralytics import YOLO

class _PersonDetectorCropper:
    def __init__(self, 
                 model_path='yolov8n.pt',  # Default to Ultralytics nano model
                 confidence_threshold=0.5,
                 device='auto'):
        """
        Initialize the Person Detector and Cropper.
        
        :param model_path: Path to YOLO model weights
        :param confidence_threshold: Minimum confidence for detection
        :param device: Device to run model on ('cpu', 'cuda', or 'auto')
        """
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load the YOLO model
        self.model = YOLO(model_path).to(self.device)
        
        # Set confidence threshold
        self.confidence_threshold = confidence_threshold
    
    def detect_and_crop_person(self, image, return_all=False):
        """
        Detect and crop person(s) from the image.
        
        :param image: Input image as NumPy array
        :param return_all: If True, return all detected persons. 
                            If False, return the largest (most confident) person
        :return: Cropped person image(s) as NumPy array
        """
        # Ensure image is NumPy array
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a NumPy array")
        
        # Detect persons
        results = self.model(image, 
                             conf=self.confidence_threshold, 
                             classes=[0],  # Person class
                             device=self.device)
        
        # Extract detected persons
        person_crops = []
        for result in results:
            # Get bounding boxes for persons
            boxes = result.boxes
            
            # Filter for persons
            person_boxes = boxes[boxes.cls == 0]
            
            # If no persons detected
            if len(person_boxes) == 0:
                return None
            
            # Extract crops
            for box in person_boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Crop the person
                person_crop = image[y1:y2, x1:x2]
                person_crops.append({
                    'crop': person_crop,
                    'confidence': float(box.conf),
                    'coordinates': (x1, y1, x2, y2)
                })
        
        # Sort by confidence if multiple persons detected
        person_crops.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Return based on return_all flag
        if return_all:
            return [crop['crop'] for crop in person_crops]
        else:
            return person_crops[0]['crop']
    
    def draw_detections(self, image):
        """
        Draw bounding boxes on detected persons.
        
        :param image: Input image as NumPy array
        :return: Image with bounding boxes drawn
        """
        # Create a copy of the image to draw on
        draw_image = image.copy()
        
        # Detect persons
        results = self.model(image, 
                             conf=self.confidence_threshold, 
                             classes=[0],  # Person class
                             device=self.device)
        
        # Draw bounding boxes
        for result in results:
            boxes = result.boxes
            
            # Filter for persons
            person_boxes = boxes[boxes.cls == 0]
            
            for box in person_boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf)
                
                # Draw rectangle
                cv2.rectangle(draw_image, (x1, y1), (x2, y2), 
                              color=(0, 255, 0), thickness=2)
                
                # Add confidence label
                label = f"Person: {confidence:.2f}"
                cv2.putText(draw_image, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                            (0, 255, 0), 2)
        
        return draw_image

# Example usage
if __name__ == "__main__":
    import cv2
    
    # Initialize the detector
    detector = _PersonDetectorCropper(
        model_path='yolov8n.pt',  # You can change to other YOLO variants
        confidence_threshold=0.5
    )
    
    # Read an image
    image = cv2.imread("/workspace/database/face_db/some.png")
    
    # Detect and crop the person
    person_crop = detector.detect_and_crop_person(image)
    
    # Draw detections on the original image
    detection_image = detector.draw_detections(image)
    
    # Display results
    if person_crop is not None:
        cv2.imwrite("./sample.png", detection_image)
        cv2.imwrite("./crop_sample.png", person_crop)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No person detected")
