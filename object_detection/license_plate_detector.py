import cv2
import os
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()

class LicensePlateDetector:
    def __init__(self):
        self.model = None
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.getenv('MODEL_NUMBER_PLATE_PATH', 'models\License_Plate_L1.pt')
        if not os.path.isabs(model_path):
            self.model_path = os.path.join(project_root, model_path)
        else:
            self.model_path = model_path
        self.confidence_threshold = float(os.getenv('LP_CONFIDENCE_THRESHOLD', 0.3))
        
    def load_model(self):
        """Load license plate detection model"""
        try:
            self.model = YOLO(self.model_path)
            return True
        except Exception as e:
            print(f"Error loading license plate model: {e}")
            return False
    
    def detect_license_plate(self, car_image):
        """Detect license plate in car image and return cropped plate"""
        if self.model is None:
            return None
            
        results = self.model(source=car_image, conf=self.confidence_threshold, verbose=False)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    if class_name == 'License_Plate':
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        license_plate = car_image[y1:y2, x1:x2]
                        return license_plate
        
        return None