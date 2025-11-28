import cv2
import logging
import os
from dotenv import load_dotenv
from ultralytics import YOLO

# Load environment variables
load_dotenv()

def get_env_config():
    """Centralized environment configuration loader"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config = {
        'model_path': os.getenv('MODEL_PATH', 'yolo11n.pt'),
        'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', 0.4)),
        'yolo_image_size': int(os.getenv('YOLO_IMAGE_SIZE', 640)),
        'max_tracked_cars': int(os.getenv('MAX_TRACKED_CARS', 30)),
        'duplicate_prevention_time': float(os.getenv('DUPLICATE_PREVENTION_TIME', 2.0)),
        'centroid_distance_threshold': int(os.getenv('CENTROID_DISTANCE_THRESHOLD', 100)),
        'min_crossing_distance': int(os.getenv('MIN_CROSSING_DISTANCE', 50)),
        'crop_padding': int(os.getenv('CROP_PADDING', 20)),
        'min_car_height': int(os.getenv('MIN_CAR_HEIGHT', 500)),
        'verbose': os.getenv('VERBOSE', 'False').lower() == 'true'
    }
    
    # Make model path absolute if relative
    if not os.path.isabs(config['model_path']):
        config['model_path'] = os.path.join(project_root, config['model_path'])
    
    return config

class CarDetector:
    def __init__(self):
        self.model = None
        self.crossed_cars = {}
        self.last_detections = []
        self.car_counter = 0
        self.tracked_cars = {}
        
        # Load settings from centralized config
        config = get_env_config()
        self.model_path = config['model_path']
        self.confidence_threshold = config['confidence_threshold']
        self.yolo_image_size = config['yolo_image_size']
        self.max_tracked_cars = config['max_tracked_cars']
        self.duplicate_prevention_time = config['duplicate_prevention_time']
        self.centroid_distance_threshold = config['centroid_distance_threshold']
        self.min_crossing_distance = config['min_crossing_distance']
        self.crop_padding = config['crop_padding']
        self.min_car_height = config['min_car_height']
        self.verbose = config['verbose']
        
    def load_model(self):
        """Load and optimize YOLO model"""
        try:
            if self.verbose:
                print("Loading YOLO11n model...")
            self.model = YOLO(self.model_path)
            self.model.fuse()
            if self.verbose:
                print("YOLO11n model loaded successfully.")
            logging.info("YOLO11n model loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            logging.error(f"Error loading model: {e}")
            return False
    
    def detect_cars(self, frame):
        """Detect cars in frame and return detections"""
        if self.model is None:
            return []
            
        results = self.model.track(source=frame, imgsz=self.yolo_image_size, conf=self.confidence_threshold, verbose=self.verbose, tracker="bytetrack.yaml")
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    if class_name == 'car':
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        track_id = int(box.id[0]) if box.id is not None else None
                        detections.append((x1, y1, x2, y2, conf, track_id))
        
        return detections
    
    def check_line_crossing(self, detections, line_y, timestamp):
        """Check if any car's centroid crossed the detection line"""
        crossings = []
        current_time = timestamp
        
        # Clean old entries
        self.crossed_cars = {k: v for k, v in self.crossed_cars.items() 
                           if current_time - v < self.duplicate_prevention_time}
        
        # Get cars with IDs
        cars_with_ids = self.assign_car_ids(detections)
        
        for x1, y1, x2, y2, conf, car_id, centroid_x, centroid_y in cars_with_ids:
            # Check if car's centroid crossed the line
            if centroid_y >= line_y:
                # Check if this car hasn't crossed recently
                if car_id not in self.crossed_cars:
                    self.crossed_cars[car_id] = current_time
                    crossings.append((x1, y1, x2, y2, conf, timestamp, car_id))
                    
                    if self.verbose:
                        print(f"Car {car_id} crossed line at centroid ({centroid_x}, {centroid_y})")
        
        return crossings
    
    def assign_car_ids(self, detections):
        """Assign unique IDs to detected cars"""
        current_cars = []
        
        for x1, y1, x2, y2, conf, track_id in detections:
            centroid_x = (x1 + x2) // 2
            centroid_y = (y1 + y2) // 2
            
            # Use ByteTracker ID or fallback to manual tracking
            if track_id is not None:
                car_id = track_id
            else:
                car_id = None
                min_distance = float('inf')
                
                for existing_id, (ex_x, ex_y) in self.tracked_cars.items():
                    distance = ((centroid_x - ex_x) ** 2 + (centroid_y - ex_y) ** 2) ** 0.5
                    if distance < self.centroid_distance_threshold and distance < min_distance:
                        min_distance = distance
                        car_id = existing_id
                
                if car_id is None:
                    self.car_counter += 1
                    car_id = self.car_counter
            
            # Update car position
            self.tracked_cars[car_id] = (centroid_x, centroid_y)
            current_cars.append((x1, y1, x2, y2, conf, car_id, centroid_x, centroid_y))
        
        # Clean up old cars not seen in current frame
        current_ids = [car[5] for car in current_cars]
        self.tracked_cars = {k: v for k, v in self.tracked_cars.items() if k in current_ids}
        
        return current_cars
    
    def draw_detections(self, frame, detections, line_y):
        """Draw bounding boxes with car IDs and detection line on frame"""
        # Draw detection line
        cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 3)
        
        # Assign car IDs
        cars_with_ids = self.assign_car_ids(detections)
        
        # Draw bounding boxes with car IDs
        for x1, y1, x2, y2, conf, car_id, cx, cy in cars_with_ids:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Car {car_id} ({conf:.2f})", (x1, y1-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
            
            # Draw centroid
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        
        return frame
    
    def crop_car(self, frame, x1, y1, x2, y2):
        """Crop car from frame with padding and dimension validation"""
        h, w = frame.shape[:2]
        
        # Add padding and ensure within frame bounds
        x1 = max(0, x1 - self.crop_padding)
        y1 = max(0, y1 - self.crop_padding)
        x2 = min(w, x2 + self.crop_padding)
        y2 = min(h, y2 + self.crop_padding)
        
        # Check if cropped dimensions meet minimum requirements
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        if crop_height < self.min_car_height:
            return None  # Reject low-quality crops
        
        # Crop the car region
        cropped_car = frame[y1:y2, x1:x2]
        
        return cropped_car