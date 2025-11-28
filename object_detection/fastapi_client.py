import requests
import base64
import cv2
import logging
import os
import threading
import numpy as np
from dotenv import load_dotenv
from license_plate_detector import LicensePlateDetector

load_dotenv()

def get_api_config():
    """Get API configuration"""
    return {
        'api_url': os.getenv('FASTAPI_URL', 'http://localhost:8000/car-crossing'),
        'debug': os.getenv('DEBUG', 'False').lower() == 'true'
    }

class FastAPIClient:
    def __init__(self):
        config = get_api_config()
        self.api_url = config['api_url']
        self.debug = config['debug']
        self.lp_detector = LicensePlateDetector()
        self.lp_detector.load_model()
    
    def send_crossing_image(self, frame, timestamp, car_id=None):
        """Send car image with license plate detection in background thread"""
        def _send_async():
            try:
                # Detect license plate
                license_plate = self.lp_detector.detect_license_plate(frame)
                
                # Create combined image
                combined_image = self._create_combined_view(frame, license_plate)
                
                _, buffer = cv2.imencode('.jpg', combined_image)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                data = {
                    "image": img_base64, 
                    "timestamp": timestamp,
                    "car_id": car_id,
                    "has_license_plate": license_plate is not None
                }
                
                response = requests.post(self.api_url, json=data, timeout=3)
                
                if response.status_code == 200:
                    lp_status = "with license plate" if license_plate is not None else "no license plate"
                    logging.info(f"Car {car_id} image sent successfully at {timestamp:.2f}s ({lp_status})")
                else:
                    logging.error(f"Failed to send image: {response.status_code}")
                    
            except Exception as e:
                logging.error(f"Error sending to FastAPI: {e}")
        
        thread = threading.Thread(target=_send_async, daemon=True)
        thread.start()
        return True
    def _create_combined_view(self, car_image, license_plate):
        """Create combined view of car and license plate"""
        if license_plate is None:
            # Just return car image with "No License Plate" text
            result = car_image.copy()
            cv2.putText(result, "No License Plate Detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return result
        
        # Resize license plate to reasonable size
        lp_height = min(100, car_image.shape[0] // 3)
        lp_aspect = license_plate.shape[1] / license_plate.shape[0]
        lp_width = int(lp_height * lp_aspect)
        license_plate_resized = cv2.resize(license_plate, (lp_width, lp_height))
        
        # Create combined image
        result = car_image.copy()
        
        # Add license plate in top-right corner
        y_offset = 10
        x_offset = car_image.shape[1] - lp_width - 10
        
        # Ensure license plate fits in image
        if x_offset > 0 and y_offset + lp_height < car_image.shape[0]:
            result[y_offset:y_offset+lp_height, x_offset:x_offset+lp_width] = license_plate_resized
            
            # Add border around license plate
            cv2.rectangle(result, (x_offset-2, y_offset-2), 
                         (x_offset+lp_width+2, y_offset+lp_height+2), (0, 255, 0), 2)
            
            # Add label
            cv2.putText(result, "License Plate", (x_offset, y_offset-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return result