import cv2
import logging
import os
from dotenv import load_dotenv
from car_detector import CarDetector, get_env_config
from fastapi_client import FastAPIClient

load_dotenv()

def get_video_config():
    """Get video-specific configuration"""
    return {
        'video_path': os.getenv('VIDEO_PATH', 'Videos/test_video_1.mp4'),
        'detection_line_position': float(os.getenv('DETECTION_LINE_POSITION', 0.8)),
        'process_every_n_frames': int(os.getenv('PROCESS_EVERY_N_FRAMES', 2)),
        'video_display_delay': int(os.getenv('VIDEO_DISPLAY_DELAY', 30)),
        'display_width': int(os.getenv('DISPLAY_WIDTH', 800)),
        'display_height': int(os.getenv('DISPLAY_HEIGHT', 600))
    }

class VideoHandler:
    def __init__(self):
        self.detector = CarDetector()
        self.api_client = FastAPIClient()
        self.cap = None
        
        # Load configuration without exposing values
        config = get_video_config()
        self.video_path = config['video_path']
        self.detection_line_position = config['detection_line_position']
        self.process_every_n_frames = config['process_every_n_frames']
        self.video_display_delay = config['video_display_delay']
        self.display_width = config['display_width']
        self.display_height = config['display_height']
        
    def initialize(self):
        if not self.detector.load_model():
            return False
            
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video file {self.video_path}")
            return False
            
        print("Video opened successfully. Processing...")
        return True
    
    def process_video(self):
        if self.cap is None:
            return
            
        frame_count = 0
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_y = int(frame_height * self.detection_line_position)
        
        # FPS calculation variables
        import time
        prev_time = time.time()
        fps_counter = 0
        display_fps = 0
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame_count += 1
            timestamp = frame_count / fps
            
            # Keep original clean frame for cropping
            original_frame = frame.copy()
            
            if frame_count % self.process_every_n_frames == 0:
                detections = self.detector.detect_cars(frame)
                self.detector.last_detections = detections
                
                crossings = self.detector.check_line_crossing(detections, line_y, timestamp)
                
                for crossing in crossings:
                    x1, y1, x2, y2, conf, ts, car_id = crossing
                    logging.info(f"Car {car_id} crossed line at timestamp: {ts:.2f}s")
                    
                    # Crop from original clean frame (no bboxes/lines)
                    cropped_car = self.detector.crop_car(original_frame, x1, y1, x2, y2)
                    
                    # Only send if crop meets quality requirements
                    if cropped_car is not None:
                        self.api_client.send_crossing_image(cropped_car, ts, car_id)
                    else:
                        logging.info(f"Car {car_id} rejected - image too small")
            
            # Draw detections on display frame only
            frame = self.detector.draw_detections(frame, self.detector.last_detections, line_y)
            
            # Calculate and display FPS
            fps_counter += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:  # Update FPS every second
                display_fps = fps_counter
                fps_counter = 0
                prev_time = current_time
            
            # Draw FPS on frame
            cv2.putText(frame, f"FPS: {display_fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Resize frame for display
            display_frame = cv2.resize(frame, (self.display_width, self.display_height))
            cv2.imshow('Car Detection', display_frame)
            
            if cv2.waitKey(self.video_display_delay) & 0xFF == ord('q'):
                break
        
        self.cleanup()
    
    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Video processing finished.")