#!/usr/bin/env python3
"""
Main Car Detection Application with direct .env configuration
"""

import logging
import sys
import os
from dotenv import load_dotenv
from video_handler import VideoHandler

# Load environment variables
load_dotenv()

def get_main_config():
    """Get main application configuration"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    config = {
        'video_path': os.getenv('VIDEO_PATH', 'Videos/test_video_1.mp4'),
        'model_path': os.getenv('MODEL_PATH', 'yolo11n.pt'),
        'log_file': os.getenv('LOG_FILE', 'car_detection.log'),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'debug': os.getenv('DEBUG', 'False').lower() == 'true',
        'verbose': os.getenv('VERBOSE', 'False').lower() == 'true'
    }
    
    # Make paths absolute if relative
    if not os.path.isabs(config['model_path']):
        config['model_path'] = os.path.join(project_root, config['model_path'])
    
    return config

def setup_logging():
    """Configure logging system"""
    config = get_main_config()
    log_file = config['log_file']
    log_level = config['log_level']
    debug = config['debug']
    
    log_level_obj = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        filename=log_file,
        level=log_level_obj,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    if debug:
        # Also log to console in debug mode
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level_obj)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logging.getLogger().addHandler(console_handler)

def main():
    """Main application entry point"""
    # Load configuration without exposing values
    config = get_main_config()
    video_path = config['video_path']
    model_path = config['model_path']
    debug = config['debug']
    verbose = config['verbose']
    
    print(" Car Detection System")
    
    if verbose or debug:
        print(f" Video: {video_path}")
        print(f" Model: {model_path}")
        print(f" Debug: {debug}")
    
    # Setup logging
    setup_logging()
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f" Error: Video file not found: {video_path}")
        return 1
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f" Error: Model file not found: {model_path}")
        return 1
    
    try:
        # Initialize video handler
        handler = VideoHandler()
        
        # Initialize
        if not handler.initialize():
            print(" Failed to initialize video handler")
            return 1
        
        # Start processing
        print(" Starting video processing... Press 'Q' to quit")
        handler.process_video()
        
        print(" Processing completed")
        return 0
        
    except KeyboardInterrupt:
        print("\  Interrupted by user")
        return 0
    except Exception as e:
        print(f" Error: {e}")
        logging.error(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())