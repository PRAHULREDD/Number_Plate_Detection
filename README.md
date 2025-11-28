# 🚗 Car Detection System

A professional car detection system using YOLO11n that detects cars crossing a detection line and saves cropped car images via FastAPI.

## 📁 Project Structure

```
car_detection/
├── 📂 object_detection/          # Core detection logic
│   ├── car_detector.py           # YOLO car detection
│   ├── video_handler.py          # Video processing pipeline
│   ├── fastapi_client.py         # API communication
│   └── main.py                   # Main application
├── 📂 server/                    # API server
│   └── api_server.py             # FastAPI server
├── 📂 models/                    # AI models
│   └── yolo11n.pt               # YOLO11n model
├── 📂 car_crossing_images/       # Saved car images
├── .env                          # Environment configuration
├── requirements.txt              # Python dependencies
├── start.bat                     # Quick launcher
└── test_video_*.mp4             # Test video files
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the System
```bash
# Option 1: Quick start
start.bat

# Option 2: Manual
cd server && python api_server.py
# In another terminal:
cd object_detection && python main.py
```

### 3. View Results
- **Gallery**: http://localhost:8000/gallery
- **Status**: http://localhost:8000/status

## ⚙️ Configuration

Edit `.env` file to customize settings:

```env
# Video Configuration
VIDEO_PATH=C:\path\to\your\video.mp4
MODEL_PATH=models/yolo11n.pt

# Detection Settings
DETECTION_LINE_POSITION=0.8        # Line position (80% from top)
CONFIDENCE_THRESHOLD=0.4           # Detection confidence
CROP_PADDING=20                    # Padding around cropped cars

# Display Settings
DISPLAY_WIDTH=800                  # Display window width
DISPLAY_HEIGHT=600                 # Display window height

# Performance
PROCESS_EVERY_N_FRAMES=2          # Process every 2nd frame
DUPLICATE_PREVENTION_TIME=2.0      # Seconds between same car
```

## 🎯 Features

- ✅ **Real-time car detection** with YOLO11n
- ✅ **Centroid-based tracking** prevents duplicates
- ✅ **Line crossing detection** with configurable position
- ✅ **Clean cropped images** without annotations
- ✅ **Car ID tracking** (Car 1, Car 2, etc.)
- ✅ **FPS display** on video
- ✅ **Web gallery** to view all detections
- ✅ **Non-blocking API** calls for smooth video

## 🖼️ Gallery Features

The web gallery shows:
- **Car ID**: "Car 1 - ⏱️ 24.72s"
- **Cropped images**: Only the detected car
- **Timestamps**: When each car crossed

## 🎮 Controls

- **Q**: Quit video processing

## 📝 Logs

Check `car_detection.log` for detailed processing logs.

## 📊 System Requirements

- Python 3.8+
- OpenCV, PyTorch, Ultralytics YOLO, FastAPI
- 4GB+ RAM recommended

---

**Professional Car Detection System**