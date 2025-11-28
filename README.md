# 🚗 Car Detection System with License Plate Recognition

A professional car detection system using YOLO11n and ByteTracker that detects cars crossing a detection line, recognizes license plates, and saves combined car and license plate images via FastAPI.

## 📁 Project Structure

```
car_detection/
├── 📂 object_detection/          # Core detection logic
│   ├── car_detector.py           # YOLO car detection with ByteTracker
│   ├── license_plate_detector.py # License plate detection
│   ├── video_handler.py          # Video processing pipeline
│   ├── fastapi_client.py         # API communication
│   └── main.py                   # Main application
├── 📂 server/                    # API server
│   └── api_server.py             # FastAPI server
├── 📂 models/                    # AI models
│   ├── yolo11n.pt               # YOLO11n car detection model
│   └── License_Plate_L1.pt      # License plate detection model
├── 📂 Videos/                    # Video files
├── .env                          # Environment configuration
├── requirements.txt              # Python dependencies
├── SETUP.md                      # Setup instructions
└── README.md                     # Documentation
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env file with your video path and settings
```

### 3. Run the System
```bash
# Terminal 1: Start API Server
cd server
python api_server.py

# Terminal 2: Start Car Detection
cd object_detection
python main.py
```

### 4. View Results
- **Gallery**: http://localhost:8000/gallery
- **Status**: http://localhost:8000/status

## ⚙️ Configuration

Edit `.env` file to customize settings:

```env
# Video Configuration
VIDEO_PATH=Videos\your_video.mp4
MODEL_PATH=models/yolo11n.pt
MODEL_NUMBER_PLATE_PATH=models\License_Plate_L1.pt

# Detection Settings
DETECTION_LINE_POSITION=0.8        # Line position (80% from top)
CONFIDENCE_THRESHOLD=0.4           # Car detection confidence
LP_CONFIDENCE_THRESHOLD=0.5        # License plate confidence
MIN_CAR_HEIGHT=500                 # Minimum car height for processing

# ByteTracker Settings
TRACK_HIGH_THRESH=0.5              # High confidence threshold
TRACK_LOW_THRESH=0.1               # Low confidence threshold
```

## 🎯 Features

- ✅ **Real-time car detection** with YOLO11n
- ✅ **ByteTracker integration** for stable car tracking
- ✅ **License plate detection** with dedicated AI model
- ✅ **Combined image output** showing car + license plate
- ✅ **Line crossing detection** with configurable position
- ✅ **Quality filtering** (minimum height validation)
- ✅ **One image per car ID** prevents duplicates
- ✅ **Sequential car numbering** (Car 1, Car 2, etc.)
- ✅ **FPS display** on video
- ✅ **Web gallery** to view all detections
- ✅ **Non-blocking API** calls for smooth video

## 🖼️ Gallery Features

The web gallery shows:
- **Car ID**: "Car 1 - ⏱️ 24.72s"
- **Combined images**: Car with license plate overlay (if detected)
- **License plate status**: Shows "No License Plate Detected" when none found
- **Timestamps**: When each car crossed the detection line

## 🎮 Controls

- **Q**: Quit video processing

## 📝 Logs

Check `car_detection.log` for detailed processing logs.

## 📊 System Requirements

- Python 3.8+
- OpenCV, PyTorch, Ultralytics YOLO, FastAPI
- 4GB+ RAM recommended

## 🛠️ Setup Instructions

See [SETUP.md](SETUP.md) for detailed installation and configuration instructions.

## 📁 Models Required

1. **yolo11n.pt** - Car detection model
2. **License_Plate_L1.pt** - License plate detection model

Place these models in the `models/` directory before running.

---

**Professional Car Detection System with License Plate Recognition**