# Setup Instructions

## Prerequisites
- Python 3.8+
- Git

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd car_detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env file with your settings
   ```

5. **Add your files**
   - Place YOLO models in `models/` folder
   - Place video files in `Videos/` folder

## Running the System

1. **Start API Server**
   ```bash
   cd server
   python api_server.py
   ```

2. **Start Car Detection** (in new terminal)
   ```bash
   cd object_detection
   python main.py
   ```

3. **View Results**
   - Gallery: http://localhost:8000/gallery
   - Status: http://localhost:8000/status

## Configuration

Edit `.env` file to customize:
- Video path
- Model paths
- Detection thresholds
- API settings