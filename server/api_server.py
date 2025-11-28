#!/usr/bin/env python3
"""
FastAPI Server with direct .env configuration
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import os
import uvicorn
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load settings from .env
FASTAPI_HOST = os.getenv('FASTAPI_HOST', '127.0.0.1')
FASTAPI_PORT = int(os.getenv('FASTAPI_PORT', 8000))
IMAGES_FOLDER = os.getenv('IMAGES_FOLDER', 'car_crossing_images')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
VERBOSE = os.getenv('VERBOSE', 'False').lower() == 'true'

# Make images folder path absolute from project root
if not os.path.isabs(IMAGES_FOLDER):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    IMAGES_FOLDER = os.path.join(project_root, IMAGES_FOLDER)

# Initialize FastAPI app
app = FastAPI(
    title="Car Detection API",
    version="1.0.0",
    description="Car Detection System API"
)

# Create images folder
os.makedirs(IMAGES_FOLDER, exist_ok=True)

# Global counter
image_count = 0

class ImageData(BaseModel):
    image: str
    timestamp: float
    car_id: int = None
    has_license_plate: bool = False

@app.get("/")
def root():
    return {
        "message": "Car Detection API Server",
        "version": "1.0.0",
        "status": "running",
        "debug_mode": DEBUG
    }

@app.get("/status")
def get_status():
    """Get server status and statistics"""
    image_files = [f for f in os.listdir(IMAGES_FOLDER) if f.endswith('.jpg')]
    return {
        "server_status": "running",
        "total_images_received": image_count,
        "images_in_folder": len(image_files),
        "latest_images": sorted(image_files)[-5:] if image_files else []
    }

@app.get("/images")
def list_images():
    """List all saved images"""
    image_files = [f for f in os.listdir(IMAGES_FOLDER) if f.endswith('.jpg')]
    return {"images": sorted(image_files, reverse=True)}

@app.get("/images/{filename}")
def get_image(filename: str):
    """Serve individual image file"""
    file_path = os.path.join(IMAGES_FOLDER, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "Image not found"}

@app.get("/gallery", response_class=HTMLResponse)
def image_gallery():
    """HTML gallery to view all images"""
    image_files = [f for f in os.listdir(IMAGES_FOLDER) if f.endswith('.jpg')]
    image_files = sorted(image_files, reverse=True)
    
    html_content = f"""
    <html>
    <head>
        <title> Car Detection Gallery</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .header {{ text-align: center; margin-bottom: 30px; background: white; padding: 20px; border-radius: 10px; }}
            .stats {{ background: #4CAF50; color: white; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
            .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
            .image-card {{ background: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .image-card img {{ width: 100%; border-radius: 8px; }}
            .image-info {{ margin-top: 10px; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1> Car Detection Gallery</h1>
            <div class="stats">
                 Total Images: {len(image_files)} | 
                 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </div>
        </div>
        <div class="gallery">
    """
    
    for img in image_files:
        try:
            # Extract car ID and timestamp from filename
            # Format: car1_20241127_161408_24.72s.jpg
            parts = img.split('_')
            car_id = parts[0].replace('car', 'Car ')  # car1 -> Car 1
            timestamp_part = parts[-1].replace('s.jpg', '')
            timestamp = f" {timestamp_part}s"
            display_name = f"{car_id} - {timestamp}"
        except:
            car_id = "Unknown Car"
            timestamp = " Unknown"
            display_name = img
            
        html_content += f"""
        <div class="image-card">
            <img src="/images/{img}" alt="{img}" />
            <div class="image-info">
                <strong>{display_name}</strong><br>
                <small>{img}</small>
            </div>
        </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    return html_content

@app.post("/car-crossing")
async def receive_car_image(data: ImageData):
    """Receive and save car crossing image"""
    global image_count
    
    try:
        # Decode base64 image
        img_data = base64.b64decode(data.image)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Generate filename with car ID
        timestamp = data.timestamp
        car_id = data.car_id or "unknown"
        filename = f"{IMAGES_FOLDER}/car{car_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{timestamp:.2f}s.jpg"
        
        # Save image
        cv2.imwrite(filename, frame)
        
        # Update counter
        image_count += 1
        
        lp_status = " (with license plate)" if data.has_license_plate else " (no license plate)"
        print(f" Car {car_id} crossing image saved: {filename}{lp_status}")
        
        return {
            "status": "success",
            "filename": filename,
            "total_received": image_count,
            "timestamp": timestamp,
            "has_license_plate": data.has_license_plate
        }
        
    except Exception as e:
        print(f" Error processing image: {e}")
        return {"status": "error", "message": str(e)}

def main():
    """Run the FastAPI server"""
    print(" Starting Car Detection API Server...")
    print(f" Images folder: {IMAGES_FOLDER}")
    print(f" Server URL: http://{FASTAPI_HOST}:{FASTAPI_PORT}")
    print(f"  Gallery URL: http://{FASTAPI_HOST}:{FASTAPI_PORT}/gallery")
    
    uvicorn.run(
        app,
        host=FASTAPI_HOST,
        port=FASTAPI_PORT,
        reload=DEBUG
    )

if __name__ == "__main__":
    main()