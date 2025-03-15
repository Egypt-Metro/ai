from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
from ultralytics import YOLO
from datetime import datetime
import logging
import os

# Configure logging for debugging and error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI()

# Configure CORS middleware to allow frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://backend-54v5.onrender.com",  # Production server on Render
        "http://localhost:8000",              # Local backend
        "http://localhost:3000",              # Local frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define response models for successful and error responses
class SuccessResponse(BaseModel):
    success: bool
    message: int  # Number of passengers detected
    confidence: float  # Confidence score

class ErrorResponse(BaseModel):
    success: bool
    error: str
    details: str

# Load YOLO model
MODEL_PATH = 'yolo_assets/Models/yolov8n.pt'
try:
    model = YOLO(MODEL_PATH)
    logger.info(f"✅ YOLO model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to load YOLO model: {e}")
    model = None  # Model is set to None in case of failure

# Function to process an image frame using YOLO
def process_frame(frame):
    try:
        if model is None:
            raise ValueError("YOLO model is not loaded")

        result = model(frame, conf=0.5, verbose=False, classes=[0])[0]
        boxes = result.boxes
        n_detection = len(boxes)
        return n_detection
    except Exception as e:
        logger.error(f"❌ Error processing frame: {e}")
        return 0

# API endpoint for image processing
@app.post("/process_image/", response_model=SuccessResponse)
async def upload_image(file: UploadFile = File(...)):
    try:
        # Ensure the uploaded file is an image
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image type")

        # Read and decode image
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Process the image and return the result
        result = process_frame(image)

        return {
            "success": True,
            "message": result,
            "confidence": 0.95  # Example confidence score
        }

    except HTTPException as he:
        logger.error(f"🚨 HTTP Error: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error processing image: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Image processing failed",
                "details": str(e)
            }
        )

# Health Check Endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "operational",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }
