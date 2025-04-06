from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
from ultralytics import YOLO
from datetime import datetime
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://backend-54v5.onrender.com",
        "http://localhost:8000",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Response Models
class SuccessResponse(BaseModel):
    success: bool
    message: int
    confidence: float

class ErrorResponse(BaseModel):
    success: bool
    error: str
    details: str

# Path to YOLO model
MODEL_PATH = 'yolo_assets/Models/yolov8n.pt'
model = None  # Global model variable

# Load model on startup
@app.on_event("startup")
def load_model():
    global model
    try:
        model = YOLO(MODEL_PATH)
        logger.info(f"✅ YOLO model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        logger.error(f"❌ Failed to load YOLO model: {e}")
        model = None

# YOLO frame processing
def process_frame(frame):
    try:
        if model is None:
            raise ValueError("YOLO model is not loaded")

        result = model(frame, conf=0.5, verbose=False, classes=[0])[0]
        boxes = result.boxes
        return len(boxes)
    except Exception as e:
        logger.error(f"❌ Error processing frame: {e}")
        return 0

# Image Upload API
@app.post("/process_image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        result = process_frame(image)

        return {
            "success": True,
            "message": result,
            "confidence": 0.95
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

# Lightweight Health Check
@app.get("/health")
async def health_check():
    return {
        "status": "ok",

    }
