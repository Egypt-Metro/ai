from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from ultralytics import YOLO
from datetime import datetime
import logging
import os

# إعداد الـ Logging لتسجيل الأخطاء والعمليات
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# إنشاء تطبيق FastAPI
app = FastAPI()

# **تهيئة CORS لربط الـ Backend مع السيرفر**
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://backend-54v5.onrender.com",  # السيرفر الأساسي على Render
        "http://localhost:8000",              # تشغيل محلي
        "http://localhost:3000",              # تشغيل الـ Frontend محليًا
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# **تحميل موديل YOLO**
MODEL_PATH = 'yolo_assets/Models/yolov8n.pt'
try:
    model = YOLO(MODEL_PATH)
    logger.info(f"✅ YOLO model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"❌ Failed to load YOLO model: {e}")
    model = None  # في حالة الفشل، يتم التعامل مع الخطأ لاحقًا

# **دالة لمعالجة الصور باستخدام YOLO**
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

# **مسار API لاستقبال ومعالجة الصور**
@app.post("/process_image/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # التأكد من أن الملف المرسل صورة
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid image type")

        # قراءة وتحليل الصورة
        image_bytes = await file.read()
        image_np = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # معالجة الصورة وإرجاع النتيجة
        result = process_frame(image)

        return {
            "success": True,
            "message": result,
            "metadata": {
                "model": "YOLOv8n",
                "confidence": 0.5,
                "timestamp": datetime.now().isoformat()
            }
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

# **Health Check Endpoint**
@app.get("/health")
async def health_check():
    return {
        "status": "operational",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }
