from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2
import os
from ultralytics import YOLO

templates = Jinja2Templates(directory="templates")


app = FastAPI()
# Load YOLO model and class names once at startup
MODEL_PATH = 'yolo_assets/Models/yolov8n.pt'
model = YOLO(MODEL_PATH)

def process_frame(frame):
    result = model(frame, conf=0.5, verbose=False, classes=[0])[0]
    boxes = result.boxes
    n_detection = len(boxes)
    return n_detection

@app.get("/", response_class=HTMLResponse)
async def get_camera_app(request: Request):
    """Serve the camera application HTML page from templates directory"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process_image/")
async def upload_image(file: UploadFile = File(...)):
    # Read image bytes and convert to NumPy array
    image_bytes = await file.read()
    image_np = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)  # Decode image

    if image is None:
        return JSONResponse(content={"error": "Invalid image format"}, status_code=400)

    result = process_frame(image)  # Process image

    return JSONResponse(content={"message": result})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
