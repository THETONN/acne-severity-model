from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from PIL import Image
import os
import sys
import logging
import numpy as np
import torch
from facenet_pytorch import MTCNN
from app.faceparsing import process_image
from app.model import predict_image, load_model
import asyncio
import pillow_heif
import tempfile
import aiofiles

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MTCNN
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)

global deep_learning_model
deep_learning_model = None

@app.on_event("startup")
async def startup_event():
    global deep_learning_model
    deep_learning_model = await load_model()
    logger.info("Deep learning model loaded successfully")

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

def convert_heif_to_jpeg(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # ตรวจสอบว่าไฟล์เป็น HEIF จริงหรือไม่
        if not pillow_heif.is_supported(file_path):
            raise ValueError(f"File is not a supported HEIF image: {file_path}")

        # อ่านไฟล์ HEIF
        heif_file = pillow_heif.read_heif(file_path)
        
        # แปลงเป็น PIL Image
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        
        # สร้างชื่อไฟล์ใหม่สำหรับ JPEG
        jpeg_path = file_path.rsplit('.', 1)[0] + '.jpg'
        
        # บันทึกเป็นไฟล์ JPEG
        image.save(jpeg_path, format="JPEG")
        
        logger.info(f"Successfully converted {file_path} to {jpeg_path}")
        return jpeg_path
    except Exception as e:
        logger.error(f"Error converting HEIF to JPEG: {e}")
        raise

@app.post("/check_face")
async def check_face(image: UploadFile = File(...)):
    temp_file = None
    try:
        logger.info(f"Received file: {image.filename}, content type: {image.content_type}")
        
        # ตรวจสอบ MIME type
        if image.content_type not in ['image/heic', 'image/heif', 'image/jpeg', 'image/png']:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as temp_file:
            file_location = temp_file.name
            async with aiofiles.open(file_location, 'wb') as out_file:
                content = await image.read()
                await out_file.write(content)
            logger.info(f"File saved to {file_location}")

        logger.info(f"File size: {os.path.getsize(file_location)} bytes")

        if file_location.lower().endswith(('.heic', '.heif')):
            file_location = convert_heif_to_jpeg(file_location)
            logger.info(f"HEIF file converted to JPEG: {file_location}")

        img = Image.open(file_location)
        logger.info(f"Image opened successfully. Size: {img.size}, Mode: {img.mode}")
        
        boxes, _ = mtcnn.detect(img)

        if boxes is None or len(boxes) == 0:
            logger.error("No face detected in the uploaded image")
            raise HTTPException(status_code=400, detail="No face detected in the uploaded image")

        logger.info(f"Face detected. Boxes: {boxes}")
        return JSONResponse(content={"message": "Face detected"})
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if temp_file:
            os.unlink(temp_file.name)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    temp_file = None
    processed_file = None
    try:
        logger.info("Received a request for prediction")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file_location = temp_file.name
            async with aiofiles.open(file_location, 'wb') as out_file:
                content = await image.read()
                await out_file.write(content)
            logger.info(f"File saved to {file_location}")

        if file_location.lower().endswith(('.heic', '.heif')):
            file_location = convert_heif_to_jpeg(file_location)
            logger.info(f"HEIF file converted to JPEG: {file_location}")

        img = Image.open(file_location)
        boxes, _ = mtcnn.detect(img)

        if boxes is None or len(boxes) == 0:
            logger.error("No face detected in the uploaded image")
            raise HTTPException(status_code=400, detail="No face detected in the uploaded image")

        segmented_image = process_image(file_location)
        logger.info("Image segmented successfully")

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as processed_file:
            processed_image_path = processed_file.name
            Image.fromarray(segmented_image).save(processed_image_path)
            logger.info(f"Segmented image saved to {processed_image_path}")

        prediction = predict_image(np.array(segmented_image), deep_learning_model)
        logger.info(f"Prediction completed with result: {prediction}")

        result = {"prediction": int(prediction)}
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if temp_file:
            os.unlink(temp_file.name)
        if processed_file:
            os.unlink(processed_file.name)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)