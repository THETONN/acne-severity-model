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
@app.post("/check_face")
async def check_face(image: UploadFile = File(...)):
    try:
        file_location = f"uploads/{image.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        logger.info(f"File saved to {file_location}")

        img = Image.open(file_location)
        boxes, _ = mtcnn.detect(img)

        if boxes is None or len(boxes) == 0:
            logger.error("No face detected in the uploaded image")
            raise HTTPException(status_code=400, detail="No face detected in the uploaded image")

        return JSONResponse(content={"message": "Face detected"})
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        logger.info("Received a request for prediction")
        file_location = f"uploads/{image.filename}"
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        logger.info(f"File saved to {file_location}")

        img = Image.open(file_location)
        boxes, _ = mtcnn.detect(img)

        if boxes is None or len(boxes) == 0:
            logger.error("No face detected in the uploaded image")
            raise HTTPException(status_code=400, detail="No face detected in the uploaded image")

        segmented_image = process_image(file_location)
        logger.info("Image segmented successfully")

        if not os.path.exists('processed'):
            os.makedirs('processed')

        processed_image_path = f"processed/{image.filename}"
        Image.fromarray(segmented_image).save(processed_image_path)
        logger.info(f"Segmented image saved to {processed_image_path}")

        prediction = predict_image(np.array(segmented_image), deep_learning_model)
        logger.info(f"Prediction completed with result: {prediction}")

        # ลบไฟล์จากโฟลเดอร์ uploads และ processed หลังการประมวลผลเสร็จสิ้น
        os.remove(file_location)
        os.remove(processed_image_path)

        result = {"prediction": int(prediction)}
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})



import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)



