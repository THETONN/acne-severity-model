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
from faceparsing import process_image
from model import predict_image
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

async def load_model():
    state_dict_path = "./app/model/resnet34_model.pth"
    deep_learning_model = models.resnet34(pretrained=False)
    deep_learning_model.fc = nn.Linear(deep_learning_model.fc.in_features, 1)
    deep_learning_model.load_state_dict(torch.load(state_dict_path, map_location=device))
    deep_learning_model.to(device)
    deep_learning_model.eval()
    return deep_learning_model

deep_learning_model = asyncio.get_event_loop().run_until_complete(load_model())

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

        prediction = predict_image(np.array(segmented_image))
        logger.info(f"Prediction completed with result: {prediction}") 

        result = {"prediction": int(prediction)}
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)