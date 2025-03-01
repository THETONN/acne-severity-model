import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.model import ModelManager
from app.faceparsing import FaceDetector, ImageSegmentation
from app.utils import save_upload_file, convert_heif_to_jpeg, fix_image_rotation
from app.config import Settings
import logging
import numpy as np

app = FastAPI()
settings = Settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    app.state.model_manager = await ModelManager.create()
    app.state.face_detector = FaceDetector()
    app.state.image_segmentation = ImageSegmentation()
    logger.info("Application started, models loaded")

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.post("/check_face")
async def check_face(
    image: UploadFile = File(...),
    face_detector: FaceDetector = Depends(lambda: app.state.face_detector)
):
    try:
        logger.info(f"Received file: {image.filename}, content type: {image.content_type}")
        file_path = await save_upload_file(image)
        logger.info(f"File saved to: {file_path}")
        logger.info(f"File size: {os.path.getsize(file_path)} bytes")

        if file_path.lower().endswith(('.heic', '.heif')):
            file_path = await convert_heif_to_jpeg(file_path)
            logger.info(f"Converted to JPEG: {file_path}")

        fix_image_rotation(file_path)
        logger.info("Image rotation fixed if needed")

        boxes = await face_detector.detect(file_path)
        if boxes is None or not np.any(boxes):  
            logger.warning("No face detected")
            raise HTTPException(status_code=400, detail="No face detected")
        
        logger.info(f"Face detected. Boxes: {boxes.tolist()}")
        return JSONResponse(content={"message": "Face detected", "boxes": boxes.tolist()})
    except Exception as e:
        logger.error(f"Error in check_face: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    model_manager: ModelManager = Depends(lambda: app.state.model_manager),
    face_detector: FaceDetector = Depends(lambda: app.state.face_detector),
    image_segmentation: ImageSegmentation = Depends(lambda: app.state.image_segmentation)
):
    try:
        logger.info(f"Received file for prediction: {image.filename}, content type: {image.content_type}")
        file_path = await save_upload_file(image)
        logger.info(f"File saved to: {file_path}")
        logger.info(f"File size: {os.path.getsize(file_path)} bytes")

        if file_path.lower().endswith(('.heic', '.heif')):
            file_path = await convert_heif_to_jpeg(file_path)
            logger.info(f"Converted to JPEG: {file_path}")

        fix_image_rotation(file_path)
        logger.info("Image rotation fixed if needed")

        boxes = await face_detector.detect(file_path)
        if boxes is None or not np.any(boxes):  
            logger.warning("No face detected")
            raise HTTPException(status_code=400, detail="No face detected")
        
        segmented_image = await image_segmentation.process(file_path)
        prediction = await model_manager.predict(segmented_image)
        
        logger.info(f"Prediction result: {prediction}")
        return JSONResponse(content={"prediction": int(prediction)})
    except Exception as e:
        logger.error(f"Error in predict: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)