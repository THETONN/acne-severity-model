import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
from app.config import Settings
import logging

logger = logging.getLogger(__name__)

class FaceDetector:
    def __init__(self):
        self.device = Settings.DEVICE
        self.mtcnn = None

    async def detect(self, image_path):
        if self.mtcnn is None:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(keep_all=True, device=self.device, thresholds=[0.6, 0.7, 0.7])
        
        img = Image.open(image_path)
        logger.info(f"Image for face detection - Size: {img.size}, Mode: {img.mode}")
        boxes, _ = self.mtcnn.detect(img)
        logger.info(f"Detected boxes: {boxes}")
        return boxes

class ImageSegmentation:
    def __init__(self):
        self.device = Settings.DEVICE
        self.image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        self.segmentation_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        self.segmentation_model.to(self.device)

    async def process(self, image_path):
        image = Image.open(image_path).convert("RGB")
        logger.info(f"Image for segmentation - Size: {image.size}, Mode: {image.mode}")
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.segmentation_model(**inputs)
        logits = outputs.logits

        upsampled_logits = nn.functional.interpolate(
            logits, size=image.size[::-1], mode='bilinear', align_corners=False
        )

        labels = upsampled_logits.argmax(dim=1)[0]
        face_parts_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        face_mask = np.isin(labels.cpu().numpy(), face_parts_labels)

        face_image = np.array(image) * face_mask[:, :, np.newaxis]
        face_image[~face_mask] = [0, 0, 0]

        logger.info(f"Segmentation complete. Face image shape: {face_image.shape}")
        return face_image