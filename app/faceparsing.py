import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import os

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
segmentation_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
segmentation_model.to(device)

def process_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unable to open or convert image: {image_path}, Error: {e}")

    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = segmentation_model(**inputs)
    logits = outputs.logits

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode='bilinear',
        align_corners=False
    )

    labels = upsampled_logits.argmax(dim=1)[0]

    face_parts_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    face_mask = np.isin(labels.cpu().numpy(), face_parts_labels)

    face_image = np.array(image) * face_mask[:, :, np.newaxis]
    face_image[~face_mask] = [0, 0, 0]

    return face_image
