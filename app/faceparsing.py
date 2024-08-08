import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import numpy as np
import os

# Convenience expression for automatically determining device
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Load models
image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
segmentation_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
segmentation_model.to(device)

# Function to process and segment a single image
def process_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")  # แปลงรูปภาพเป็น RGB
    except Exception as e:
        raise ValueError(f"Unable to open or convert image: {image_path}, Error: {e}")
    
    # Run inference on image
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = segmentation_model(**inputs)
    logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

    # Resize output to match input image dimensions
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # H x W
        mode='bilinear',
        align_corners=False
    )

    # Get label masks
    labels = upsampled_logits.argmax(dim=1)[0]

    # Define the specific labels for skin and face parts
    face_parts_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # รวมส่วนต่างๆ ของใบหน้า
    face_mask = np.isin(labels.cpu().numpy(), face_parts_labels)

    # Apply the mask to the original image
    face_image = np.array(image) * face_mask[:, :, np.newaxis]
    face_image[~face_mask] = [0, 0, 0]  # Set background to black

    return face_image
