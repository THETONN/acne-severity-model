import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import gdown

# URL ของโมเดลบน Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1qTzRho4zqzXcEdZIX_7bkLs4s6yiu0DN"

async def download_model():
    output = './app/model/resnet34_model.pth'
    if not os.path.exists(output):
        gdown.download(MODEL_URL, output, quiet=False)

async def load_model():
    await download_model()
    state_dict_path = "./app/model/resnet34_model.pth"
    deep_learning_model = models.resnet34(weights=None)
    
    # Adjust the fully connected layer to match the model's state_dict
    num_features = deep_learning_model.fc.in_features
    deep_learning_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 3)
    )
    
    # Load state_dict with weights_only=True for security
    state_dict = torch.load(state_dict_path, map_location=device, weights_only=True)
    
    deep_learning_model.load_state_dict(state_dict)
    deep_learning_model.to(device)
    deep_learning_model.eval()
    return deep_learning_model

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def predict_image(image_array):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(Image.fromarray(image_array)).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = deep_learning_model(input_tensor)
    return output.item()
