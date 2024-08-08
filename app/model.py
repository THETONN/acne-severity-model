import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import gdown

# URL ของโมเดลบน Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1qTzRho4zqzXcEdZIX_7bkLs4s6yiu0DN"

# ดาวน์โหลดโมเดลถ้าไฟล์ไม่อยู่
def download_model():
    output = './app/model/resnet34_model.pth'
    if not os.path.exists(output):
        gdown.download(MODEL_URL, output, quiet=False)

download_model()

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# โหลดโมเดลแบบ asynchronous
async def load_model():
    deep_learning_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    num_features = deep_learning_model.fc.in_features
    deep_learning_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 3)
    )

    # โหลด state dictionary ของโมเดล
    state_dict_path = "./app/model/resnet34_model.pth"
    if os.path.exists(state_dict_path):
        deep_learning_model.load_state_dict(torch.load(state_dict_path, map_location=device))
    
    deep_learning_model.to(device)
    deep_learning_model.eval()
    return deep_learning_model

# เรียกใช้งานโมเดล
deep_learning_model = None

async def init_model():
    global deep_learning_model
    deep_learning_model = await load_model()

# ฟังก์ชันการทำนายภาพ
def predict_image(image_array):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_tensor = preprocess(Image.fromarray(image_array)).unsqueeze(0).to(device)
    
    # รันการทำนายแบบ asynchronous
    with torch.no_grad():
        outputs = deep_learning_model(image_tensor)
    
    # ได้ผลลัพธ์การทำนาย
    _, prediction = torch.max(outputs, 1)
    
    return prediction.item()

# เรียก init_model เมื่อเริ่มต้นเซิร์ฟเวอร์
import asyncio
asyncio.run(init_model())
