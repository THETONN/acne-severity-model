import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import gdown
from app.config import Settings

class ModelManager:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None

    @classmethod
    async def create(cls):
        self = cls()
        await self.download_model()
        await self.load_model()
        return self

    async def download_model(self):
        output = Settings.MODEL_PATH
        if not os.path.exists(output):
            gdown.download(Settings.MODEL_URL, output, quiet=False)

    async def load_model(self):
        self.model = models.resnet34(weights=None)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 3)
        )
        state_dict = torch.load(Settings.MODEL_PATH, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    async def predict(self, image_array):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(Image.fromarray(image_array)).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
        return output.argmax().item()