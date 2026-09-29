import cv2
import torch
import numpy as np
from torchvision import models


def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Could not load image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def preprocess_image(image, size=(224, 224)):
    image = cv2.resize(image, size)
    image = image.astype("float32") / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype="float32")
    std = np.array([0.229, 0.224, 0.225], dtype="float32")

    image = (image - mean) / std

    return image

def prepare_image(path, size=(224, 224)):
    image = load_image(path)
    image = preprocess_image(image, size)

    return image

def create_image_model():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()

    return model

def extract_image_features(model, image):
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        features = model(image)

    return features.squeeze(0).numpy()