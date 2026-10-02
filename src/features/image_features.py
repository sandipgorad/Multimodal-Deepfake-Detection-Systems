import cv2
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.fc = torch.nn.Identity()
model = model.to(device)
model.eval()

transform = weights.transforms()


def load_image(path):
    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Could not load image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return Image.fromarray(image)


def extract_image_features(image):
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(tensor)

    return features.squeeze(0).cpu().numpy()