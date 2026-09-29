import os
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.features.video_features import (
    extract_video_frames,
    extract_video_features
)

from src.models.video_classifier import VideoClassifier

def load_video_dataset():
    features = []
    labels = []

    video_dir = "data/raw/videos"

    for filename in os.listdir(video_dir):
        path = os.path.join(video_dir, filename)

        if filename.startswith("real_"):
            label = 0
        elif filename.startswith("fake_"):
            label = 1
        else:
            continue

        frames = extract_video_frames(path)
        video_feature = extract_video_features(frames)

        features.append(video_feature)
        labels.append(label)

    return np.array(features), np.array(labels)

def train_video_classifier():
    X, y = load_video_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = VideoClassifier()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(30):
        optimizer.zero_grad()

        output = model(X_train)
        loss = loss_fn(output, y_train)

        loss.backward()
        optimizer.step()

    return model, X_test, y_test

def evaluate_video_classifier(model, X_test, y_test):
    X_test = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        output = model(X_test)
        predictions = torch.argmax(output, dim=1).numpy()

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, zero_division=0))
    print("Recall:", recall_score(y_test, predictions, zero_division=0))
    print("F1:", f1_score(y_test, predictions, zero_division=0))

def save_video_model(model, path="models/video_classifier.pth"):
    torch.save(model.state_dict(), path)
    print("Video model saved:", path)