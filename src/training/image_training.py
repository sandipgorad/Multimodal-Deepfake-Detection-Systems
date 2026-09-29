import os
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from src.features.image_features import (
    prepare_image,
    create_image_model,
    extract_image_features
)

from src.models.image_classifier import ImageClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def load_image_dataset():
    images = []
    labels = []

    image_dir = "data/raw/images"

    for filename in os.listdir(image_dir):
        path = os.path.join(image_dir, filename)

        if filename.startswith("real_"):
            label = 0
        elif filename.startswith("fake_"):
            label = 1
        else:
            continue

        image = prepare_image(path)

        images.append(image)
        labels.append(label)

    return np.array(images), np.array(labels)

def train_image_classifier():
    X, y = load_image_dataset()

    model = create_image_model()

    features = []

    for image in X:
        feature = extract_image_features(model, image)
        features.append(feature)

    X = np.array(features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classifier = ImageClassifier()

    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    for epoch in range(30):
        optimizer.zero_grad()

        output = classifier(X_train)
        loss = loss_fn(output, y_train)

        loss.backward()
        optimizer.step()

    return classifier, X_test, y_test

def evaluate_image_classifier(model, X_test, y_test):
    X_test = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        output = model(X_test)
        predictions = torch.argmax(output, dim=1).numpy()

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, zero_division=0))
    print("Recall:", recall_score(y_test, predictions, zero_division=0))
    print("F1:", f1_score(y_test, predictions, zero_division=0))

def save_image_model(model, path="models/image_classifier.pth"):
    torch.save(model.state_dict(), path)
    print("Image model saved:", path)