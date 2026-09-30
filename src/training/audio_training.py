import os
import numpy as np
import torch
import joblib
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import train_test_split

from src.features.audio_features import (
    load_audio,
    extract_audio_features
)

from src.models.audio_classifier import AudioClassifier

def load_audio_dataset():
    audios = []
    labels = []

    audio_dir = "data/raw/audios"

    for filename in os.listdir(audio_dir):
        path = os.path.join(audio_dir, filename)

        if filename.startswith("real_"):
            label = 0
        elif filename.startswith("fake_"):
            label = 1
        else:
            continue

        audio, sample_rate = load_audio(path)
        features = extract_audio_features(audio, sample_rate)

        audios.append(features)
        labels.append(label)

    return np.array(audios), np.array(labels)

def train_audio_classifier():
    X, y = load_audio_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = CalibratedClassifierCV(
        SVC(kernel="linear", random_state=42),
        ensemble=False
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test

def evaluate_audio_classifier(model, X_test, y_test):
    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, zero_division=0))
    print("Recall:", recall_score(y_test, predictions, zero_division=0))
    print("F1:", f1_score(y_test, predictions, zero_division=0))

def save_audio_model(model, path="models/audio_classifier.joblib"):
    joblib.dump(model, path)
    print("Audio model saved:", path)

