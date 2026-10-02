import joblib
import numpy as np

from src.features.audio_features import load_audio, extract_audio_features
from src.features.image_features import load_image, extract_image_features
from src.features.video_features import extract_video_features


MODEL_DIR = "models/v3"

image_model = joblib.load(f"{MODEL_DIR}/image_model.joblib")
audio_model = joblib.load(f"{MODEL_DIR}/audio_model.joblib")
fusion_model = joblib.load(f"{MODEL_DIR}/fusion_model.joblib")


def predict_image(path):
    image = load_image(path)
    features = extract_image_features(image)

    probability = image_model.predict_proba([features])[0]

    label = "DEEPFAKE" if probability[1] >= 0.5 else "REAL"
    confidence = float(max(probability))

    return label, confidence, float(probability[1])


def predict_audio(path):
    audio, sample_rate = load_audio(path)
    features = extract_audio_features(audio, sample_rate)

    probability = audio_model.predict_proba([features])[0]

    label = "DEEPFAKE" if probability[1] >= 0.5 else "REAL"
    confidence = float(max(probability))

    return label, confidence, float(probability[1])


def predict_video(path):
    visual_features = extract_video_features(path)

    audio, sample_rate = load_audio(path)
    audio_features = extract_audio_features(audio, sample_rate)

    combined_features = np.hstack(
        [visual_features, audio_features]
    )

    probability = fusion_model.predict_proba([combined_features])[0]

    label = "DEEPFAKE" if probability[1] >= 0.5 else "REAL"
    confidence = float(max(probability))

    return label, confidence, float(probability[1])