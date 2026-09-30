import torch

from src.features.image_features import (
    prepare_image,
    create_image_model,
    extract_image_features
)

from src.models.image_classifier import ImageClassifier
from src.features.audio_features import load_audio, extract_audio_features
from src.models.audio_classifier import AudioClassifier

from src.features.video_features import (
    extract_video_frames,
    extract_video_features
)

from src.models.video_classifier import VideoClassifier
from src.fusion.multimodal_fusion import fuse_predictions

def predict_image(path):
    feature_model = create_image_model()

    classifier = ImageClassifier()
    classifier.load_state_dict(
        torch.load("models/image_classifier.pth", map_location="cpu")
    )
    classifier.eval()

    image = prepare_image(path)
    features = extract_image_features(feature_model, image)

    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = classifier(features)
        probability = torch.softmax(output, dim=1)[0]

    label = "DEEPFAKE" if probability[1] > probability[0] else "REAL"
    confidence = float(torch.max(probability))
    deepfake_probability = float(probability[1])

    return label, confidence, deepfake_probability

def predict_audio(path):
    import joblib

    classifier = joblib.load("models/audio_classifier.joblib")

    audio, sample_rate = load_audio(path)
    features = extract_audio_features(audio, sample_rate)

    probability = classifier.predict_proba([features])[0]

    label = "DEEPFAKE" if probability[1] >= 0.5 else "REAL"
    confidence = float(max(probability))

    return label, confidence, float(probability[1])

def predict_video(path):
    classifier = VideoClassifier()
    classifier.load_state_dict(
        torch.load("models/video_classifier.pth", map_location="cpu")
    )
    classifier.eval()

    frames = extract_video_frames(path)
    features = extract_video_features(frames)

    features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        output = classifier(features)
        probability = torch.softmax(output, dim=1)[0]

    label = "DEEPFAKE" if probability[1] > probability[0] else "REAL"
    confidence = float(torch.max(probability))

    return label, confidence, float(probability[1])

def predict_multimodal(image_path=None, audio_path=None, video_path=None):

    image_probability = None
    audio_probability = None
    video_probability = None

    if image_path:
        _, _, image_probability = predict_image(image_path)

    if audio_path:
        _, _, audio_probability = predict_audio(audio_path)

    if video_path:
        _, _, video_probability = predict_video(video_path)

    return fuse_predictions(
        image_probability,
        audio_probability,
        video_probability
    )