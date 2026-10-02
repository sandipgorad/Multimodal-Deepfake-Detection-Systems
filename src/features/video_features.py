import cv2
import numpy as np
from PIL import Image

from src.features.image_features import extract_image_features


def extract_video_frames(path, num_frames=8):
    video = cv2.VideoCapture(path)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        video.release()
        raise ValueError(f"Could not read video: {path}")

    indices = np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int,
    )

    frames = []

    for index in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, index)

        success, frame = video.read()

        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    video.release()

    if not frames:
        raise ValueError(f"Could not extract frames from: {path}")

    return frames


def extract_video_features(path):
    frames = extract_video_frames(path)

    features = []

    for frame in frames:
        image = Image.fromarray(frame)
        feature = extract_image_features(image)
        features.append(feature)

    return np.mean(np.array(features), axis=0)