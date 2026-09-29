import cv2
import numpy as np

from src.features.image_features import preprocess_image

from src.features.image_features import create_image_model, extract_image_features



def extract_video_frames(path, num_frames=10):
    video = cv2.VideoCapture(path)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frames = []

    if total_frames == 0:
        video.release()
        raise ValueError("Could not read video")

    step = max(total_frames // num_frames, 1)

    for i in range(0, total_frames, step):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)

        success, frame = video.read()

        if success:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = preprocess_image(frame)
            frames.append(frame)

        if len(frames) == num_frames:
            break

    video.release()

    return frames

def extract_video_features(frames):
    model = create_image_model()

    features = []

    for frame in frames:
        feature = extract_image_features(model, frame)
        features.append(feature)

    return np.mean(np.array(features), axis=0)