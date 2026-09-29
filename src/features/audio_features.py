import librosa
import numpy as np


def load_audio(path):
    audio, sample_rate = librosa.load(path, sr=16000)

    return audio, sample_rate


def extract_audio_features(audio, sample_rate):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    features = np.mean(mfcc, axis=1)

    return features