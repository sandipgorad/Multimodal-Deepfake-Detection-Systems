import subprocess

import librosa
import numpy as np


def load_audio(path):
    command = [
        "ffmpeg",
        "-i",
        path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "f32le",
        "-",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    audio = np.frombuffer(result.stdout, dtype=np.float32)

    return audio, 16000


def extract_audio_features(audio, sample_rate):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=64,
        n_fft=1024,
        hop_length=256,
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    return np.concatenate(
        [
            np.mean(mel_db, axis=1),
            np.std(mel_db, axis=1),
        ]
    )