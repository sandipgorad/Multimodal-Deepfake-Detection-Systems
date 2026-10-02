from fastapi import FastAPI, UploadFile, File
import tempfile
import os

from src.inference.predict import predict_image, predict_audio, predict_video

ALLOWED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".wav", ".mp3", ".m4a",
    ".mp4", ".avi", ".mov"
}

def is_allowed_file(filename):
    extension = os.path.splitext(filename)[1].lower()
    return extension in ALLOWED_EXTENSIONS

app = FastAPI(title="Multimodal Deepfake Detection API")

@app.get("/")
def root():
    return {"message": "Deepfake Detection API is running"}


@app.post("/predict/image")
async def predict_image_api(file: UploadFile = File(...)):
    if not is_allowed_file(file.filename):
        return {"error": "Unsupported file type"}
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    try:
        label, confidence, probability = predict_image(temp_path)

        return {
            "modality": "image",
            "prediction": label,
            "confidence": confidence,
            "deepfake_probability": probability
        }

    finally:
        os.remove(temp_path)

@app.post("/predict/audio")
async def predict_audio_api(file: UploadFile = File(...)):
    if not is_allowed_file(file.filename):
        return {"error": "Unsupported file type"}
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    try:
        label, confidence, probability = predict_audio(temp_path)

        return {
            "modality": "audio",
            "prediction": label,
            "confidence": confidence,
            "deepfake_probability": probability
        }

    finally:
        os.remove(temp_path)

@app.post("/predict/video")
async def predict_video_api(file: UploadFile = File(...)):
    if not is_allowed_file(file.filename):
        return {"error": "Unsupported file type"}
    suffix = os.path.splitext(file.filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
        temp.write(await file.read())
        temp_path = temp.name

    try:
        label, confidence, probability = predict_video(temp_path)

        return {
            "modality": "video",
            "prediction": label,
            "confidence": confidence,
            "deepfake_probability": probability
        }

    finally:
        os.remove(temp_path)