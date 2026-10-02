# Multimodal Deepfake Detection System

A multimodal deepfake detection system for analyzing **images, audio, and video** using deep visual features, audio features, and multimodal feature fusion.

The project investigates whether combining evidence from different modalities can improve deepfake detection compared with analyzing a single modality independently.

---

## Overview

Deepfakes can manipulate different aspects of digital media, including facial appearance, video content, and speech or audio.

A system that examines only one modality can miss manipulation that is detectable in another modality. This project therefore implements separate detection pipelines for:

* **Image**
* **Audio**
* **Video**
* **Video + Audio**

The video pipeline combines visual information extracted from sampled video frames with audio information extracted from the same video.

The current system is designed as a research prototype and provides a FastAPI backend for running predictions on uploaded media.

---

## System Architecture

```text
                    MULTIMODAL DEEPFAKE
                       DETECTION SYSTEM
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
          IMAGE             AUDIO            VIDEO
             │                │                │
             ▼                ▼                ▼
         ResNet50        Mel Spectrogram    Sample Frames
         Features          Features             │
             │                │                ▼
             │                │             ResNet50
             │                │             Features
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                       Feature Fusion
                              │
                              ▼
                    REAL / DEEPFAKE
                       + Confidence
```

---

## Supported Modalities

### 1. Image Detection

A single image is processed using a pretrained **ResNet50** network.

The final classification layer is removed and the resulting **2048-dimensional feature representation** is passed to the trained machine-learning classifier.

```text
Image
  ↓
ResNet50
  ↓
2048-D Visual Features
  ↓
Classifier
  ↓
REAL / DEEPFAKE
```

---

### 2. Audio Detection

Audio is converted to:

* Mono
* 16 kHz
* Float32 PCM

A Mel spectrogram is then generated using:

* 64 Mel bands
* FFT size: 1024
* Hop length: 256

Mean and standard deviation are calculated across time, producing a **128-dimensional audio feature vector**.

```text
Audio
  ↓
FFmpeg
  ↓
16 kHz Mono Audio
  ↓
Mel Spectrogram
  ↓
Mean + Standard Deviation
  ↓
128-D Audio Features
  ↓
Classifier
  ↓
REAL / DEEPFAKE
```

---

### 3. Video Detection

The video pipeline uniformly samples **8 frames** from the video.

Each sampled frame is processed using ResNet50 to obtain a 2048-dimensional visual representation.

The frame representations are mean-pooled into a single video-level visual feature vector.

The video's audio is processed separately and combined with the visual representation.

```text
Video
 ├──► 8 Sampled Frames
 │       ↓
 │    ResNet50
 │       ↓
 │   Visual Features
 │
 └──► Audio
         ↓
      Mel Features
         │
         ▼
    Feature Fusion
         ↓
     Classifier
         ↓
   REAL / DEEPFAKE
```

---

## Multimodal Fusion

The current multimodal model uses **feature-level fusion**.

The 2048-dimensional visual feature vector and 128-dimensional audio feature vector are concatenated:

```text
2048-D Visual Features
          +
128-D Audio Features
          ↓
2176-D Combined Representation
          ↓
Fusion Classifier
```

This allows the classifier to make its decision using both visual and audio evidence.

---

## Cross-Modal Consistency Experiment

A consistency signal was also investigated by comparing the visual and audio deepfake probabilities.

The experiment tested whether disagreement between modalities could improve the final prediction.

In the current evaluation, adding the consistency adjustment **did not change the final test metrics** compared with basic visual + audio fusion.

Therefore, the current results do **not** claim an improvement from the consistency component.

---

# Dataset

The experiments use the **FakeAVCeleb** dataset.

Dataset source:

https://www.kaggle.com/datasets/shreyaty08/fakeavceleb

The experimental subset used in this project contains:

* **499 subjects**
* **1,996 videos**
* **4 categories**
* **499 videos per category**

### Categories

| Category                |    Videos |
| ----------------------- | --------: |
| Fake Video + Fake Audio |       499 |
| Fake Video + Real Audio |       499 |
| Real Video + Fake Audio |       499 |
| Real Video + Real Audio |       499 |
| **Total**               | **1,996** |

The dataset was constructed using subjects common across the four categories.

---

## Subject-Level Dataset Split

To reduce identity leakage between training and evaluation, subjects were divided rather than randomly splitting individual videos.

| Split      | Subjects |    Videos |
| ---------- | -------: | --------: |
| Training   |      350 |     1,400 |
| Validation |       74 |       296 |
| Test       |       75 |       300 |
| **Total**  |  **499** | **1,996** |

Each split contains equal representation from the four dataset categories.

---

## Dataset Labels

For the primary video classification task:

```text
FakeVideo → DEEPFAKE
RealVideo → REAL
```

The audio labels are also represented by the FakeAudio / RealAudio categories and are used for the audio and multimodal experiments.

The dataset itself is **not included in this repository**.

---

# Experimental Results

The final held-out test set contains **300 videos** from subjects not present in the training set.

### Main Results

| Model                        |   Accuracy | Precision | Recall |         F1 |    ROC-AUC |
| ---------------------------- | ---------: | --------: | -----: | ---------: | ---------: |
| Image                        |     67.67% |    65.32% | 75.33% |     69.97% |     0.7401 |
| Audio                        |     72.67% |    73.94% | 70.00% |     71.92% |     0.8220 |
| Visual                       |     74.33% |    72.39% | 78.67% |     75.40% |     0.8264 |
| Visual + Audio               | **76.33%** |    74.23% | 80.67% | **77.32%** | **0.8579** |
| Visual + Audio + Consistency |     76.33% |    74.23% | 80.67% |     77.32% |     0.8579 |

The multimodal visual + audio model achieved higher test accuracy, F1, and ROC-AUC than the visual-only baseline in this experimental setup.

The consistency experiment produced the same metrics as basic visual + audio fusion.

> These results represent the current experimental configuration and should not be interpreted as state-of-the-art performance.

---

## Video + Audio Confusion Matrix

The visual + audio fusion model produced the following results on the 300-video test set:

```text
                 Predicted
                 REAL   DEEPFAKE

Actual REAL       108      42
Actual DEEPFAKE    29     121
```

Total correct predictions:

```text
108 + 121 = 229 / 300
```

Test accuracy:

```text
76.33%
```

---

## ROC-AUC Comparison

```text
Visual              0.8264
Audio               0.8220
Visual + Audio      0.8579
```

The ROC-AUC result provides an additional comparison of the models beyond a single classification threshold.

---

# Model Artifacts

The trained v3 models are stored in:

```text
models/v3/
```

Current artifacts include:

```text
models/v3/
├── audio_model.joblib
├── visual_model.joblib
├── image_model.joblib
├── image_audio_model.joblib
├── fusion_model.joblib
├── evaluation_results.json
└── evaluation/
    ├── roc_comparison.png
    └── video_audio_confusion_matrix.png
```

The models were trained using the experimental dataset described above.

---

# Backend API

The project exposes the trained models through a **FastAPI** backend.

## Start the API

```bash
uvicorn src.api:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Endpoints

### Health Check

```http
GET /
```

### Image Prediction

```http
POST /predict/image
```

### Audio Prediction

```http
POST /predict/audio
```

### Video Prediction

```http
POST /predict/video
```

The prediction endpoints accept uploaded media files and return the detected modality, prediction, confidence, and deepfake probability.

Example:

```json
{
  "modality": "video",
  "prediction": "DEEPFAKE",
  "confidence": 0.9999,
  "deepfake_probability": 0.9999
}
```

The confidence value represents the model's confidence in its predicted class. It should not be interpreted as guaranteed real-world accuracy.

---

# Project Structure

```text
Multimodal-Deepfake-Detection-System/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── processed_audios/
│       ├── processed_images/
│       └── processed_videos/
│
├── models/
│   └── v3/
│       ├── audio_model.joblib
│       ├── visual_model.joblib
│       ├── image_model.joblib
│       ├── image_audio_model.joblib
│       ├── fusion_model.joblib
│       ├── evaluation_results.json
│       └── evaluation/
│
├── src/
│   ├── api.py
│   ├── features/
│   │   ├── audio_features.py
│   │   ├── image_features.py
│   │   └── video_features.py
│   │
│   └── inference/
│       └── predict.py
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# Technology Stack

### Machine Learning

* Python
* PyTorch
* Torchvision
* ResNet50
* Scikit-learn
* Joblib

### Audio Processing

* Librosa
* FFmpeg
* NumPy

### Video / Image Processing

* OpenCV
* Pillow

### Backend

* FastAPI
* Uvicorn
* Python Multipart

---

# Environment Versions

The current local development environment uses:

```text
Python        3.12.2
NumPy         2.5.3
PyTorch       2.14.1
Torchvision   0.29.1
Librosa       1.0.0
OpenCV        5.0.0
Scikit-learn  1.6.1
Joblib        1.6.0
Pillow        12.3.0
FastAPI       0.142.2
Uvicorn       0.54.0
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/sandipgorad/Multimodal-Deepfake-Detection-Systems.git
cd Multimodal-Deepfake-Detection-Systems
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

FFmpeg is required because the audio pipeline uses FFmpeg to extract audio from video files.

Verify FFmpeg:

```bash
ffmpeg -version
```

---

# Running the Backend

Start the FastAPI server:

```bash
uvicorn src.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

From Swagger UI, image, audio, and video files can be uploaded directly to their respective prediction endpoints.

---

# Local Testing Data

Small local test samples are maintained under:

```text
data/processed/
```

These files are intended for local inference and API testing.

The complete FakeAVCeleb dataset is not stored in the repository.

---

# Research Experiments

The project currently evaluates multiple configurations:

### Experiment 1 — Image

Single-frame image representation → classifier.

### Experiment 2 — Audio

Audio features → classifier.

### Experiment 3 — Visual

Multiple uniformly sampled video frames → ResNet50 → mean pooled visual representation → classifier.

### Experiment 4 — Visual + Audio

Video visual representation + audio representation → feature fusion → classifier.

### Experiment 5 — Visual + Audio + Consistency

Visual and audio predictions are additionally compared for cross-modal consistency.

This experiment currently produces the same evaluation metrics as the visual + audio fusion model.

---

# Limitations

The current system has several limitations:

* The current models use relatively compact feature representations rather than end-to-end multimodal neural networks.
* Video analysis uses 8 uniformly sampled frames rather than processing every frame.
* Audio representation is based on aggregated Mel-spectrogram statistics.
* The current fusion approach is feature concatenation followed by a classical classifier.
* The current consistency mechanism did not improve the reported test metrics.
* Performance depends on the dataset and experimental setup.
* A model's confidence score does not guarantee that a prediction is correct.
* The current results should not be generalized to all real-world media without further evaluation.
* The current image experiment represents a single-frame image-input baseline.

---

# Future Work

Potential future improvements include:

* Stronger temporal video modeling
* More advanced audio representations
* End-to-end multimodal neural architectures
* Improved cross-modal consistency modeling
* Larger and more diverse evaluation datasets
* External/unseen media evaluation
* More robust threshold calibration
* Explainable prediction outputs
* Frontend interface for practical media analysis

---

# Current Status

### Completed

* Image detection pipeline
* Audio detection pipeline
* Video detection pipeline
* Visual + audio feature fusion
* Model training and evaluation
* Subject-level dataset split
* Local model inference
* FastAPI backend
* Image/audio/video API endpoints
* Swagger API documentation

### Next

**Frontend interface and integration with the FastAPI backend.**

---

# Research Note

This repository contains the current implementation and experimental results of the project.

The reported metrics are specific to the described dataset split, feature extraction process, models, and evaluation procedure. Further experiments are required before making broader claims about real-world deepfake detection performance.
