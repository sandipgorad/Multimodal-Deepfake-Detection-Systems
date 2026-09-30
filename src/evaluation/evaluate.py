from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from src.training.image_training import train_image_classifier
from src.training.audio_training import train_audio_classifier
from src.training.video_training import train_video_classifier


def print_results(name, y_test, predictions):

    print(f"\n===== {name} =====")

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, zero_division=0))
    print("Recall:", recall_score(y_test, predictions, zero_division=0))
    print("F1:", f1_score(y_test, predictions, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


def evaluate_all():

    # IMAGE
    image_model, X_test, y_test = train_image_classifier()

    import torch

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        predictions = torch.argmax(
            image_model(X_test_tensor), dim=1
        ).numpy()

    print_results("IMAGE", y_test, predictions)

    # AUDIO
    audio_model, X_test, y_test = train_audio_classifier()

    predictions = audio_model.predict(X_test)

    print_results("AUDIO", y_test, predictions)

    # VIDEO
    video_model, X_test, y_test = train_video_classifier()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        predictions = torch.argmax(
            video_model(X_test_tensor), dim=1
        ).numpy()

    print_results("VIDEO", y_test, predictions)


if __name__ == "__main__":
    evaluate_all()