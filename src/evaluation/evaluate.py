from src.training.image_training import (
    train_image_classifier,
    evaluate_image_classifier
)

from src.training.audio_training import (
    train_audio_classifier,
    evaluate_audio_classifier
)

from src.training.video_training import (
    train_video_classifier,
    evaluate_video_classifier
)

def evaluate_all():

    print("\n===== IMAGE =====")
    image_model, X_test, y_test = train_image_classifier()
    evaluate_image_classifier(image_model, X_test, y_test)

    print("\n===== AUDIO =====")
    audio_model, X_test, y_test = train_audio_classifier()
    evaluate_audio_classifier(audio_model, X_test, y_test)

    print("\n===== VIDEO =====")
    video_model, X_test, y_test = train_video_classifier()
    evaluate_video_classifier(video_model, X_test, y_test)

if __name__ == "__main__":
    evaluate_all()