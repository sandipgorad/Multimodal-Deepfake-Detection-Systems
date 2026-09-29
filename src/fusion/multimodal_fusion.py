def fuse_predictions(
    image_probability=None,
    audio_probability=None,
    video_probability=None
):
    probabilities = []
    weights = []

    if image_probability is not None:
        probabilities.append(image_probability)
        weights.append(1.0)

    if audio_probability is not None:
        probabilities.append(audio_probability)
        weights.append(1.0)

    if video_probability is not None:
        probabilities.append(video_probability)
        weights.append(1.0)

    if not probabilities:
        raise ValueError("No predictions provided")

    final_probability = sum(
        probability * weight
        for probability, weight in zip(probabilities, weights)
    ) / sum(weights)

    label = "DEEPFAKE" if final_probability >= 0.5 else "REAL"

    return label, final_probability