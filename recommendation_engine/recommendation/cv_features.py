def get_default_cv_features():
    """
    Default values when the user does not upload an image.
    """

    return {
        "face_shape": None,
        "skin_tone": None,
        "skin_undertone": None,
        "hair_color": None,
        "confidence": {
            "face_shape": 0.0,
            "skin_tone": 0.0,
            "skin_undertone": 0.0,
            "hair_color": 0.0
        }
    }


def get_cv_features():
    """
    Temporary test CV output.

    Later, replace this function with the actual
    OpenCV/MediaPipe output from the teammate's module.
    """

    return {
        "face_shape": "round",
        "skin_tone": "medium-light",
        "skin_undertone": "warm",
        "hair_color": "brown",

        "confidence": {
            "face_shape": 0.87,
            "skin_tone": 0.91,
            "skin_undertone": 0.76,
            "hair_color": 0.84
        }
    }


def validate_cv_features(features):

    valid_face_shapes = {
        "oval",
        "round",
        "square",
        "oblong",
        "heart",
        "diamond"
    }

    valid_skin_tones = {
        "deep",
        "medium-deep",
        "medium",
        "medium-light",
        "light"
    }

    valid_undertones = {
        "warm",
        "cool",
        "neutral"
    }

    valid_hair_colors = {
        "black",
        "dark_brown",
        "brown",
        "light_brown",
        "blonde",
        "red_or_auburn",
        "gray_or_white"
    }

    if features["face_shape"] not in valid_face_shapes:
        return False

    if features["skin_tone"] not in valid_skin_tones:
        return False

    if features["skin_undertone"] not in valid_undertones:
        return False

    if features["hair_color"] not in valid_hair_colors:
        return False

    for value in features["confidence"].values():

        if not 0.0 <= value <= 1.0:
            return False

    return True


if __name__ == "__main__":

    features = get_cv_features()

    print("CV FEATURES")
    print("=" * 40)

    print("Face shape:", features["face_shape"])
    print("Skin tone:", features["skin_tone"])
    print("Skin undertone:", features["skin_undertone"])
    print("Hair color:", features["hair_color"])

    print("\nConfidence")
    print("-" * 40)

    for key, value in features["confidence"].items():
        print(f"{key}: {value}")

    print("\nValidation:", validate_cv_features(features))