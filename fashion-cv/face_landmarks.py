from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
FaceLandmarksConnections = mp.tasks.vision.FaceLandmarksConnections


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values -= values.max()
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def _interpolate_boundary(points, target_y):
    if len(points) < 2:
        return None
    for (y1, x1), (y2, x2) in zip(points, points[1:]):
        if y1 <= target_y <= y2:
            if y1 == y2:
                return (x1 + x2) / 2.0
            t = (target_y - y1) / (y2 - y1)
            return x1 + t * (x2 - x1)
    return None


def analyze_face_shape(image_path, output_path=None):
    image_path = Path(image_path)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Face landmark model not found: {MODEL_PATH}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
    )

    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise ValueError("No face detected.")

    face = result.face_landmarks[0]
    oval_indices = {
        index
        for connection in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
        for index in (connection.start, connection.end)
    }
    oval_points = [(face[i].x, face[i].y, i) for i in oval_indices]

    face_left = min(p[0] for p in oval_points)
    face_right = max(p[0] for p in oval_points)
    face_top = min(p[1] for p in oval_points)
    face_bottom = max(p[1] for p in oval_points)
    face_height = face_bottom - face_top

    if face_right <= face_left or face_height <= 0:
        raise ValueError("Invalid face geometry.")

    center_x = (face_left + face_right) / 2.0
    left_boundary = sorted((y, x) for x, y, _ in oval_points if x < center_x)
    right_boundary = sorted((y, x) for x, y, _ in oval_points if x >= center_x)

    levels = {
        "upper": 0.25,
        "upper_middle": 0.35,
        "middle": 0.50,
        "lower_middle": 0.65,
        "lower": 0.75,
    }

    widths = {}
    for name, ratio in levels.items():
        target_y = face_top + ratio * face_height
        left_x = _interpolate_boundary(left_boundary, target_y)
        right_x = _interpolate_boundary(right_boundary, target_y)
        widths[name] = None if left_x is None or right_x is None else right_x - left_x

    valid_widths = [v for v in widths.values() if v is not None and v > 0]
    if not valid_widths:
        raise ValueError("Could not calculate face width profile.")

    max_width = max(valid_widths)
    normalized = {k: (v / max_width if v is not None else None) for k, v in widths.items()}
    length_width_ratio = face_height / max_width

    upper = normalized["upper"]
    upper_middle = normalized["upper_middle"]
    middle = normalized["middle"]
    lower_middle = normalized["lower_middle"]
    lower = normalized["lower"]
    jaw_taper = (middle - lower) if middle is not None and lower is not None else 0.0

    scores = {name: 0.0 for name in ("round", "oval", "square", "oblong", "heart", "diamond")}

    scores["round"] += max(0.0, 1.0 - abs(length_width_ratio - 1.0) / 0.35)
    scores["oval"] += max(0.0, 1.0 - abs(length_width_ratio - 1.25) / 0.45)
    scores["oblong"] += max(0.0, (length_width_ratio - 1.20) / 0.50)

    if lower is not None:
        scores["square"] += 1.0 if lower >= 0.90 else 0.0
        scores["round"] += 0.45 if lower < 0.90 else 0.0
        scores["heart"] += 0.65 if lower < 0.82 else 0.0

    if upper is not None and lower is not None:
        scores["heart"] += 1.0 if upper >= 0.92 and lower < 0.82 else 0.0
        scores["square"] += 1.0 if upper >= 0.90 and lower >= 0.88 and jaw_taper < 0.08 else 0.0

    if middle is not None and upper is not None and lower is not None:
        scores["diamond"] += 1.5 if middle > upper + 0.04 and middle > lower + 0.04 else 0.0

    scores["oblong"] += 1.0 if length_width_ratio >= 1.35 else 0.0
    scores["oval"] += 0.7 if 1.05 <= length_width_ratio < 1.35 and lower is not None and lower < 0.88 else 0.0
    scores["round"] += 0.8 if length_width_ratio <= 1.08 and lower is not None and lower < 0.90 else 0.0

    class_names = list(scores)
    probabilities = _softmax([scores[name] for name in class_names])
    best_index = int(np.argmax(probabilities))
    face_shape = class_names[best_index]
    confidence = float(probabilities[best_index])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vis = image.copy()
        for connection in FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL:
            start, end = face[connection.start], face[connection.end]
            p1 = (int(start.x * width), int(start.y * height))
            p2 = (int(end.x * width), int(end.y * height))
            cv2.line(vis, p1, p2, (0, 0, 255), 2)

        cv2.putText(vis, f"Face: {face_shape}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"Confidence: {confidence:.2f}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(str(output_path), vis)

    return {
        "face_shape": face_shape,
        "confidence": confidence,
        "features": {
            "length_width_ratio": float(length_width_ratio),
            "upper_width_ratio": None if upper is None else float(upper),
            "upper_middle_width_ratio": None if upper_middle is None else float(upper_middle),
            "middle_width_ratio": None if middle is None else float(middle),
            "lower_middle_width_ratio": None if lower_middle is None else float(lower_middle),
            "lower_width_ratio": None if lower is None else float(lower),
            "jaw_taper": float(jaw_taper),
        },
    }
