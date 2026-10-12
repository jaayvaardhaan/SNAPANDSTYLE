"""Landmark-based *estimated* facial outline classification.

Six face-shape labels are subjective: this is a geometric heuristic, not a
validated diagnostic or an identity/biometric classifier. Quality signals and
alternatives are provided to avoid false certainty.
"""
from pathlib import Path
from functools import lru_cache
import math

import cv2
import mediapipe as mp
import numpy as np
from image_utils import bgr_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"
SHAPES = ("oval", "round", "square", "oblong", "heart", "diamond")


@lru_cache(maxsize=1)
def _get_landmarker():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Face landmark model not found: {MODEL_PATH}")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=2,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def landmarker_is_loaded():
    return _get_landmarker.cache_info().currsize > 0


def _slice_polygon_width(polygon, y):
    """Width of a closed polygon at fixed y, in *rotated pixel* coordinates."""
    intersections = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        y1, y2 = float(a[1]), float(b[1])
        if abs(y2 - y1) < 1e-8 or not (min(y1, y2) <= y < max(y1, y2)):
            continue
        t = (y - y1) / (y2 - y1)
        intersections.append(float(a[0] + t * (b[0] - a[0])))
    return max(intersections) - min(intersections) if len(intersections) >= 2 else None


def _ordered_oval_indices():
    # Connections are edges; construct the contour in traversal order, rather
    # than using set iteration (which creates an invalid zig-zagging polygon).
    edges = mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
    graph = {}
    for edge in edges:
        graph.setdefault(edge.start, []).append(edge.end)
        graph.setdefault(edge.end, []).append(edge.start)
    start = min(graph)
    route, previous, current = [], None, start
    for _ in range(len(graph)):
        route.append(current)
        neighbors = graph[current]
        choices = [n for n in neighbors if n != previous]
        if not choices:
            raise RuntimeError("Face oval is not a continuous loop")
        nxt = choices[0]
        previous, current = current, nxt
        if current == start:
            break
    if len(route) < 25 or len(set(route)) != len(graph):
        raise RuntimeError("Could not order MediaPipe face-oval edges")
    return route


def _gaussian_distance(features, center, scale):
    return sum(((x - y) / s) ** 2 for x, y, s in zip(features, center, scale))


def classify_face_geometry(length_ratio, upper_ratio, lower_ratio):
    """Return qualitative shape and *relative* scores from outline geometry.

    Prototype values are hand-defined guidance, not learned from population
    ground truth. Approximate labels can be unstable near category boundaries.
    """
    x = (length_ratio, upper_ratio, lower_ratio)
    templates = {
        "oval":    ((1.31, .85, .79), (.20, .15, .14)),
        "round":   ((1.10, .86, .85), (.17, .15, .14)),
        "square":  ((1.19, .87, .92), (.19, .15, .12)),
        "oblong":  ((1.53, .86, .86), (.18, .16, .15)),
        "heart":   ((1.29, .93, .66), (.20, .12, .13)),
        "diamond": ((1.30, .73, .68), (.20, .11, .13)),
    }
    logits = np.array([-0.5 * _gaussian_distance(x, *templates[k]) for k in SHAPES])
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    ranking = sorted(zip(SHAPES, weights), key=lambda p: -p[1])
    return ranking[0][0], [(name, round(float(prob), 4)) for name, prob in ranking]


def analyze_face_shape(image_path, output_path=None):
    bgr = bgr_image(image_path)
    h, w = bgr.shape[:2]
    if min(h, w) < 128:
        raise ValueError("Face image too small; use at least 128px width and height")
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detection = _get_landmarker().detect(mp_image)
    if len(detection.face_landmarks) != 1:
        raise ValueError(f"Expected exactly one visible face, found {len(detection.face_landmarks)}")
    face = detection.face_landmarks[0]

    contour_ids = _ordered_oval_indices()
    contour = np.array([(face[i].x * w, face[i].y * h) for i in contour_ids], dtype=np.float64)
    left_eye, right_eye = [(face[i].x * w, face[i].y * h) for i in (33, 263)]
    eye_pair = sorted((left_eye, right_eye), key=lambda p: p[0])
    roll = math.degrees(math.atan2(eye_pair[1][1] - eye_pair[0][1], eye_pair[1][0] - eye_pair[0][0]))

    eye_mid = np.mean(eye_pair, axis=0)
    theta = -math.radians(roll)
    rot = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta), math.cos(theta)]], dtype=np.float64)
    aligned = (contour - eye_mid) @ rot.T
    minxy = aligned.min(axis=0)
    maxxy = aligned.max(axis=0)
    face_h = maxxy[1] - minxy[1]
    if face_h < 30:
        raise ValueError("Face landmarks are too small or invalid")
    levels = {"upper": .25, "upper_middle": .35, "middle": .50,
              "lower_middle": .65, "lower": .75}
    widths = {name: _slice_polygon_width(aligned, minxy[1] + frac * face_h)
              for name, frac in levels.items()}
    if any(value is None or value <= 0 for value in widths.values()):
        raise ValueError("Could not measure face outline at required heights")
    max_width = max(widths.values())
    length_ratio = float(face_h / max_width)
    normalized = {name: float(value / max_width) for name, value in widths.items()}
    label, ranking = classify_face_geometry(length_ratio, normalized["upper"], normalized["lower"])

    # A very rough lateral pose proxy; no camera calibration / PnP claimed.
    face_center_x = (float(np.min(contour[:, 0])) + float(np.max(contour[:, 0]))) / 2
    nose_x = float(face[1].x * w)
    image_fw = float(np.ptp(contour[:, 0]))
    yaw_proxy = abs(nose_x - face_center_x) / max(image_fw / 2, 1)
    warnings = []
    if abs(roll) > 12:
        warnings.append("head_tilt")
    if yaw_proxy > .22:
        warnings.append("head_turn_or_asymmetry")
    if image_fw < w * .19:
        warnings.append("face_too_small")
    if np.any(contour[:, 0] < 2) or np.any(contour[:, 0] > w - 3):
        warnings.append("face_clipped")

    # Do not imply that softmax distances are calibrated probabilities.
    margin = ranking[0][1] - ranking[1][1]
    certainty = float(np.clip(.38 + .36 * margin, .35, .70))
    certainty *= .77 if warnings else 1.0

    x_min, y_min = contour.min(axis=0)
    x_max, y_max = contour.max(axis=0)
    fw, fh = x_max - x_min, y_max - y_min
    # Cheeks are more robust for skin analysis than forehead or whole-body skin.
    # Each item is (cx, cy, rx, ry) in pixel coordinates.
    eye_y = float(eye_mid[1])
    mouth_y = float(np.mean([face[13].y, face[14].y]) * h)
    cheek_y = eye_y + .47 * (mouth_y - eye_y)
    cheeks = [(float(x_min + fw * frac), cheek_y, float(fw * .115), float(fh * .09))
              for frac in (.29, .71)]

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        vis = bgr.copy()
        pts = np.rint(contour).astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 200, 0), 2)
        for cx, cy, rx, ry in cheeks:
            cv2.ellipse(vis, (round(cx), round(cy)), (max(1, round(rx)), max(1, round(ry))), 0, 0, 360, (0, 160, 255), 1)
        cv2.putText(vis, f"Estimated face: {label}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, .68, (255, 255, 255), 2)
        if not cv2.imwrite(str(path), vis):
            raise OSError(f"Failed to save {path}")

    return {
        "face_shape": label,
        "confidence": round(float(certainty), 4),
        "features": {"length_width_ratio": round(length_ratio, 4),
                     **{f"{name}_width_ratio": round(v, 4) for name, v in normalized.items()},
                     "roll_degrees": round(roll, 2),
                     "yaw_proxy": round(yaw_proxy, 3)},
        "candidates": [{"label": name, "relative_weight": score} for name, score in ranking[:3]],
        "quality_warnings": warnings,
        "sampling": {"face_bbox": tuple(map(float, (x_min, y_min, x_max, y_max))), "cheek_ellipses": cheeks},
    }
