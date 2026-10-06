from pathlib import Path
import math

import cv2
import numpy as np


HAIR_COLORS = ("black", "dark_brown", "brown", "light_brown", "blonde", "red_or_auburn", "gray_or_white")


def _softmax(values):
    values = np.asarray(values, dtype=np.float64)
    values -= values.max()
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def analyze_hair(image_path, hair_mask_path, output_path=None):
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(hair_mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Could not load hair mask: {hair_mask_path}")

    height, width = image.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    mask = ((mask > 0) * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    ys, xs = np.where(mask > 0)
    if len(xs) < 150:
        raise ValueError("Not enough hair pixels.")

    pixels = image[ys, xs]
    lab = cv2.cvtColor(pixels[:, None, :], cv2.COLOR_BGR2LAB)[:, 0].astype(np.float32)
    hsv = cv2.cvtColor(pixels[:, None, :], cv2.COLOR_BGR2HSV)[:, 0].astype(np.float32)

    low, high = np.percentile(lab[:, 0], (5, 95))
    stable = (lab[:, 0] >= low) & (lab[:, 0] <= high)
    if stable.sum() < 150:
        stable[:] = True

    median_lab = np.median(lab[stable], axis=0)
    median_hsv = np.median(hsv[stable], axis=0)

    lightness = float(median_lab[0] / 255.0 * 100.0)
    a = float(median_lab[1] - 128.0)
    b = float(median_lab[2] - 128.0)
    chroma = math.hypot(a, b)
    hue, saturation, value = map(float, median_hsv)

    scores = {name: 0.0 for name in HAIR_COLORS}

    if lightness < 24 and chroma < 9:
        scores["black"] += 3.0
    if lightness < 40 and chroma >= 8 and b >= 5:
        scores["dark_brown"] += 3.0
    if 35 <= lightness < 58 and chroma >= 8 and b >= 5:
        scores["brown"] += 3.0
    if 50 <= lightness < 72 and chroma >= 7 and b >= 7:
        scores["light_brown"] += 3.0
    if lightness >= 65 and chroma >= 8 and b >= 5:
        scores["blonde"] += 3.0
    if a > 10 and b > 5 and a > b and chroma >= 10:
        scores["red_or_auburn"] += 3.0
    if lightness >= 55 and chroma < 10 and saturation < 50:
        scores["gray_or_white"] += 3.0

    centers = {"black": 18.0, "dark_brown": 30.0, "brown": 46.0, "light_brown": 62.0, "blonde": 78.0}
    spreads = {"black": 10.0, "dark_brown": 15.0, "brown": 15.0, "light_brown": 15.0, "blonde": 18.0}
    for name, center in centers.items():
        scores[name] += max(0.0, 2.0 - abs(lightness - center) / spreads[name])

    if a > b:
        scores["red_or_auburn"] += min(1.0, chroma / 25.0) * 1.5

    probabilities = _softmax([scores[name] for name in HAIR_COLORS])
    best_index = int(np.argmax(probabilities))
    hair_color = HAIR_COLORS[best_index]
    confidence = float(probabilities[best_index])

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vis = image.copy()
        vis[ys, xs] = (0, 255, 0)
        cv2.putText(vis, f"Hair: {hair_color}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(vis, f"Confidence: {confidence:.2f}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(str(output_path), vis)

    return {
        "hair_color": hair_color,
        "confidence": confidence,
        "measurements": {
            "lab_lightness": round(lightness, 2),
            "lab_a": round(a, 2),
            "lab_b": round(b, 2),
            "chroma": round(chroma, 2),
            "hsv_hue": round(hue, 2),
            "hsv_saturation": round(saturation, 2),
            "hsv_value": round(value, 2),
        },
    }
