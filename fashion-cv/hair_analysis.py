"""Robust, explainable *apparent* hair-color classification.

No single RGB selfie establishes natural/undyed hair colour. Dark hair in
warm light is intrinsically ambiguous, so include candidates and quality flags.
The numeric 'confidence' is a conservative heuristic, NOT model calibration.
"""
from pathlib import Path
import math

import cv2
import numpy as np
from image_utils import bgr_image, grayscale_mask

HAIR_COLORS = ("black", "dark_brown", "brown", "light_brown", "blonde", "red_or_auburn", "gray_or_white")


def _elliptical_distance(vector, center, spreads):
    return sum(((v - c) / s) ** 2 for v, c, s in zip(vector, center, spreads))


def classify_hair_pixels(bgr_pixels):
    """Pure classification function for repeatable tests and diagnostics.

    Use shadow-tolerant lower quartiles + body median, rather than letting
    warmer specular highlights dominate the whole-region average.
    """
    pixels = np.asarray(bgr_pixels, dtype=np.uint8).reshape(-1, 3)
    if len(pixels) < 120:
        raise ValueError("Not enough reliable hair pixels")
    lab = cv2.cvtColor(pixels[:, None, :], cv2.COLOR_BGR2LAB)[:, 0].astype(np.float64)
    hsv = cv2.cvtColor(pixels[:, None, :], cv2.COLOR_BGR2HSV)[:, 0].astype(np.float64)
    lvals = lab[:, 0] * (100.0 / 255.0)
    q10, q25, q50, q75, q90 = np.percentile(lvals, [10, 25, 50, 75, 90])

    # Lower quartile is a proxy for underlying strands; extremely dark pixels
    # are excluded because they are often shadow or a mask artefact.
    selected = (lvals >= q10) & (lvals <= q75)
    if selected.sum() < 120:
        selected = np.ones(len(lvals), dtype=bool)
    a = float(np.median(lab[selected, 1]) - 128.0)
    b = float(np.median(lab[selected, 2]) - 128.0)
    chroma = float(math.hypot(a, b))
    saturation = float(np.median(hsv[selected, 1]))
    hue = float(np.median(hsv[selected, 0]))
    vmedian = float(np.median(hsv[selected, 2]))
    dark_fraction = float(np.mean(lvals < 25.0))
    vector = (float(q25), float(q50), chroma)

    prototypes = {
        "black":       ((13, 20, 6),  (11, 15, 15)),
        "dark_brown":  ((23, 34, 14), (12, 15, 15)),
        "brown":       ((34, 48, 19), (12, 14, 17)),
        "light_brown": ((49, 62, 20), (13, 13, 18)),
        "blonde":      ((65, 77, 16), (16, 15, 20)),
    }
    scores = {key: -0.5 * _elliptical_distance(vector, *params)
              for key, params in prototypes.items()}

    # Black hair may carry gold/brown reflections. The *base strands* being
    # consistently dark is stronger evidence than yellow-ish specular pixels.
    if q25 < 19 and q50 < 38:
        scores["black"] += .55
    if q25 >= 20 and b > 4:
        scores["dark_brown"] += .24
    if q25 >= 30 and b > 6:
        scores["brown"] += .18
    if b < 2 and saturation < 25 and q50 < 35:
        scores["black"] += .15

    # Dedicated classes; not all bright non-dark hair is blonde.
    # a/b are centered OpenCV CIELAB components; gray is low-chroma.
    scores["red_or_auburn"] = (
        -0.5 * ((q50 - 49) / 23) ** 2
        + min(1.6, max(0.0, a - 6.0) / 10.0)
        + min(0.9, max(0.0, a - b + 2) / 12.0)
        - (2.0 if a < 16 else 0.0)
    )
    scores["gray_or_white"] = (
        -0.5 * ((q50 - 72) / 24) ** 2
        -0.5 * (chroma / 9) ** 2
        - (1.5 if q50 < 43 else 0.0)
        + (.5 if saturation < 35 else 0)
    )
    if b < 1 and q50 < 45:
        scores["dark_brown"] -= .3

    order = sorted(scores, key=scores.get, reverse=True)
    margin = float(scores[order[0]] - scores[order[1]])
    certainty = float(np.clip(.43 + .18 * margin, .35, .78))
    warnings = []
    if q50 < 12:
        warnings.append("too_dark_to_distinguish_black_and_brown")
        certainty *= .75
    if q90 - q10 > 48:
        warnings.append("uneven_hair_lighting")
        certainty *= .82
    if (order[0] in ("black", "dark_brown") and margin < .75):
        warnings.append("black_dark_brown_ambiguous")
    if len(pixels) < 1000:
        warnings.append("small_hair_region")
        certainty *= .8

    return {
        "hair_color": order[0],
        "confidence": round(certainty, 4),
        "candidates": [{"label": name, "relative_score": round(float(scores[name]), 3)} for name in order[:3]],
        "quality_warnings": warnings,
        "measurements": {
            "lab_lightness": round(float(q50), 2),
            "lab_lightness_p10": round(float(q10), 2),
            "lab_lightness_p25": round(float(q25), 2),
            "lab_lightness_p75": round(float(q75), 2),
            "lab_lightness_p90": round(float(q90), 2),
            "lab_a": round(a, 2), "lab_b": round(b, 2), "chroma": round(chroma, 2),
            "hsv_hue": round(hue, 2), "hsv_saturation": round(saturation, 2),
            "hsv_value": round(vmedian, 2), "dark_pixel_fraction": round(dark_fraction, 3),
            "sampled_hair_pixels": int(len(pixels)),
        },
    }


def analyze_hair(image_path, hair_mask_path, output_path=None, face_sampling=None):
    image = bgr_image(image_path)
    mask = grayscale_mask(hair_mask_path)
    h, w = image.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.erode(mask, kernel, iterations=1)
    raw_count = int(np.count_nonzero(mask))

    # For photos with multiple objects, sample near the actual detected face.
    if face_sampling is not None:
        xmin, ymin, xmax, ymax = face_sampling["face_bbox"]
        fw, fh = xmax - xmin, ymax - ymin
        x1, x2 = max(0, round(xmin - .42 * fw)), min(w, round(xmax + .42 * fw))
        y1, y2 = max(0, round(ymin - .65 * fh)), min(h, round(ymax + .20 * fh))
        roi = np.zeros_like(mask)
        if x2 > x1 and y2 > y1:
            roi[y1:y2, x1:x2] = 255
        localized = cv2.bitwise_and(mask, roi)
        if np.count_nonzero(localized) >= max(350, .10 * raw_count):
            mask = localized

    count = int(np.count_nonzero(mask))
    if count < 200:
        raise ValueError("Not enough segmented hair pixels; recapture with visible hair")
    ys, xs = np.where(mask != 0)
    # Bound CPU/memory cost for 4K inputs. Spatially deterministic sampling.
    stride = max(1, math.ceil(len(xs) / 100_000))
    result = classify_hair_pixels(image[ys[::stride], xs[::stride]])
    result["measurements"]["total_hair_pixels"] = count

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        vis = image.copy()
        layer = image.copy()
        layer[mask > 0] = (30, 180, 30)
        cv2.addWeighted(layer, .3, vis, .7, 0, vis)
        label = result["hair_color"].replace("_", " ")
        cv2.putText(vis, f"Hair (estimated): {label}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    .65, (250, 250, 250), 2)
        if not cv2.imwrite(str(path), vis):
            raise OSError(f"Could not save {path}")
    return result
