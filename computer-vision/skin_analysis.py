from pathlib import Path
import math

import cv2
import numpy as np


MST_RGB = np.array([
    [246, 237, 228], [243, 231, 219], [247, 234, 208], [234, 218, 186], [215, 189, 150],
    [160, 126, 86], [130, 92, 67], [96, 65, 52], [58, 49, 42], [41, 36, 32],
], dtype=np.uint8)


def _rgb_to_lab(rgb):
    pixel = np.uint8([[rgb]])
    lab = cv2.cvtColor(pixel, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float64)
    return np.array([lab[0] / 255.0 * 100.0, lab[1] - 128.0, lab[2] - 128.0])


MST_LAB = np.array([_rgb_to_lab(rgb) for rgb in MST_RGB])


def _ciede2000(lab1, lab2):
    L1, a1, b1 = map(float, lab1)
    L2, a2, b2 = map(float, lab2)
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    C_bar = (C1 + C2) / 2.0
    G = 0.5 * (1 - math.sqrt(C_bar**7 / (C_bar**7 + 25.0**7)))

    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360.0

    dLp = L2 - L1
    dCp = C2p - C1p
    dh = h2p - h1p
    if abs(dh) <= 180:
        dhp = dh
    elif dh > 180:
        dhp = dh - 360
    else:
        dhp = dh + 360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    Lbp = (L1 + L2) / 2.0
    Cbp = (C1p + C2p) / 2.0
    if abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2.0
    elif h1p + h2p < 360:
        hbp = (h1p + h2p + 360) / 2.0
    else:
        hbp = (h1p + h2p - 360) / 2.0

    T = (1 - 0.17 * math.cos(math.radians(hbp - 30))
         + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6))
         - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dtheta = 30 * math.exp(-((hbp - 275) / 25) ** 2)
    RC = 2 * math.sqrt(Cbp**7 / (Cbp**7 + 25.0**7))
    SL = 1 + 0.015 * (Lbp - 50) ** 2 / math.sqrt(20 + (Lbp - 50) ** 2)
    SC = 1 + 0.045 * Cbp
    SH = 1 + 0.015 * Cbp * T
    RT = -math.sin(math.radians(2 * dtheta)) * RC

    return math.sqrt(
        (dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    )


def analyze_skin(image_path, skin_mask_path, output_path=None):
    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(skin_mask_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    if mask is None:
        raise FileNotFoundError(f"Could not load skin mask: {skin_mask_path}")

    height, width = image.shape[:2]
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    mask = ((mask > 0) * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    sampling_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    sampling_mask = cv2.erode(mask, sampling_kernel, iterations=1)

    ys, xs = np.where(sampling_mask > 0)
    if len(xs) < 200:
        raise ValueError("Not enough skin pixels.")

    pixels = image[ys, xs]
    lab_cv = cv2.cvtColor(pixels[:, None, :], cv2.COLOR_BGR2LAB)[:, 0].astype(np.float32)
    low, high = np.percentile(lab_cv[:, 0], (5, 95))
    stable = (lab_cv[:, 0] >= low) & (lab_cv[:, 0] <= high)
    if stable.sum() < 200:
        stable[:] = True

    stable_lab = lab_cv[stable]
    skin_lab = np.column_stack((stable_lab[:, 0] / 255.0 * 100.0, stable_lab[:, 1] - 128.0, stable_lab[:, 2] - 128.0))
    representative = np.median(skin_lab, axis=0)
    rep_L, rep_a, rep_b = map(float, representative)

    distances = np.array([_ciede2000(representative, ref) for ref in MST_LAB])
    logits = -distances / 5.0
    logits -= logits.max()
    mst_prob = np.exp(logits)
    mst_prob /= mst_prob.sum()

    tone_prob = {
        "light": float(mst_prob[0] + mst_prob[1]),
        "medium-light": float(mst_prob[2] + mst_prob[3]),
        "medium": float(mst_prob[4] + mst_prob[5]),
        "medium-deep": float(mst_prob[6] + mst_prob[7]),
        "deep": float(mst_prob[8] + mst_prob[9]),
    }
    skin_tone = max(tone_prob, key=tone_prob.get)

    lightness_consistency = 1.0 - min(1.0, float(np.std(skin_lab[:, 0])) / 35.0)
    tone_confidence = float(np.clip(tone_prob[skin_tone] * (0.70 + 0.30 * lightness_consistency), 0.0, 1.0))

    chroma = math.hypot(rep_a, rep_b)
    hue_angle = math.degrees(math.atan2(rep_b, rep_a)) % 360.0

    undertone_scores = {"warm": 0.0, "cool": 0.0, "neutral": 0.0}
    if rep_b > 15:
        undertone_scores["warm"] += 0.45
    if rep_b >= rep_a * 0.85:
        undertone_scores["warm"] += 0.30
    if rep_a > 20 and rep_a > rep_b * 1.15:
        undertone_scores["cool"] += 0.50
    if rep_b < 12:
        undertone_scores["cool"] += 0.25
    if chroma < 14:
        undertone_scores["neutral"] += 0.60
    if abs(rep_a - rep_b) <= 8 and chroma >= 14:
        undertone_scores["neutral"] += 0.35

    undertone = max(undertone_scores, key=undertone_scores.get)
    ordered = sorted(undertone_scores.values(), reverse=True)
    margin = ordered[0] - ordered[1]
    # Heuristic certainty only; intentionally never reports 1.0.
    undertone_confidence = 0.85 if margin >= 0.40 else 0.70 if margin >= 0.20 else 0.55 if margin >= 0.10 else 0.35

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vis = image.copy()
        vis[ys, xs] = (0, 255, 0)
        cv2.putText(vis, f"Tone: {skin_tone}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(vis, f"Undertone: {undertone}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imwrite(str(output_path), vis)

    return {
        "skin_tone": skin_tone,
        "skin_undertone": undertone,
        "confidence": {
            "skin_tone": tone_confidence,
            "skin_undertone": float(undertone_confidence),
        },
        "measurements": {
            "lab_L": round(rep_L, 2),
            "lab_a": round(rep_a, 2),
            "lab_b": round(rep_b, 2),
            "chroma": round(chroma, 2),
            "hue_angle": round(hue_angle, 2),
        },
    }
