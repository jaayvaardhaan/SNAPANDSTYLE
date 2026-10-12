from pathlib import Path
import math

import cv2
import numpy as np
from cv_image_utils import bgr_image, grayscale_mask


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
    if C1p * C2p < 1e-12:
        dhp = 0.0
    elif abs(dh) <= 180:
        dhp = dh
    elif dh > 180:
        dhp = dh - 360
    else:
        dhp = dh + 360
    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    Lbp = (L1 + L2) / 2.0
    Cbp = (C1p + C2p) / 2.0
    if C1p * C2p < 1e-12:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
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



def _lab_pixels(bgr_pixels):
    px = np.asarray(bgr_pixels, dtype=np.uint8).reshape(-1, 3)
    data = cv2.cvtColor(px[:, None, :], cv2.COLOR_BGR2LAB)[:, 0].astype(np.float64)
    data[:, 0] *= 100.0 / 255.0
    data[:, 1:] -= 128.0
    return data


def _ellipse_mask(h, w, ellipses):
    canvas = np.zeros((h, w), dtype=np.uint8)
    for cx, cy, rx, ry in ellipses:
        cv2.ellipse(canvas, (round(cx), round(cy)),
                    (max(1, round(rx)), max(1, round(ry))),
                    0, 0, 360, 255, -1)
    return canvas


def _normalized_softmax(vals):
    arr = np.asarray(vals, dtype=np.float64)
    ex = np.exp(arr - arr.max())
    return ex / ex.sum()


def classify_skin_samples(lab_samples, per_cheek=None):
    """Classification from skin samples in CIELAB (L: 0-100, a/b centered).

    This is an *apparent photographed* skin shade; camera lighting/white balance
    can change the nearest MST swatch. Undertone in particular is not reliably
    recoverable from a single uncalibrated selfie.
    """
    lab_samples = np.asarray(lab_samples, dtype=np.float64).reshape(-1, 3)
    if len(lab_samples) < 180:
        raise ValueError("Insufficient visible skin for reliable analysis")
    L = lab_samples[:, 0]
    lo, hi = np.percentile(L, [8, 92])
    selected = lab_samples[(L >= lo) & (L <= hi)]
    if len(selected) < 180:
        selected = lab_samples
    center = np.median(selected, axis=0)
    repL, repa, repb = map(float, center)
    chroma = math.hypot(repa, repb)

    distances = np.array([_ciede2000(center, color) for color in MST_LAB])
    groups = {
        "light": (0, 1),
        "medium-light": (2, 3),
        "medium": (4, 5),
        "medium-deep": (6, 7),
        "deep": (8, 9),
    }
    # A group's nearest reference, not summed swatch probabilities that unfairly
    # favor groups spanning broad areas of Lab space.
    group_dist = {name: float(min(distances[list(indices)])) for name, indices in groups.items()}
    group_names = list(groups)
    shade_weights = _normalized_softmax([-group_dist[name] / 5.0 for name in group_names])
    rank = sorted(zip(group_names, shade_weights), key=lambda a: -a[1])
    tone = rank[0][0]
    tone_margin = float(rank[0][1] - rank[1][1])
    tone_certainty = float(np.clip(.42 + .48 * tone_margin, .35, .79))

    # Simple chromaticity proxy; explicitly NOT ground truth undertone.
    warmth = repb - repa
    if chroma < 13 or abs(warmth) < 7:
        undertone = "neutral"
    elif warmth >= 7 and repb >= 13:
        undertone = "warm"
    elif warmth <= -7 and repa >= 12:
        undertone = "cool"
    else:
        undertone = "neutral"

    # Confidence capped deliberately: illumination and white balance may flip
    # the result even if the score difference is large.
    undertone_certainty = float(np.clip(.34 + .008 * max(0, abs(warmth) - 5), .32, .56))
    warnings = ["undertone_unreliable_without_calibrated_light"]
    if min(distances) > 17:
        warnings.append("photographed_skin_far_from_reference_swatch")
        tone_certainty *= .7
    if repL < 18 or repL > 88:
        warnings.append("skin_exposure_outside_preferred_range")
        tone_certainty *= .8
        undertone_certainty *= .8
    if per_cheek and len(per_cheek) >= 2:
        if abs(float(per_cheek[0][0]) - float(per_cheek[1][0])) > 12:
            warnings.append("uneven_light_across_cheeks")
            tone_certainty *= .8
            undertone_certainty *= .8
        cheek_color_difference = _ciede2000(per_cheek[0], per_cheek[1])
        if cheek_color_difference > 14:
            warnings.append("cheek_colors_disagree_possible_shadow_or_color_cast")
            undertone_certainty *= .75

    return {
        "skin_tone": tone,
        "skin_undertone": undertone,
        "confidence": {"skin_tone": round(float(tone_certainty), 4),
                       "skin_undertone": round(float(undertone_certainty), 4)},
        "candidates": [{"label": name, "relative_weight": round(float(weight), 4)} for name, weight in rank[:3]],
        "quality_warnings": warnings,
        "measurements": {
            "lab_L": round(repL, 2), "lab_a": round(repa, 2),
            "lab_b": round(repb, 2), "chroma": round(chroma, 2),
            "closest_mst_index": int(np.argmin(distances) + 1),
            "closest_mst_delta_e00": round(float(np.min(distances)), 2),
            "sampled_skin_pixels": int(len(selected)),
        },
    }


def analyze_skin(image_path, skin_mask_path, output_path=None, face_sampling=None):
    bgr = bgr_image(image_path)
    mask = grayscale_mask(skin_mask_path)
    h, w = bgr.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, erode, iterations=1)
    warnings = []

    cheek_medians = []
    if face_sampling is not None:
        cheek_masks = [_ellipse_mask(h, w, [ellipse]) for ellipse in face_sampling["cheek_ellipses"]]
        for patch in cheek_masks:
            patch_mask = cv2.bitwise_and(mask, patch)
            pix = bgr[patch_mask != 0]
            if len(pix) >= 160:
                cheek_medians.append(np.median(_lab_pixels(pix), axis=0))
        union = _ellipse_mask(h, w, face_sampling["cheek_ellipses"])
        combined = cv2.bitwise_and(mask, union)
        if np.count_nonzero(combined) >= 350:
            mask = combined
        else:
            # Restrict fallback to the detected face; never sample visible hands.
            xmin, ymin, xmax, ymax = face_sampling["face_bbox"]
            face_mask = np.zeros((h, w), dtype=np.uint8)
            x1, x2 = max(0, round(xmin + .12 * (xmax - xmin))), min(w, round(xmax - .12 * (xmax - xmin)))
            y1, y2 = max(0, round(ymin + .25 * (ymax - ymin))), min(h, round(ymax - .14 * (ymax - ymin)))
            if x2 > x1 and y2 > y1:
                face_mask[y1:y2, x1:x2] = 255
            mask = cv2.bitwise_and(mask, face_mask)
            warnings.append("cheek_sampling_unavailable_using_central_face")
    else:
        warnings.append("face_guided_skin_sampling_unavailable")

    pix = bgr[mask != 0]
    if len(pix) < 200:
        raise ValueError("Not enough exposed, segmented facial skin; check pose/obstructions")
    stride = max(1, math.ceil(len(pix) / 100_000))
    result = classify_skin_samples(_lab_pixels(pix[::stride]), per_cheek=cheek_medians)
    result["quality_warnings"] = list(dict.fromkeys(warnings + result["quality_warnings"]))

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        vis = bgr.copy()
        layer = bgr.copy()
        layer[mask > 0] = (40, 190, 40)
        cv2.addWeighted(layer, .28, vis, .72, 0, vis)
        cv2.putText(vis, f"Tone (estimated): {result['skin_tone']}", (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, .64, (255, 255, 255), 2)
        if not cv2.imwrite(str(path), vis):
            raise OSError(f"Could not save {path}")
    return result
