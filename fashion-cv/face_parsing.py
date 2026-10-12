"""BiSeNet face parsing for skin/hair masks (CelebAMask-HQ 19-class order).

Requires models/resnet18.onnx exported with RGB ImageNet-normalized input.
The model file must match that preprocessing; inspect/export metadata if not.
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from image_utils import bgr_image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "resnet18.onnx"
INPUT_SIZE = 512
SKIN_CLASS, CLOTH_CLASS, HAIR_CLASS = 1, 16, 17


@lru_cache(maxsize=1)
def _get_session():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Face parsing model not found: {MODEL_PATH}")
    available = ort.get_available_providers()
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    if not providers:
        raise RuntimeError(f"No supported ONNX execution provider. Installed: {available}")
    return ort.InferenceSession(str(MODEL_PATH), providers=providers)


def parser_is_loaded():
    return _get_session.cache_info().currsize > 0


def _largest_component(mask, min_area=50):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return mask
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < min_area:
        return np.zeros_like(mask)
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def _decode_logits(prediction):
    pred = np.asarray(prediction)
    if pred.ndim != 4 or pred.shape[0] != 1:
        raise ValueError(f"Unexpected parsing output shape: {pred.shape}")
    if pred.shape[1] == 19:  # NCHW
        labels = np.argmax(pred[0], axis=0)
    elif pred.shape[-1] == 19:  # NHWC
        labels = np.argmax(pred[0], axis=-1)
    else:
        raise ValueError(f"Expected 19 parsing classes, got: {pred.shape}")
    return labels.astype(np.uint8)


def parse_face(image_path, output_dir="output"):
    # output_dir=None: return masks as NumPy arrays, without writing files.
    in_memory = output_dir is None
    if not in_memory:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    image = bgr_image(image_path)
    h, w = image.shape[:2]
    if min(h, w) < 128:
        raise ValueError("Image is too small for reliable face parsing (min 128 px per side).")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # BiSeNet's conventional preprocessing: 512x512 RGB, ImageNet normalize.
    scaled = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb_float = scaled.astype(np.float32) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    tensor = ((rgb_float - mean) / std).transpose(2, 0, 1)[None]

    session = _get_session()
    in_spec = session.get_inputs()[0]
    if len(in_spec.shape) != 4:
        raise ValueError(f"Unexpected ONNX input shape: {in_spec.shape}")
    outputs = session.run(None, {in_spec.name: tensor})
    if not outputs:
        raise ValueError("ONNX parser returned no outputs")
    candidate = next((out for out in outputs if np.asarray(out).ndim == 4 and (np.asarray(out).shape[1] == 19 or np.asarray(out).shape[-1] == 19)), None)
    if candidate is None:
        raise ValueError("ONNX model did not return a 19-class segmentation tensor")
    labels = _decode_logits(candidate)
    labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)

    masks = {
        "skin_mask": (labels == SKIN_CLASS).astype(np.uint8) * 255,
        "hair_mask": (labels == HAIR_CLASS).astype(np.uint8) * 255,
        "cloth_mask": (labels == CLOTH_CLASS).astype(np.uint8) * 255,
    }

    # Avoid tiny misclassified islands but retain all substantial hair regions.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for key, mask in masks.items():
        masks[key] = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if in_memory:
        return {
            **masks,
            "statistics": {f"{key.removesuffix('_mask')}_pixels": int(np.count_nonzero(mask)) for key, mask in masks.items()},
        }

    paths = {key: output_dir / f"{key}.png" for key in masks}
    for key, mask in masks.items():
        if not cv2.imwrite(str(paths[key]), mask):
            raise OSError(f"Could not write {paths[key]}")

    label_path = output_dir / "face_parsing_labels.png"
    if not cv2.imwrite(str(label_path), labels):
        raise OSError(f"Could not write {label_path}")
    # Visualization is not an analysis input.
    palette = np.array([
        [0, 0, 0], [120, 180, 255], [100, 255, 100], [100, 220, 100], [0, 255, 0],
        [0, 200, 0], [255, 0, 255], [255, 180, 80], [255, 150, 60], [255, 200, 100],
        [180, 120, 80], [80, 80, 255], [100, 100, 255], [120, 120, 255], [80, 180, 180],
        [100, 200, 200], [180, 180, 180], [80, 50, 200], [255, 255, 0],
    ], dtype=np.uint8)
    vis_path = output_dir / "face_parsing_mask.jpg"
    if not cv2.imwrite(str(vis_path), cv2.addWeighted(image, 0.65, palette[labels], 0.35, 0)):
        raise OSError(f"Could not write {vis_path}")

    return {
        **{key: str(path) for key, path in paths.items()},
        "label_mask": str(label_path),
        "visualization": str(vis_path),
        "statistics": {f"{key.removesuffix('_mask')}_pixels": int(np.count_nonzero(mask)) for key, mask in masks.items()},
    }
