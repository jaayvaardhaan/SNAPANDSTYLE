from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "resnet18.onnx"
INPUT_SIZE = 512
SKIN_CLASS, CLOTH_CLASS, HAIR_CLASS = 1, 16, 17


@lru_cache(maxsize=1)
def _get_session():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Face parsing model not found: {MODEL_PATH}")

    available = ort.get_available_providers()
    preferred = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider") if p in available]
    if not preferred:
        preferred = available
    return ort.InferenceSession(str(MODEL_PATH), providers=preferred)


def parse_face(image_path, output_dir="output"):
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    height, width = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = ((resized - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)

    session = _get_session()
    input_name = session.get_inputs()[0].name
    prediction = session.run(None, {input_name: tensor})[0]
    class_mask = np.argmax(np.squeeze(prediction, axis=0), axis=0).astype(np.uint8)
    labels = cv2.resize(class_mask, (width, height), interpolation=cv2.INTER_NEAREST)

    paths = {
        "label_mask": output_dir / "face_parsing_labels.png",
        "skin_mask": output_dir / "skin_mask.png",
        "hair_mask": output_dir / "hair_mask.png",
        "cloth_mask": output_dir / "cloth_mask.png",
        "visualization": output_dir / "face_parsing_mask.png",
    }

    masks = {
        "skin_mask": (labels == SKIN_CLASS).astype(np.uint8) * 255,
        "hair_mask": (labels == HAIR_CLASS).astype(np.uint8) * 255,
        "cloth_mask": (labels == CLOTH_CLASS).astype(np.uint8) * 255,
    }

    cv2.imwrite(str(paths["label_mask"]), labels)
    for key, mask in masks.items():
        cv2.imwrite(str(paths[key]), mask)

    colors = np.array([
        [0, 0, 0], [120, 180, 255], [100, 255, 100], [100, 220, 100], [0, 255, 0],
        [0, 200, 0], [255, 0, 255], [255, 180, 80], [255, 150, 60], [255, 200, 100],
        [180, 120, 80], [80, 80, 255], [100, 100, 255], [120, 120, 255], [80, 180, 180],
        [100, 200, 200], [180, 180, 180], [80, 50, 200], [255, 255, 0],
    ], dtype=np.uint8)
    overlay = cv2.addWeighted(image, 0.60, colors[labels], 0.40, 0)
    cv2.imwrite(str(paths["visualization"]), overlay)

    return {
        "skin_mask": str(paths["skin_mask"]),
        "hair_mask": str(paths["hair_mask"]),
        "cloth_mask": str(paths["cloth_mask"]),
        "label_mask": str(paths["label_mask"]),
        "statistics": {name.replace("_mask", "_pixels"): int(np.count_nonzero(mask)) for name, mask in masks.items()},
    }
