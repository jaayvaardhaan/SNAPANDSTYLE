"""Image/mask adapters supporting both in-memory API and existing file-based CLI."""
from pathlib import Path
import cv2
import numpy as np


def bgr_image(source):
    if isinstance(source, np.ndarray):
        if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] != 3:
            raise ValueError("Expected a uint8 BGR image with shape (height, width, 3)")
        return np.ascontiguousarray(source)
    image = cv2.imread(str(Path(source)), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {source}")
    return image


def grayscale_mask(source):
    if isinstance(source, np.ndarray):
        if source.dtype != np.uint8 or source.ndim != 2:
            raise ValueError("Expected a uint8 grayscale mask")
        return source
    mask = cv2.imread(str(Path(source)), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read segmentation mask: {source}")
    return mask
