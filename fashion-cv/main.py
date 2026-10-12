"""Snap&Style four-feature appearance CV pipeline.

CLI examples:
  python main.py
  python main.py ../images/to_process/capture_123.jpg
  python main.py path/to/image.jpg --output-dir output/jobs/capture_123
"""
import argparse
import json
import os
from pathlib import Path

import cv2

from face_landmarks import analyze_face_shape
from face_parsing import parse_face
from hair_analysis import analyze_hair
from skin_analysis import analyze_skin

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "input"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _unknown_hair(reason):
    """Preserve the other CV results when hair is not visible enough to classify."""
    return {
        "hair_color": "unknown",
        "confidence": 0.0,
        "candidates": [],
        "quality_warnings": ["hair_not_visible_enough_to_classify"],
        "measurements": {"reason": str(reason)},
    }


def find_input_image(input_dir=DEFAULT_INPUT_DIR):
    folder = Path(input_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing input folder: {folder}")
    preferred = folder / "test.jpg"  # Keep existing behavior for manual testing.
    if preferred.is_file():
        return preferred
    images = [path for path in folder.iterdir()
              if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not images:
        raise FileNotFoundError(f"No images in {folder}")
    return max(images, key=lambda path: path.stat().st_mtime_ns)


def _atomic_write_json(path, document):
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)



def build_profile(face, parsing, hair, skin, width, height):
    """Return the standard appearance JSON without touching the filesystem."""
    warnings = list(dict.fromkeys(face["quality_warnings"] + hair["quality_warnings"] + skin["quality_warnings"]))
    profile = {
        # These keys/labels are retained for the existing frontend/recommender.
        "face_shape": face["face_shape"],
        "skin_tone": skin["skin_tone"],
        "skin_undertone": skin["skin_undertone"],
        "hair_color": hair["hair_color"],
        "confidence": {
            "face_shape": float(face["confidence"]),
            "skin_tone": float(skin["confidence"]["skin_tone"]),
            "skin_undertone": float(skin["confidence"]["skin_undertone"]),
            "hair_color": float(hair["confidence"]),
        },
        "quality_warnings": warnings,
        "analysis_info": {
            "version": "2.0",
            "confidence_note": "Relative heuristic reliability, NOT calibrated probability of correctness",
            "hair_note": "Apparent photographed hair color; warm lighting can blur black/dark brown",
            "undertone_note": "Undertone from an uncalibrated selfie is inherently uncertain",
            "image_size": {"width": width, "height": height},
            "face": {key: val for key, val in face.items() if key not in ("sampling", "face_shape", "confidence")},
            "hair": {key: val for key, val in hair.items() if key not in ("hair_color", "confidence")},
            "skin": {key: val for key, val in skin.items() if key not in ("skin_tone", "skin_undertone", "confidence")},
            "segmentation": parsing["statistics"],
        },
    }
    return profile

def run_pipeline(image_path=None, output_dir=DEFAULT_OUTPUT_DIR):
    """Analyze one *explicit image*, return (profile, output_json_path)."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "appearance_profile.json"
    # Remove stale output *before* any input validation; failed reruns cannot
    # look successful because an older JSON happened to be present.
    json_path.unlink(missing_ok=True)
    image = Path(image_path).resolve() if image_path is not None else find_input_image().resolve()
    if not image.is_file() or image.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise FileNotFoundError(f"Unsupported or missing input image: {image}")
    test_image = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if test_image is None:
        raise ValueError(f"Could not decode input image: {image}")
    h, w = test_image.shape[:2]
    if min(w, h) < 256:
        raise ValueError("Input image is too small for reliable facial features (minimum 256px each side)")

    print(f"[CV] Processing: {image.name}", flush=True)
    face = analyze_face_shape(image, output_dir / "face_shape.jpg")
    print("[CV] Face landmarks ready", flush=True)
    parsing = parse_face(image, output_dir)
    print("[CV] Face parsing ready", flush=True)
    try:
        hair = analyze_hair(image, parsing["hair_mask"], output_dir / "hair_analysis.jpg",
                            face_sampling=face["sampling"])
    except ValueError as exc:
        if "hair" not in str(exc).lower():
            raise
        hair = _unknown_hair(exc)
    print("[CV] Hair analysis ready", flush=True)
    skin = analyze_skin(image, parsing["skin_mask"], output_dir / "skin_analysis.jpg",
                        face_sampling=face["sampling"])
    print("[CV] Skin analysis ready", flush=True)

    profile = build_profile(face, parsing, hair, skin, w, h)
    _atomic_write_json(json_path, profile)
    print(f"[CV] Saved: {json_path}", flush=True)
    return profile, json_path


def analyze_array(image_bgr):
    """Analyze a BGR NumPy image in RAM. No capture, mask, or JSON is saved."""
    from image_utils import bgr_image
    image_bgr = bgr_image(image_bgr.get() if isinstance(image_bgr, cv2.UMat) else image_bgr)
    h, w = image_bgr.shape[:2]
    if min(w, h) < 256:
        raise ValueError("Image too small (minimum 256 pixels per side)")
    face = analyze_face_shape(image_bgr, output_path=None)
    parsing = parse_face(image_bgr, output_dir=None)
    try:
        hair = analyze_hair(image_bgr, parsing["hair_mask"], output_path=None,
                            face_sampling=face["sampling"])
    except ValueError as exc:
        if "hair" not in str(exc).lower():
            raise
        hair = _unknown_hair(exc)
    skin = analyze_skin(image_bgr, parsing["skin_mask"], output_path=None,
                        face_sampling=face["sampling"])
    return build_profile(face, parsing, hair, skin, w, h)


def main():
    parser = argparse.ArgumentParser(description="Snap&Style appearance analysis")
    parser.add_argument("image", nargs="?", help="Image path (default: input/test.jpg)")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    try:
        profile, out = run_pipeline(args.image, args.output_dir)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"[CV ERROR] {exc}\n")
    print(json.dumps({key: profile[key] for key in (
        "face_shape", "skin_tone", "skin_undertone", "hair_color", "confidence", "quality_warnings")}, indent=2))
    print(f"Result JSON: {out}")


if __name__ == "__main__":
    main()
