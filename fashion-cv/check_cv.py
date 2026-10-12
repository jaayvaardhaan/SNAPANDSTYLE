"""Snap&Style CV preflight. It never writes the supplied photo or CV outputs."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
REQUIRED_PACKAGES = (
    "numpy", "opencv-python", "mediapipe", "onnxruntime", "fastapi", "uvicorn"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_dependencies():
    missing = []
    versions = {}
    for package in REQUIRED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        raise RuntimeError("Missing Python packages: " + ", ".join(missing))
    return versions


def check_models():
    expected = {
        "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        "resnet18.onnx": "0d9bd318e46987c3bdbfacae9e2c0f461cae1c6ac6ea6d43bbe541a91727e33f",
    }
    result = {}
    for name, expected_hash in expected.items():
        path = MODEL_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing model: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Model checksum mismatch for {name}. Expected {expected_hash}, got {actual_hash}"
            )
        result[name] = {"bytes": path.stat().st_size, "sha256": actual_hash}
    return result


def deep_check():
    from face_landmarks import _get_landmarker
    from face_parsing import _get_session

    _get_landmarker()
    session = _get_session()
    return {
        "face_landmarker": "initialized",
        "face_parser_providers": session.get_providers(),
        "face_parser_inputs": [item.name for item in session.get_inputs()],
    }


def analyze_image(path):
    from main import analyze_array

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    profile = analyze_array(image)
    public_keys = (
        "face_shape", "skin_tone", "skin_undertone", "hair_color",
        "confidence", "quality_warnings",
    )
    return {key: profile[key] for key in public_keys}


def main():
    parser = argparse.ArgumentParser(description="Validate Snap&Style CV installation")
    parser.add_argument("--deep", action="store_true", help="Initialize both model runtimes")
    parser.add_argument("--image", type=Path, help="Run in-memory inference on one image")
    args = parser.parse_args()

    report = {
        "status": "ready",
        "dependencies": check_dependencies(),
        "models": check_models(),
    }
    if args.deep or args.image:
        report["initialization"] = deep_check()
    if args.image:
        report["profile"] = analyze_image(args.image.resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(f"CV preflight failed: {exc}") from exc
