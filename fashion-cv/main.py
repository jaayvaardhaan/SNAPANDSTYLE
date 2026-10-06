import argparse
import json
from pathlib import Path

from face_landmarks import analyze_face_shape
from face_parsing import parse_face
from hair_analysis import analyze_hair
from skin_analysis import analyze_skin


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "input"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_input_image(input_dir=DEFAULT_INPUT_DIR):
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    preferred = input_dir / "test.jpg"
    if preferred.is_file():
        return preferred

    images = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not images:
        raise FileNotFoundError(f"No supported image found in: {input_dir}")
    return max(images, key=lambda p: p.stat().st_mtime)


def run_pipeline(image_path=None, output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path) if image_path else find_input_image()

    face = analyze_face_shape(image_path, output_dir / "face_shape.jpg")
    parsing = parse_face(image_path, output_dir)
    hair = analyze_hair(image_path, parsing["hair_mask"], output_dir / "hair_analysis.jpg")
    skin = analyze_skin(image_path, parsing["skin_mask"], output_dir / "skin_analysis.jpg")

    profile = {
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
    }

    profile_path = output_dir / "appearance_profile.json"
    profile_path.write_text(json.dumps(profile, indent=4), encoding="utf-8")
    return profile, profile_path


def main():
    parser = argparse.ArgumentParser(description="Fashion CV appearance analysis")
    parser.add_argument("image", nargs="?", help="Optional path to an input image")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for generated outputs")
    args = parser.parse_args()

    try:
        profile, profile_path = run_pipeline(args.image, args.output_dir)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Error: {exc}") from exc

    print(json.dumps(profile, indent=4))
    print(f"Saved: {profile_path}")


if __name__ == "__main__":
    main()
