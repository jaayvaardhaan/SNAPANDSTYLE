import json
import csv
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset" / "deepfashion2"
VALIDATION_DIR = DATASET_DIR / "validation"

IMAGE_DIR = VALIDATION_DIR / "image"
ANNO_DIR = VALIDATION_DIR / "annos"

PROCESSED_DIR = PROJECT_DIR / "dataset" / "processed"
OUTPUT_CSV = PROCESSED_DIR / "clothing_items.csv"


# ============================================================
# SETTINGS
# ============================================================

# We are processing only 500 images for the first test.
MAX_IMAGES = 500


# ============================================================
# DEEPFASHION2 CATEGORY MAPPING
# ============================================================

CATEGORY_MAP = {
    1: "TOP",
    2: "TOP",
    3: "OUTERWEAR",
    4: "OUTERWEAR",
    5: "TOP",
    6: "TOP",
    7: "BOTTOM",
    8: "BOTTOM",
    9: "BOTTOM",
    10: "DRESS",
    11: "DRESS",
    12: "DRESS",
    13: "DRESS"
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIND JSON FILES
# ============================================================

json_files = sorted(ANNO_DIR.glob("*.json"))

print("Total annotation files found:", len(json_files))

if MAX_IMAGES is not None:
    json_files = json_files[:MAX_IMAGES]

print("Processing:", len(json_files), "images")


# ============================================================
# CSV COLUMNS
# ============================================================

fieldnames = [
    "item_id",
    "image_path",
    "category_id",
    "category_name",
    "our_category",
    "style_id",
    "source",
    "pair_id",
    "bounding_box",
    "scale",
    "occlusion",
    "zoom_in",
    "viewpoint"
]


# ============================================================
# PROCESS DATASET
# ============================================================

rows = []

for index, json_file in enumerate(json_files, start=1):

    # --------------------------------------------------------
    # Read JSON
    # --------------------------------------------------------

    with open(json_file, "r") as f:
        data = json.load(f)

    image_name = json_file.stem + ".jpg"

    image_path = IMAGE_DIR / image_name

    # Make sure corresponding image exists
    if not image_path.exists():
        print("Image missing:", image_name)
        continue

    # --------------------------------------------------------
    # Process clothing items
    # --------------------------------------------------------

    for key, item in data.items():

        if not key.startswith("item"):
            continue

        category_id = item.get("category_id")

        category_name = item.get("category_name")

        our_category = CATEGORY_MAP.get(
            category_id,
            "OTHER"
        )

        item_id = f"{json_file.stem}_{key}"

        row = {
            "item_id": item_id,

            "image_path": str(
                image_path.relative_to(PROJECT_DIR)
            ),

            "category_id": category_id,

            "category_name": category_name,

            "our_category": our_category,

            "style_id": item.get("style"),

            "source": data.get("source"),

            "pair_id": data.get("pair_id"),

            "bounding_box": item.get("bounding_box"),

            "scale": item.get("scale"),

            "occlusion": item.get("occlusion"),

            "zoom_in": item.get("zoom_in"),

            "viewpoint": item.get("viewpoint")
        }

        rows.append(row)

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if index % 50 == 0:
        print(
            f"Processed {index}/{len(json_files)} images..."
        )


# ============================================================
# WRITE CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(rows)


# ============================================================
# SUMMARY
# ============================================================

print("\n===================================")
print("PROCESSING COMPLETE")
print("===================================")

print("Images processed:", len(json_files))
print("Clothing items extracted:", len(rows))

print("\nOutput file:")
print(OUTPUT_CSV)