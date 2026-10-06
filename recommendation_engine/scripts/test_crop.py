from PIL import Image
from pathlib import Path

# Original image
image_path = Path(
    "dataset/deepfashion2/validation/image/000001.jpg"
)

# Output folder
output_dir = Path(
    "dataset/processed/test_crops"
)
output_dir.mkdir(parents=True, exist_ok=True)

# Open image
image = Image.open(image_path)

# Bounding boxes from 000001.json
items = {
    "000001_item1": [199, 190, 287, 269],
    "000001_item2": [204, 189, 293, 414]
}

for item_id, box in items.items():

    x1, y1, x2, y2 = box

    # Crop
    cropped = image.crop((x1, y1, x2, y2))

    # Save
    output_path = output_dir / f"{item_id}.jpg"

    cropped.save(output_path)

    print(
        f"{item_id}: "
        f"{cropped.size} -> {output_path}"
    )

print("\nCropping complete.")