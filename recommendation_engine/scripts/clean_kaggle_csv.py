import csv
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "dataset"
    / "kaggle_fashion"
    / "styles.csv"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "dataset"
    / "kaggle_fashion"
    / "styles_clean.csv"
)


COLUMNS = [
    "id",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "year",
    "usage",
    "productDisplayName"
]


total_rows = 0
fixed_rows = 0
invalid_rows = 0


with open(INPUT_FILE, "r", encoding="utf-8") as infile, \
     open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as outfile:

    writer = csv.writer(outfile)

    writer.writerow(COLUMNS)

    for line_number, line in enumerate(infile, start=1):

        # Skip completely empty lines
        if not line.strip():
            continue

        fields = line.rstrip("\n\r").split(",")

        total_rows += 1

        # We need at least the 10 expected fields
        if len(fields) < 10:
            print(
                f"Skipping invalid row {line_number}: "
                f"only {len(fields)} fields"
            )
            invalid_rows += 1
            continue

        # First 9 fields are fixed.
        # Everything after that belongs to productDisplayName.
        fixed_fields = fields[:9]

        product_name = ",".join(fields[9:])

        row = fixed_fields + [product_name]

        if len(fields) > 10:
            fixed_rows += 1

        writer.writerow(row)


print()
print("=" * 50)
print("CSV CLEANING COMPLETE")
print("=" * 50)

print("Rows processed:", total_rows)
print("Rows containing extra commas:", fixed_rows)
print("Invalid rows skipped:", invalid_rows)

print()
print("Clean CSV:")
print(OUTPUT_FILE)