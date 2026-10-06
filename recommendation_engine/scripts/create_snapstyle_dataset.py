import pandas as pd
import os

# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_FILE = "dataset/kaggle_fashion/styles_clean.csv"
OUTPUT_FILE = "dataset/kaggle_fashion/snapstyle_items.csv"
IMAGE_DIR = "dataset/kaggle_fashion/images"

# --------------------------------------------------
# Article type -> SnapStyle category
# --------------------------------------------------

CATEGORY_MAP = {

    # -------------------------
    # TOPWEAR
    # -------------------------
    "Tshirts": "topwear",
    "Lounge Tshirts": "topwear",
    "Shirts": "topwear",
    "Tops": "topwear",
    "Kurtas": "topwear",
    "Kurtis": "topwear",
    "Tunics": "topwear",
    "Sweatshirts": "topwear",
    "Sweaters": "topwear",
    "Jackets": "topwear",
    "Blazers": "topwear",
    "Waistcoat": "topwear",
    "Shrug": "topwear",
    "Nehru Jackets": "topwear",

    # -------------------------
    # BOTTOMWEAR
    # -------------------------
    "Jeans": "bottomwear",
    "Trousers": "bottomwear",
    "Shorts": "bottomwear",
    "Track Pants": "bottomwear",
    "Lounge Pants": "bottomwear",
    "Capris": "bottomwear",
    "Leggings": "bottomwear",
    "Jeggings": "bottomwear",
    "Skirts": "bottomwear",
    "Patiala": "bottomwear",
    "Salwar": "bottomwear",
    "Churidar": "bottomwear",
    "Tights": "bottomwear",

    # -------------------------
    # ONE-PIECE
    # -------------------------
    "Dresses": "one_piece",
    "Jumpsuit": "one_piece",
    "Rompers": "one_piece",
    "Sarees": "one_piece",
    "Kurta Sets": "one_piece",
    "Salwar and Dupatta": "one_piece",
    "Lehenga Choli": "one_piece",
    "Clothing Set": "one_piece",

    # -------------------------
    # FOOTWEAR
    # -------------------------
    "Casual Shoes": "footwear",
    "Sports Shoes": "footwear",
    "Formal Shoes": "footwear",
    "Heels": "footwear",
    "Flats": "footwear",
    "Flip Flops": "footwear",
    "Sandals": "footwear",
    "Sports Sandals": "footwear",
    "Booties": "footwear",
}


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original rows: {len(df)}")


# --------------------------------------------------
# Remove invalid gender/header row
# --------------------------------------------------

df = df[
    df["gender"].isin([
        "Men",
        "Women",
        "Unisex"
    ])
].copy()

print(f"After gender filtering: {len(df)}")


# --------------------------------------------------
# Keep only article types used by SnapStyle
# --------------------------------------------------

df["our_category"] = df["articleType"].map(CATEGORY_MAP)

df = df[df["our_category"].notna()].copy()

print(f"After clothing/footwear filtering: {len(df)}")


# --------------------------------------------------
# Create image path
# --------------------------------------------------

df["image_path"] = df["id"].astype(str) + ".jpg"


# --------------------------------------------------
# Check whether image exists
# --------------------------------------------------

df["image_exists"] = df["image_path"].apply(
    lambda x: os.path.exists(os.path.join(IMAGE_DIR, x))
)


# --------------------------------------------------
# Select final columns
# --------------------------------------------------

df = df[
    [
        "id",
        "gender",
        "articleType",
        "our_category",
        "baseColour",
        "season",
        "year",
        "usage",
        "productDisplayName",
        "image_path",
        "image_exists"
    ]
]


# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print()
print("SnapStyle dataset created successfully!")
print(f"Output: {OUTPUT_FILE}")
print(f"Rows: {len(df)}")

print()
print("Category distribution:")
print(df["our_category"].value_counts().to_string())

print()
print("Gender distribution:")
print(df["gender"].value_counts().to_string())

print()
print("Image availability:")
print(df["image_exists"].value_counts().to_string())
