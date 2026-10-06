import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT_FILE = "dataset/kaggle_fashion/snapstyle_items.csv"
OUTPUT_FILE = "dataset/kaggle_fashion/snapstyle_items_tagged.csv"


# --------------------------------------------------
# Style tags
#
# Usage gives the broad style.
# Article type only adds characteristics.
# --------------------------------------------------

def get_style_tags(row):

    usage = row["usage"]
    article = row["articleType"]

    tags = set()

    # -------------------------
    # Usage-based style
    # -------------------------

    if pd.notna(usage):

        if usage == "Casual":
            tags.update(["casual", "everyday"])

        elif usage == "Sports":
            tags.update(["sports", "active"])

        elif usage == "Ethnic":
            tags.update(["ethnic", "traditional"])

        elif usage == "Formal":
            tags.update(["formal", "professional"])

        elif usage == "Smart Casual":
            tags.add("smart_casual")

        elif usage == "Party":
            tags.add("party")

        elif usage == "Travel":
            tags.add("travel")

    # -------------------------
    # Article characteristics
    # -------------------------

    if article in ["Tshirts", "Lounge Tshirts"]:
        tags.add("basic")

    if article in ["Shirts", "Tops", "Tunics"]:
        tags.add("upper_body")

    if article in ["Kurtas", "Kurtis", "Nehru Jackets"]:
        tags.add("ethnic")

    if article == "Kurta Sets":
        tags.update(["ethnic", "traditional"])

    if article in ["Jeans", "Jeggings"]:
        tags.add("denim")

    if article in [
        "Track Pants",
        "Lounge Pants",
        "Lounge Shorts"
    ]:
        tags.add("relaxed")

    if article in [
        "Sweaters",
        "Sweatshirts",
        "Jackets",
        "Blazers"
    ]:
        tags.add("layering")

    if article in [
        "Sports Shoes",
        "Sports Sandals"
    ]:
        tags.add("athletic")

    if article in [
        "Heels",
        "Flats"
    ]:
        tags.add("dressy")

    return ",".join(sorted(tags))


# --------------------------------------------------
# Occasion tags
#
# IMPORTANT:
# Occasion comes primarily from usage.
# Article type does NOT override usage.
# --------------------------------------------------

def get_occasion_tags(usage):

    if pd.isna(usage):
        return ""

    if usage == "Casual":
        return "casual_outing,college"

    elif usage == "Sports":
        return "sports"

    elif usage == "Ethnic":
        return "ethnic_event,festival"

    elif usage == "Formal":
        return "office,interview"

    elif usage == "Smart Casual":
        return "casual_outing,office"

    elif usage == "Party":
        return "party"

    elif usage == "Travel":
        return "travel"

    return ""


# --------------------------------------------------
# Weather tags
# --------------------------------------------------

def get_weather_tags(season):

    if pd.isna(season):
        return ""

    if season == "Summer":
        return "warm"

    elif season == "Winter":
        return "cold"

    elif season == "Fall":
        return "cool"

    elif season == "Spring":
        return "mild"

    return ""


# --------------------------------------------------
# Formality
# --------------------------------------------------

def get_formality(usage):

    if pd.isna(usage):
        return "unknown"

    if usage == "Formal":
        return "formal"

    elif usage == "Smart Casual":
        return "smart_casual"

    elif usage == "Party":
        return "dressy"

    elif usage == "Ethnic":
        return "semi_formal"

    elif usage == "Sports":
        return "active"

    elif usage == "Casual":
        return "casual"

    elif usage == "Travel":
        return "casual"

    return "unknown"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading SnapStyle dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")


# --------------------------------------------------
# Generate tags
# --------------------------------------------------

print("Generating tags...")

df["style_tags"] = df.apply(
    get_style_tags,
    axis=1
)

df["occasion_tags"] = df["usage"].apply(
    get_occasion_tags
)

df["weather_tags"] = df["season"].apply(
    get_weather_tags
)

df["formality"] = df["usage"].apply(
    get_formality
)


# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print()
print("Tagged dataset created successfully!")

print(f"Output: {OUTPUT_FILE}")
print(f"Rows: {len(df)}")

print()
print("Columns:")
print(df.columns.tolist())

print()
print("Formality distribution:")
print(df["formality"].value_counts(dropna=False).to_string())

print()
print("Missing usage values:")
print(df["usage"].isna().sum())

print()
print("Missing season values:")
print(df["season"].isna().sum())

print()
print("Missing colour values:")
print(df["baseColour"].isna().sum())
