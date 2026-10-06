import pandas as pd


# --------------------------------------------------
# Load tagged dataset
# --------------------------------------------------

DATASET_PATH = "dataset/kaggle_fashion/snapstyle_items_tagged.csv"


def load_dataset():
    """
    Load the processed SnapStyle clothing dataset.
    """
    return pd.read_csv(DATASET_PATH)


# --------------------------------------------------
# Helper: check whether a value exists in a
# comma-separated tag column
# --------------------------------------------------

def has_tag(value, tag):

    if pd.isna(value):
        return False

    tags = str(value).split(",")

    return tag in tags


# --------------------------------------------------
# Main filtering function
# --------------------------------------------------

def filter_items(
    df,
    gender=None,
    occasion=None,
    season=None,
    category=None
):
    """
    Filter clothing items using hard constraints.

    Gender, occasion and category are hard filters.

    Season is intentionally NOT a hard filter.
    It will be considered later during scoring.
    """

    result = df.copy()

    # --------------------------------------------------
    # Gender
    # --------------------------------------------------

    if gender is not None:

        result = result[
            (result["gender"] == gender) |
            (result["gender"] == "Unisex")
        ]

    # --------------------------------------------------
    # Occasion
    # --------------------------------------------------

    if occasion is not None:

        result = result[
            result["occasion_tags"].apply(
                lambda x: has_tag(x, occasion)
            )
        ]

    # --------------------------------------------------
    # Season
    #
    # Season is NOT a hard filter.
    # It will be considered during scoring.
    # --------------------------------------------------

    # We intentionally do not filter by season here.

    # --------------------------------------------------
    # Category
    # --------------------------------------------------

    if category is not None:

        result = result[
            result["our_category"] == category
        ]

    return result.reset_index(drop=True)


# --------------------------------------------------
# Get candidates for all outfit categories
# --------------------------------------------------

def get_outfit_candidates(
    df,
    gender=None,
    occasion=None,
    season=None
):
    """
    Get candidate clothing items separated by
    outfit category.
    """

    candidates = {}

    categories = [
        "topwear",
        "bottomwear",
        "one_piece",
        "footwear"
    ]

    for category in categories:

        candidates[category] = filter_items(
            df,
            gender=gender,
            occasion=occasion,
            season=season,
            category=category
        )

    return candidates


# --------------------------------------------------
# Test the filtering module
# --------------------------------------------------

if __name__ == "__main__":

    print("Loading SnapStyle dataset...")

    df = load_dataset()

    print(f"Total items: {len(df)}")

    print()
    print("Testing filter:")
    print("Gender: Men")
    print("Occasion: interview")
    print("Season preference: Winter")

    filtered = filter_items(
        df,
        gender="Men",
        occasion="interview",
        season="Winter"
    )

    print()
    print(f"Filtered items: {len(filtered)}")

    print()
    print("Category distribution:")

    print(
        filtered["our_category"]
        .value_counts()
        .to_string()
    )

    print()
    print("Sample candidates:")

    print(
        filtered[
            [
                "id",
                "gender",
                "articleType",
                "our_category",
                "baseColour",
                "season",
                "usage",
                "formality"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )
