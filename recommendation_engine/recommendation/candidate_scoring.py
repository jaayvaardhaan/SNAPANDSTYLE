import pandas as pd


ITEM_WEIGHTS = {
    "occasion": 35,
    "season": 25,
    "formality": 20,
    "style": 10,
    "article": 10
}


# --------------------------------------------------
# Occasion score
# --------------------------------------------------

def get_occasion_score(item, requested_occasion):

    if requested_occasion is None:
        return 1.0

    if pd.isna(item["occasion_tags"]):
        return 0.0

    tags = str(item["occasion_tags"]).split(",")

    if requested_occasion in tags:
        return 1.0

    return 0.0


# --------------------------------------------------
# Season score
# --------------------------------------------------

def get_season_score(item, requested_season):

    if requested_season is None:
        return 1.0

    if pd.isna(item["season"]):
        return 0.5

    if item["season"] == requested_season:
        return 1.0

    return 0.4


# --------------------------------------------------
# Formality score
# --------------------------------------------------

def get_formality_score(item, requested_occasion):

    if pd.isna(item["formality"]):
        return 0.5

    formality = item["formality"]

    formal_occasions = [
        "interview",
        "office"
    ]

    casual_occasions = [
        "college",
        "casual_outing"
    ]

    if requested_occasion in formal_occasions:

        if formality == "formal":
            return 1.0

        if formality == "smart_casual":
            return 0.8

        if formality == "semi_formal":
            return 0.6

        return 0.2

    if requested_occasion in casual_occasions:

        if formality == "casual":
            return 1.0

        if formality == "smart_casual":
            return 0.9

        if formality == "semi_formal":
            return 0.5

        return 0.2

    return 0.5


# --------------------------------------------------
# Style score
# --------------------------------------------------

def get_style_score(item):

    if pd.isna(item["style_tags"]):
        return 0.5

    tags = str(item["style_tags"]).split(",")

    useful_tags = [
        "casual",
        "everyday",
        "formal",
        "professional",
        "sports",
        "active",
        "ethnic",
        "traditional",
        "smart_casual",
        "party",
        "travel"
    ]

    matches = sum(
        1
        for tag in tags
        if tag in useful_tags
    )

    if matches >= 2:
        return 1.0

    if matches == 1:
        return 0.8

    return 0.5


# --------------------------------------------------
# Article compatibility
# --------------------------------------------------

ARTICLE_RULES = {

    "interview": {

        "topwear": {
            "Shirts": 1.0,
            "Blazers": 1.0,
            "Waistcoat": 0.9,
            "Nehru Jackets": 0.8,
            "Kurtas": 0.4,
            "Tshirts": 0.2,
            "Lounge Tshirts": 0.1,
            "Sweatshirts": 0.1,
            "Sweaters": 0.4,
            "Jackets": 0.6,
            "Tops": 0.3,
            "Tunics": 0.3
        },

        "bottomwear": {
            "Trousers": 1.0,
            "Jeans": 0.3,
            "Shorts": 0.1,
            "Track Pants": 0.1,
            "Lounge Pants": 0.1,
            "Capris": 0.1,
            "Leggings": 0.1,
            "Jeggings": 0.1,
            "Skirts": 0.5
        },

        "footwear": {
            "Formal Shoes": 1.0,
            "Casual Shoes": 0.6,
            "Flats": 0.5,
            "Heels": 0.5,
            "Booties": 0.6,
            "Sandals": 0.2,
            "Flip Flops": 0.1,
            "Sports Shoes": 0.2,
            "Sports Sandals": 0.1
        }
    },

    "office": {

        "topwear": {
            "Shirts": 1.0,
            "Blazers": 1.0,
            "Waistcoat": 0.9,
            "Nehru Jackets": 0.8,
            "Tshirts": 0.4,
            "Sweaters": 0.6,
            "Jackets": 0.6
        },

        "bottomwear": {
            "Trousers": 1.0,
            "Jeans": 0.4,
            "Shorts": 0.1,
            "Track Pants": 0.1,
            "Lounge Pants": 0.1
        },

        "footwear": {
            "Formal Shoes": 1.0,
            "Casual Shoes": 0.7,
            "Heels": 0.7,
            "Flats": 0.7,
            "Booties": 0.7,
            "Sandals": 0.3,
            "Flip Flops": 0.1,
            "Sports Shoes": 0.2,
            "Sports Sandals": 0.1
        }
    },

    "college": {

        "topwear": {
            "Tshirts": 1.0,
            "Lounge Tshirts": 0.9,
            "Shirts": 0.8,
            "Tops": 0.9,
            "Sweatshirts": 0.9,
            "Sweaters": 0.8,
            "Jackets": 0.7,
            "Kurtas": 0.6,
            "Blazers": 0.3
        },

        "bottomwear": {
            "Jeans": 1.0,
            "Trousers": 0.7,
            "Shorts": 0.8,
            "Track Pants": 0.8,
            "Lounge Pants": 0.7,
            "Capris": 0.8,
            "Leggings": 0.8,
            "Jeggings": 0.8,
            "Skirts": 0.8
        },

        "footwear": {
            "Casual Shoes": 1.0,
            "Sports Shoes": 0.9,
            "Sports Sandals": 0.8,
            "Sandals": 0.8,
            "Flats": 0.8,
            "Booties": 0.7,
            "Formal Shoes": 0.5,
            "Heels": 0.6,
            "Flip Flops": 0.6
        }
    },

    "casual_outing": {

        "topwear": {
            "Tshirts": 1.0,
            "Lounge Tshirts": 0.9,
            "Shirts": 0.9,
            "Tops": 1.0,
            "Sweatshirts": 0.9,
            "Sweaters": 0.8,
            "Jackets": 0.8,
            "Kurtas": 0.7,
            "Blazers": 0.4
        },

        "bottomwear": {
            "Jeans": 1.0,
            "Shorts": 0.8,
            "Trousers": 0.8,
            "Track Pants": 0.8,
            "Lounge Pants": 0.8,
            "Capris": 0.8,
            "Leggings": 0.8,
            "Jeggings": 0.8,
            "Skirts": 0.9
        },

        "footwear": {
            "Casual Shoes": 1.0,
            "Sports Shoes": 0.9,
            "Sandals": 0.9,
            "Sports Sandals": 0.8,
            "Flats": 0.9,
            "Booties": 0.8,
            "Heels": 0.7,
            "Formal Shoes": 0.5,
            "Flip Flops": 0.7
        }
    },

    "sports": {

        "topwear": {
            "Tshirts": 0.9,
            "Lounge Tshirts": 0.9,
            "Sweatshirts": 0.8,
            "Jackets": 0.7,
            "Tops": 0.8
        },

        "bottomwear": {
            "Track Pants": 1.0,
            "Shorts": 1.0,
            "Lounge Pants": 0.9,
            "Capris": 0.9,
            "Tights": 0.9,
            "Leggings": 0.9
        },

        "footwear": {
            "Sports Shoes": 1.0,
            "Sports Sandals": 0.9,
            "Casual Shoes": 0.6,
            "Sandals": 0.5,
            "Formal Shoes": 0.1,
            "Heels": 0.1,
            "Flats": 0.3,
            "Flip Flops": 0.3
        }
    },

    "ethnic_event": {

        "topwear": {
            "Kurtas": 1.0,
            "Kurtis": 1.0,
            "Nehru Jackets": 1.0,
            "Tunics": 0.8,
            "Tops": 0.5,
            "Shirts": 0.4,
            "Tshirts": 0.2
        },

        "bottomwear": {
            "Churidar": 1.0,
            "Salwar": 1.0,
            "Patiala": 1.0,
            "Trousers": 0.5,
            "Jeans": 0.2
        },

        "footwear": {
            "Flats": 0.9,
            "Sandals": 0.9,
            "Heels": 0.8,
            "Casual Shoes": 0.5,
            "Formal Shoes": 0.4,
            "Sports Shoes": 0.2,
            "Flip Flops": 0.3
        }
    },

    "festival": {

        "topwear": {
            "Kurtas": 1.0,
            "Kurtis": 1.0,
            "Nehru Jackets": 1.0,
            "Tops": 0.7,
            "Tunics": 0.8,
            "Shirts": 0.5,
            "Tshirts": 0.3
        },

        "bottomwear": {
            "Churidar": 1.0,
            "Salwar": 1.0,
            "Patiala": 1.0,
            "Skirts": 0.9,
            "Trousers": 0.5,
            "Jeans": 0.3
        },

        "footwear": {
            "Flats": 1.0,
            "Sandals": 0.9,
            "Heels": 0.9,
            "Casual Shoes": 0.5,
            "Formal Shoes": 0.4,
            "Sports Shoes": 0.2,
            "Flip Flops": 0.4
        }
    },

    "party": {

        "topwear": {
            "Blazers": 1.0,
            "Tops": 1.0,
            "Shirts": 0.9,
            "Jackets": 0.9,
            "Kurtis": 0.8,
            "Kurtas": 0.7,
            "Tshirts": 0.5
        },

        "bottomwear": {
            "Trousers": 0.8,
            "Jeans": 0.7,
            "Skirts": 1.0,
            "Leggings": 0.8,
            "Shorts": 0.6
        },

        "footwear": {
            "Heels": 1.0,
            "Booties": 0.9,
            "Formal Shoes": 0.8,
            "Flats": 0.8,
            "Casual Shoes": 0.7,
            "Sandals": 0.7,
            "Sports Shoes": 0.3,
            "Flip Flops": 0.3
        }
    }
}


def get_article_score(
    item,
    requested_occasion
):

    if requested_occasion is None:
        return 0.5

    category = item["our_category"]
    article_type = item["articleType"]

    rules = ARTICLE_RULES.get(
        requested_occasion
    )

    if rules is None:
        return 0.5

    category_rules = rules.get(
        category,
        {}
    )

    return category_rules.get(
        article_type,
        0.5
    )


# --------------------------------------------------
# Calculate complete item score
# --------------------------------------------------

def calculate_item_score(
    item,
    requested_occasion=None,
    requested_season=None
):

    occasion = get_occasion_score(
        item,
        requested_occasion
    )

    season = get_season_score(
        item,
        requested_season
    )

    formality = get_formality_score(
        item,
        requested_occasion
    )

    style = get_style_score(item)

    article = get_article_score(
        item,
        requested_occasion
    )

    occasion_points = (
        occasion * ITEM_WEIGHTS["occasion"]
    )

    season_points = (
        season * ITEM_WEIGHTS["season"]
    )

    formality_points = (
        formality * ITEM_WEIGHTS["formality"]
    )

    style_points = (
        style * ITEM_WEIGHTS["style"]
    )

    article_points = (
        article * ITEM_WEIGHTS["article"]
    )

    total = (
        occasion_points
        + season_points
        + formality_points
        + style_points
        + article_points
    )

    return round(total, 2)


# --------------------------------------------------
# Select top candidates
# --------------------------------------------------

def select_top_candidates(
    df,
    requested_occasion=None,
    requested_season=None,
    limit=10
):

    if df.empty:
        return df.copy()

    result = df.copy()

    result["item_score"] = result.apply(
        lambda row: calculate_item_score(
            row,
            requested_occasion,
            requested_season
        ),
        axis=1
    )

    result = result.sort_values(
        by="item_score",
        ascending=False
    )

    return result.head(
        limit
    ).reset_index(
        drop=True
    )


# --------------------------------------------------
# Select all categories
# --------------------------------------------------

def select_all_candidates(
    candidates,
    requested_occasion=None,
    requested_season=None,
    limit=10
):

    selected = {}

    for category, items in candidates.items():

        selected[category] = select_top_candidates(
            items,
            requested_occasion,
            requested_season,
            limit
        )

    return selected


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    from filtering import (
        load_dataset,
        filter_items
    )

    print(
        "Loading SnapStyle dataset..."
    )

    df = load_dataset()

    gender = "Men"
    occasion = "interview"
    season = "Winter"

    candidates = {

        "topwear": filter_items(
            df,
            gender=gender,
            occasion=occasion,
            category="topwear"
        ),

        "bottomwear": filter_items(
            df,
            gender=gender,
            occasion=occasion,
            category="bottomwear"
        ),

        "one_piece": filter_items(
            df,
            gender=gender,
            occasion=occasion,
            category="one_piece"
        ),

        "footwear": filter_items(
            df,
            gender=gender,
            occasion=occasion,
            category="footwear"
        )
    }

    print()
    print(
        "Original candidate counts:"
    )

    for category, items in candidates.items():

        print(
            f"{category}: {len(items)}"
        )

    selected = select_all_candidates(
        candidates,
        requested_occasion=occasion,
        requested_season=season,
        limit=10
    )

    print()
    print(
        "Selected candidate counts:"
    )

    for category, items in selected.items():

        print(
            f"{category}: {len(items)}"
        )

    print()
    print(
        "TOP SELECTED CANDIDATES"
    )

    for category, items in selected.items():

        print()
        print(
            f"--- {category.upper()} ---"
        )

        if items.empty:

            print(
                "No candidates"
            )

            continue

        print(
            items[
                [
                    "id",
                    "articleType",
                    "baseColour",
                    "season",
                    "usage",
                    "formality",
                    "item_score"
                ]
            ].to_string(
                index=False
            )
        )