import pandas as pd

from filtering import load_dataset, filter_items
from candidate_scoring import select_all_candidates
from combination import generate_outfits
from cv_features import get_cv_features, validate_cv_features


USER_PREFERENCES = {
    "preferred_colour": "Blue",
    "preferred_style": "formal",
    "preferred_formality": "formal"
}


COLOUR_GROUPS = {

    "neutral": {
        "Black", "White", "Grey", "Off White", "Cream",
        "Beige", "Brown", "Navy Blue", "Khaki", "Tan", "Charcoal"
    },

    "blue": {
        "Blue", "Navy Blue", "Turquoise Blue", "Teal"
    },

    "red": {
        "Red", "Maroon", "Burgundy", "Rust",
        "Pink", "Peach", "Magenta"
    },

    "green": {
        "Green", "Olive", "Sea Green", "Lime Green"
    },

    "yellow": {
        "Yellow", "Mustard", "Gold"
    },

    "purple": {
        "Purple", "Lavender", "Mauve"
    },

    "orange": {
        "Orange"
    }
}


COMPATIBLE_COLOURS = {
    ("blue", "neutral"),
    ("neutral", "blue"),
    ("blue", "red"),
    ("red", "blue"),
    ("blue", "yellow"),
    ("yellow", "blue"),
    ("blue", "orange"),
    ("orange", "blue"),
    ("green", "neutral"),
    ("neutral", "green"),
    ("green", "yellow"),
    ("yellow", "green"),
    ("purple", "neutral"),
    ("neutral", "purple"),
    ("purple", "yellow"),
    ("yellow", "purple")
}


def get_colour_group(colour):

    if pd.isna(colour):
        return None

    for group, colours in COLOUR_GROUPS.items():

        if colour in colours:
            return group

    return None


def colour_compatibility(colour1, colour2):

    if pd.isna(colour1) or pd.isna(colour2):
        return 0.45

    if colour1 == colour2:
        return 0.68

    group1 = get_colour_group(colour1)
    group2 = get_colour_group(colour2)

    if group1 == "neutral" and group2 == "neutral":
        return 0.95

    if group1 == "neutral" or group2 == "neutral":
        return 0.92

    if group1 == group2:
        return 0.78

    if (group1, group2) in COMPATIBLE_COLOURS:
        return 0.72

    return 0.52


def season_score(item_season, requested_season):

    if pd.isna(item_season):
        return 0.45

    if item_season == requested_season:
        return 1.0

    nearby = {
        "Winter": {
            "Fall": 0.65,
            "Spring": 0.50,
            "Summer": 0.35
        },

        "Summer": {
            "Spring": 0.65,
            "Fall": 0.50,
            "Winter": 0.35
        },

        "Spring": {
            "Summer": 0.65,
            "Winter": 0.50,
            "Fall": 0.65
        },

        "Fall": {
            "Winter": 0.65,
            "Summer": 0.50,
            "Spring": 0.65
        }
    }

    return nearby.get(
        requested_season, {}
    ).get(
        item_season, 0.40
    )


def occasion_score(item, occasion):

    if pd.isna(item["occasion_tags"]):
        return 0.25

    tags = str(
        item["occasion_tags"]
    ).split(",")

    if occasion in tags:
        return 1.0

    return 0.20


def formality_score(item_formality, occasion):

    if pd.isna(item_formality):
        return 0.40

    formality = str(
        item_formality
    ).lower()

    if occasion in ["interview", "office"]:

        values = {
            "formal": 1.0,
            "smart_casual": 0.80,
            "semi_formal": 0.60,
            "casual": 0.30,
            "active": 0.20,
            "dressy": 0.60
        }

        return values.get(
            formality, 0.40
        )

    if occasion in [
        "college",
        "casual_outing"
    ]:

        values = {
            "casual": 1.0,
            "smart_casual": 0.95,
            "semi_formal": 0.70,
            "formal": 0.55,
            "active": 0.80,
            "dressy": 0.65
        }

        return values.get(
            formality, 0.40
        )

    if occasion == "sports":

        values = {
            "active": 1.0,
            "casual": 0.80,
            "smart_casual": 0.50,
            "semi_formal": 0.20,
            "formal": 0.10
        }

        return values.get(
            formality, 0.30
        )

    return 0.50


def style_score(item1, item2):

    tags1 = set()
    tags2 = set()

    if not pd.isna(item1["style_tags"]):

        tags1 = set(
            str(
                item1["style_tags"]
            ).split(",")
        )

    if not pd.isna(item2["style_tags"]):

        tags2 = set(
            str(
                item2["style_tags"]
            ).split(",")
        )

    common = len(
        tags1.intersection(tags2)
    )

    if common >= 3:
        return 1.0

    if common == 2:
        return 0.92

    if common == 1:
        return 0.78

    return 0.50


def article_compatibility(
    top,
    bottom,
    footwear,
    occasion
):

    score = 0.0
    count = 0

    if occasion == "interview":

        top_rules = {
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
        }

        bottom_rules = {
            "Trousers": 1.0,
            "Jeans": 0.35,
            "Shorts": 0.1,
            "Track Pants": 0.1,
            "Lounge Pants": 0.1,
            "Capris": 0.1,
            "Leggings": 0.1,
            "Jeggings": 0.1,
            "Skirts": 0.5
        }

        footwear_rules = {
            "Formal Shoes": 1.0,
            "Casual Shoes": 0.65,
            "Booties": 0.6,
            "Flats": 0.5,
            "Heels": 0.5,
            "Sports Shoes": 0.2,
            "Sports Sandals": 0.2,
            "Sandals": 0.2,
            "Flip Flops": 0.1
        }

        score += top_rules.get(
            top["articleType"], 0.40
        )
        count += 1

        score += bottom_rules.get(
            bottom["articleType"], 0.40
        )
        count += 1

        score += footwear_rules.get(
            footwear["articleType"], 0.40
        )
        count += 1

    else:

        score = 0.70
        count = 1

    return score / count


def preference_score(
    item,
    preferences
):

    colour = item["baseColour"]

    preferred_colour = preferences.get(
        "preferred_colour"
    )

    if preferred_colour is None:

        colour_pref = 0.50

    elif colour == preferred_colour:

        colour_pref = 1.0

    elif (
        get_colour_group(colour)
        == get_colour_group(preferred_colour)
    ):

        colour_pref = 0.75

    elif get_colour_group(colour) == "neutral":

        colour_pref = 0.60

    else:

        colour_pref = 0.30

    preferred_style = preferences.get(
        "preferred_style"
    )

    if preferred_style:

        tags = []

        if not pd.isna(
            item["style_tags"]
        ):

            tags = str(
                item["style_tags"]
            ).split(",")

        style_pref = (
            1.0
            if preferred_style in tags
            else 0.40
        )

    else:

        style_pref = 0.50

    preferred_formality = preferences.get(
        "preferred_formality"
    )

    if preferred_formality:

        formality_pref = (
            1.0
            if str(
                item["formality"]
            ).lower()
            == preferred_formality.lower()
            else 0.40
        )

    else:

        formality_pref = 0.50

    return (
        colour_pref * 0.40
        + style_pref * 0.30
        + formality_pref * 0.30
    )


# ============================================================
# CV-BASED SCORING
# ============================================================

SKIN_TONE_COLOUR_RULES = {

    "deep": {
        "neutral": 0.90,
        "blue": 0.95,
        "red": 0.90,
        "green": 0.95,
        "yellow": 0.90,
        "purple": 0.95,
        "orange": 0.85
    },

    "medium-deep": {
        "neutral": 0.90,
        "blue": 0.95,
        "red": 0.92,
        "green": 0.92,
        "yellow": 0.90,
        "purple": 0.92,
        "orange": 0.85
    },

    "medium": {
        "neutral": 0.92,
        "blue": 0.95,
        "red": 0.92,
        "green": 0.90,
        "yellow": 0.88,
        "purple": 0.92,
        "orange": 0.85
    },

    "medium-light": {
        "neutral": 0.95,
        "blue": 0.92,
        "red": 0.90,
        "green": 0.88,
        "yellow": 0.82,
        "purple": 0.92,
        "orange": 0.80
    },

    "light": {
        "neutral": 0.95,
        "blue": 0.90,
        "red": 0.88,
        "green": 0.85,
        "yellow": 0.78,
        "purple": 0.90,
        "orange": 0.78
    }
}


UNDERTONE_COLOUR_RULES = {

    "warm": {
        "blue": 0.85,
        "red": 0.95,
        "green": 0.92,
        "yellow": 1.0,
        "purple": 0.80,
        "orange": 1.0,
        "neutral": 0.92
    },

    "cool": {
        "blue": 1.0,
        "red": 0.88,
        "green": 0.85,
        "yellow": 0.75,
        "purple": 0.95,
        "orange": 0.70,
        "neutral": 0.95
    },

    "neutral": {
        "blue": 0.95,
        "red": 0.92,
        "green": 0.92,
        "yellow": 0.90,
        "purple": 0.92,
        "orange": 0.88,
        "neutral": 0.98
    }
}


HAIR_COLOUR_RULES = {

    "black": {
        "blue": 0.95,
        "red": 0.90,
        "green": 0.90,
        "yellow": 0.88,
        "purple": 0.92,
        "orange": 0.88,
        "neutral": 0.95
    },

    "dark_brown": {
        "blue": 0.92,
        "red": 0.92,
        "green": 0.92,
        "yellow": 0.90,
        "purple": 0.90,
        "orange": 0.92,
        "neutral": 0.95
    },

    "brown": {
        "blue": 0.90,
        "red": 0.92,
        "green": 0.92,
        "yellow": 0.90,
        "purple": 0.90,
        "orange": 0.92,
        "neutral": 0.95
    },

    "light_brown": {
        "blue": 0.92,
        "red": 0.90,
        "green": 0.88,
        "yellow": 0.88,
        "purple": 0.92,
        "orange": 0.88,
        "neutral": 0.95
    },

    "blonde": {
        "blue": 0.95,
        "red": 0.88,
        "green": 0.85,
        "yellow": 0.75,
        "purple": 0.95,
        "orange": 0.75,
        "neutral": 0.95
    },

    "red_or_auburn": {
        "blue": 0.95,
        "red": 0.80,
        "green": 0.95,
        "yellow": 0.88,
        "purple": 0.92,
        "orange": 0.75,
        "neutral": 0.95
    },

    "gray_or_white": {
        "blue": 0.95,
        "red": 0.90,
        "green": 0.88,
        "yellow": 0.82,
        "purple": 0.95,
        "orange": 0.80,
        "neutral": 1.0
    }
}


FACE_SHAPE_STYLE_RULES = {

    "oval": {
        "formal": 1.0,
        "casual": 1.0,
        "ethnic": 1.0,
        "sport": 1.0
    },

    "round": {
        "formal": 0.95,
        "casual": 1.0,
        "ethnic": 0.95,
        "sport": 1.0
    },

    "square": {
        "formal": 1.0,
        "casual": 1.0,
        "ethnic": 0.95,
        "sport": 1.0
    },

    "oblong": {
        "formal": 0.95,
        "casual": 1.0,
        "ethnic": 0.95,
        "sport": 1.0
    },

    "heart": {
        "formal": 0.95,
        "casual": 1.0,
        "ethnic": 0.95,
        "sport": 1.0
    },

    "diamond": {
        "formal": 1.0,
        "casual": 1.0,
        "ethnic": 0.95,
        "sport": 1.0
    }
}


def cv_colour_score(
    item,
    cv_features
):

    colour = item["baseColour"]

    if pd.isna(colour):
        return 0.50

    group = get_colour_group(
        colour
    )

    if group is None:
        return 0.50

    scores = []

    skin_tone = cv_features.get(
        "skin_tone"
    )

    if skin_tone in SKIN_TONE_COLOUR_RULES:

        score = SKIN_TONE_COLOUR_RULES[
            skin_tone
        ].get(
            group,
            0.50
        )

        confidence = cv_features[
            "confidence"
        ].get(
            "skin_tone",
            0.0
        )

        scores.append(
            score * confidence
            + 0.50 * (1 - confidence)
        )

    undertone = cv_features.get(
        "skin_undertone"
    )

    if undertone in UNDERTONE_COLOUR_RULES:

        score = UNDERTONE_COLOUR_RULES[
            undertone
        ].get(
            group,
            0.50
        )

        confidence = cv_features[
            "confidence"
        ].get(
            "skin_undertone",
            0.0
        )

        scores.append(
            score * confidence
            + 0.50 * (1 - confidence)
        )

    hair = cv_features.get(
        "hair_color"
    )

    if hair in HAIR_COLOUR_RULES:

        score = HAIR_COLOUR_RULES[
            hair
        ].get(
            group,
            0.50
        )

        confidence = cv_features[
            "confidence"
        ].get(
            "hair_color",
            0.0
        )

        scores.append(
            score * confidence
            + 0.50 * (1 - confidence)
        )

    if not scores:
        return 0.50

    return sum(scores) / len(scores)


def cv_style_score(
    item,
    cv_features
):

    face_shape = cv_features.get(
        "face_shape"
    )

    if face_shape is None:
        return 0.50

    confidence = cv_features[
        "confidence"
    ].get(
        "face_shape",
        0.0
    )

    tags = []

    if not pd.isna(
        item["style_tags"]
    ):

        tags = [
            x.strip().lower()
            for x in str(
                item["style_tags"]
            ).split(",")
        ]

    if "formal" in tags:
        style = "formal"

    elif "ethnic" in tags:
        style = "ethnic"

    elif (
        "sports" in tags
        or "active" in tags
    ):

        style = "sport"

    else:
        style = "casual"

    face_rules = FACE_SHAPE_STYLE_RULES.get(
        face_shape,
        {}
    )

    score = face_rules.get(
        style,
        0.80
    )

    return (
        score * confidence
        + 0.50 * (1 - confidence)
    )


def cv_item_score(
    item,
    cv_features
):

    colour_score = cv_colour_score(
        item,
        cv_features
    )

    style_score_value = cv_style_score(
        item,
        cv_features
    )

    return (
        colour_score * 0.75
        + style_score_value * 0.25
    )


def outfit_cv_score(
    outfit,
    cv_features
):

    scores = []

    for key in [
        "topwear",
        "bottomwear",
        "one_piece",
        "footwear"
    ]:

        if key in outfit:

            scores.append(
                cv_item_score(
                    outfit[key],
                    cv_features
                )
            )

    if not scores:
        return 0.50

    return sum(scores) / len(scores)


# ============================================================
# OUTFIT SCORING
# ============================================================

def score_normal_outfit(
    outfit,
    occasion,
    season,
    preferences,
    cv_features
):

    top = outfit["topwear"]
    bottom = outfit["bottomwear"]
    footwear = outfit["footwear"]

    occasion_value = sum([
        occasion_score(
            top,
            occasion
        ),
        occasion_score(
            bottom,
            occasion
        ),
        occasion_score(
            footwear,
            occasion
        )
    ]) / 3

    season_value = sum([
        season_score(
            top["season"],
            season
        ),
        season_score(
            bottom["season"],
            season
        ),
        season_score(
            footwear["season"],
            season
        )
    ]) / 3

    colour_value = sum([
        colour_compatibility(
            top["baseColour"],
            bottom["baseColour"]
        ),
        colour_compatibility(
            top["baseColour"],
            footwear["baseColour"]
        ),
        colour_compatibility(
            bottom["baseColour"],
            footwear["baseColour"]
        )
    ]) / 3

    style_value = sum([
        style_score(
            top,
            bottom
        ),
        style_score(
            top,
            footwear
        ),
        style_score(
            bottom,
            footwear
        )
    ]) / 3

    formality_value = sum([
        formality_score(
            top["formality"],
            occasion
        ),
        formality_score(
            bottom["formality"],
            occasion
        ),
        formality_score(
            footwear["formality"],
            occasion
        )
    ]) / 3

    article_value = article_compatibility(
        top,
        bottom,
        footwear,
        occasion
    )

    preference_value = sum([
        preference_score(
            top,
            preferences
        ),
        preference_score(
            bottom,
            preferences
        ),
        preference_score(
            footwear,
            preferences
        )
    ]) / 3

    cv_value = outfit_cv_score(
        outfit,
        cv_features
    )

    score = (
        occasion_value * 23
        + season_value * 14
        + colour_value * 14
        + style_value * 10
        + formality_value * 12
        + article_value * 9
        + preference_value * 8
        + cv_value * 10
    )

    return {
        "score": round(score, 2),

        "occasion": round(
            occasion_value * 23,
            2
        ),

        "season": round(
            season_value * 14,
            2
        ),

        "colour": round(
            colour_value * 14,
            2
        ),

        "style": round(
            style_value * 10,
            2
        ),

        "formality": round(
            formality_value * 12,
            2
        ),

        "article": round(
            article_value * 9,
            2
        ),

        "preference": round(
            preference_value * 8,
            2
        ),

        "cv": round(
            cv_value * 10,
            2
        )
    }


def score_one_piece_outfit(
    outfit,
    occasion,
    season,
    preferences,
    cv_features
):

    dress = outfit["one_piece"]
    footwear = outfit["footwear"]

    occasion_value = (
        occasion_score(
            dress,
            occasion
        )
        + occasion_score(
            footwear,
            occasion
        )
    ) / 2

    season_value = (
        season_score(
            dress["season"],
            season
        )
        + season_score(
            footwear["season"],
            season
        )
    ) / 2

    colour_value = colour_compatibility(
        dress["baseColour"],
        footwear["baseColour"]
    )

    style_value = style_score(
        dress,
        footwear
    )

    formality_value = (
        formality_score(
            dress["formality"],
            occasion
        )
        + formality_score(
            footwear["formality"],
            occasion
        )
    ) / 2

    article_value = 0.80

    preference_value = (
        preference_score(
            dress,
            preferences
        )
        + preference_score(
            footwear,
            preferences
        )
    ) / 2

    cv_value = outfit_cv_score(
        outfit,
        cv_features
    )

    score = (
        occasion_value * 23
        + season_value * 14
        + colour_value * 14
        + style_value * 10
        + formality_value * 12
        + article_value * 9
        + preference_value * 8
        + cv_value * 10
    )

    return {
        "score": round(score, 2),

        "occasion": round(
            occasion_value * 23,
            2
        ),

        "season": round(
            season_value * 14,
            2
        ),

        "colour": round(
            colour_value * 14,
            2
        ),

        "style": round(
            style_value * 10,
            2
        ),

        "formality": round(
            formality_value * 12,
            2
        ),

        "article": round(
            article_value * 9,
            2
        ),

        "preference": round(
            preference_value * 8,
            2
        ),

        "cv": round(
            cv_value * 10,
            2
        )
    }


def score_outfit(
    outfit,
    occasion,
    season,
    preferences,
    cv_features
):

    if "one_piece" in outfit:

        return score_one_piece_outfit(
            outfit,
            occasion,
            season,
            preferences,
            cv_features
        )

    return score_normal_outfit(
        outfit,
        occasion,
        season,
        preferences,
        cv_features
    )


# ============================================================
# TOP 10 DIVERSITY
# ============================================================

def select_diverse_top_outfits(
    scored_outfits,
    limit=10
):

    selected = []
    used_signatures = set()

    for outfit in scored_outfits:

        if "one_piece" in outfit:

            signature = (
                "one_piece",
                str(
                    outfit["one_piece"]["id"]
                ),
                str(
                    outfit["footwear"]["id"]
                )
            )

        else:

            signature = (
                "normal",
                str(
                    outfit["topwear"]["id"]
                ),
                str(
                    outfit["bottomwear"]["id"]
                )
            )

        if signature in used_signatures:
            continue

        selected.append(outfit)

        used_signatures.add(
            signature
        )

        if len(selected) >= limit:
            break

    return selected


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "Loading SnapStyle dataset...\n"
    )

    df = load_dataset()

    gender = "Men"
    occasion = "interview"
    season = "Winter"

    print("User requirements:")
    print("Gender:", gender)
    print("Occasion:", occasion)
    print("Season:", season)

    print("\nUser preferences:")

    for key, value in USER_PREFERENCES.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # CV FEATURES
    # --------------------------------------------------------

    cv_features = get_cv_features()

    if not validate_cv_features(
        cv_features
    ):

        print(
            "\nInvalid CV features."
        )

        return

    print("\nCV FEATURES:")
    print("-" * 40)

    print(
        "Face shape:",
        cv_features["face_shape"]
    )

    print(
        "Skin tone:",
        cv_features["skin_tone"]
    )

    print(
        "Skin undertone:",
        cv_features["skin_undertone"]
    )

    print(
        "Hair color:",
        cv_features["hair_color"]
    )

    print("\nCV CONFIDENCE:")

    for key, value in cv_features[
        "confidence"
    ].items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # HARD FILTERING
    # --------------------------------------------------------

    candidates = {}

    categories = [
        "topwear",
        "bottomwear",
        "one_piece",
        "footwear"
    ]

    print(
        "\nAfter hard filtering:"
    )

    for category in categories:

        candidates[category] = filter_items(
            df,
            gender=gender,
            occasion=occasion,
            season=season,
            category=category
        )

        print(
            f"{category}: "
            f"{len(candidates[category])}"
        )

    # --------------------------------------------------------
    # ITEM-LEVEL CANDIDATE SCORING
    # --------------------------------------------------------

    selected_candidates = (
        select_all_candidates(
            candidates,
            occasion,
            season
        )
    )

    print(
        "\nAfter candidate scoring:"
    )

    for category in categories:

        print(
            f"{category}: "
            f"{len(selected_candidates[category])}"
        )

    # --------------------------------------------------------
    # DYNAMIC COMBINATIONS
    # --------------------------------------------------------

    outfits = generate_outfits(
        selected_candidates
    )

    print(
        "\nGenerated combinations:",
        len(outfits)
    )

    # --------------------------------------------------------
    # OUTFIT SCORING
    # --------------------------------------------------------

    scored_outfits = []

    for outfit in outfits:

        breakdown = score_outfit(
            outfit,
            occasion,
            season,
            USER_PREFERENCES,
            cv_features
        )

        outfit_copy = outfit.copy()

        outfit_copy["breakdown"] = (
            breakdown
        )

        scored_outfits.append(
            outfit_copy
        )

    scored_outfits.sort(
        key=lambda x:
        x["breakdown"]["score"],
        reverse=True
    )

    print(
        "Scored outfits:",
        len(scored_outfits)
    )

    # --------------------------------------------------------
    # DIVERSITY
    # --------------------------------------------------------

    final_outfits = (
        select_diverse_top_outfits(
            scored_outfits,
            limit=10
        )
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TOP 10 FINAL RECOMMENDATIONS"
    )

    print(
        "=" * 70
    )

    for index, outfit in enumerate(
        final_outfits,
        start=1
    ):

        breakdown = outfit[
            "breakdown"
        ]

        print(
            f"\nRecommendation {index}"
        )

        print(
            "-" * 50
        )

        if "one_piece" in outfit:

            item = outfit[
                "one_piece"
            ]

            print(
                "One-piece:",
                item["articleType"],
                f"({item['baseColour']})",
                f"[ID: {item['id']}]"
            )

        else:

            top = outfit[
                "topwear"
            ]

            bottom = outfit[
                "bottomwear"
            ]

            print(
                "Topwear:",
                top["articleType"],
                f"({top['baseColour']})",
                f"[ID: {top['id']}]"
            )

            print(
                "Bottomwear:",
                bottom["articleType"],
                f"({bottom['baseColour']})",
                f"[ID: {bottom['id']}]"
            )

        footwear = outfit[
            "footwear"
        ]

        print(
            "Footwear:",
            footwear["articleType"],
            f"({footwear['baseColour']})",
            f"[ID: {footwear['id']}]"
        )

        print(
            f"\nTOTAL SCORE: "
            f"{breakdown['score']}/100"
        )

        print(
            f"Occasion: "
            f"{breakdown['occasion']}"
        )

        print(
            f"Season: "
            f"{breakdown['season']}"
        )

        print(
            f"Colour: "
            f"{breakdown['colour']}"
        )

        print(
            f"Style: "
            f"{breakdown['style']}"
        )

        print(
            f"Formality: "
            f"{breakdown['formality']}"
        )

        print(
            f"Article: "
            f"{breakdown['article']}"
        )

        print(
            f"Preference: "
            f"{breakdown['preference']}"
        )

        print(
            f"CV Analysis: "
            f"{breakdown['cv']}"
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CV FEATURES USED IN SCORING"
    )

    print(
        "=" * 70
    )

    print(
        "Face shape:",
        cv_features["face_shape"]
    )

    print(
        "Skin tone:",
        cv_features["skin_tone"]
    )

    print(
        "Skin undertone:",
        cv_features["skin_undertone"]
    )

    print(
        "Hair color:",
        cv_features["hair_color"]
    )


if __name__ == "__main__":
    main()