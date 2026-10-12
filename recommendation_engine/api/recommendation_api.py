from pathlib import Path
import math
import sys

from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = PROJECT_ROOT / "recommendation_engine" / "recommendation"
DATASET_PATH = PROJECT_ROOT / "datasets" / "snapstyle_items_tagged.csv"

sys.path.insert(0, str(ENGINE_DIR))

import catalog_filtering
from candidate_ranking import select_all_candidates
from outfit_generation import generate_outfits
from outfit_scoring import cv_item_score, preference_score, score_outfit, select_diverse_top_outfits
from appearance_features import get_default_cv_features

catalog_filtering.DATASET_PATH = str(DATASET_PATH)

app = FastAPI(title="SnapAndStyle Recommendation API")

OCCASION_TAGS = {
    "Casual": "casual_outing", "College": "college", "Office": "office",
    "Party": "party", "Wedding": "ethnic_event", "Date": "casual_outing",
    "Interview": "interview", "Travel": "travel",
}
STYLE_TAGS = {
    "Minimal": "basic", "Casual": "casual", "Streetwear": "denim",
    "Formal": "formal", "Elegant": "dressy", "Traditional": "traditional",
    "Sporty": "sports", "Trendy": "party",
}
FORMALITY = {
    "Formal": "formal", "Elegant": "dressy", "Traditional": "semi_formal",
    "Sporty": "active", "Casual": "casual", "Minimal": "casual",
    "Streetwear": "casual", "Trendy": "dressy",
}

DATE_ARTICLES = {
    "topwear": {"Shirts", "Blazers", "Waistcoat", "Kurtas", "Kurtis", "Nehru Jackets", "Tops", "Tunics", "Sweaters", "Jackets"},
    "bottomwear": {"Trousers", "Skirts", "Churidar", "Patiala", "Salwar"},
    "one_piece": {"Dresses", "Jumpsuit", "Kurta Sets", "Salwar and Dupatta"},
    "footwear": {"Formal Shoes", "Casual Shoes", "Heels", "Flats", "Booties"},
}
TRADITIONAL_DATE_ARTICLES = {
    "topwear": {"Kurtas", "Kurtis"},
    "bottomwear": {"Churidar", "Patiala", "Salwar", "Skirts", "Trousers"},
    "one_piece": {"Kurta Sets", "Salwar and Dupatta"},
    "footwear": {"Formal Shoes", "Heels", "Flats", "Booties"},
}
PARTY_ARTICLES = {
    "topwear": {"Shirts", "Blazers", "Tops", "Tunics", "Kurtas", "Kurtis"},
    "bottomwear": {"Trousers", "Jeans", "Skirts", "Churidar", "Patiala", "Salwar"},
    "one_piece": {"Dresses", "Jumpsuit", "Kurta Sets"},
    "footwear": DATE_ARTICLES["footwear"],
}
WEDDING_ARTICLES = {
    "topwear": {"Kurtas", "Kurtis", "Shirts", "Blazers", "Tops", "Tunics"},
    "bottomwear": {"Churidar", "Patiala", "Salwar", "Trousers", "Skirts"},
    "one_piece": {"Sarees", "Lehenga Choli", "Kurta Sets", "Salwar and Dupatta", "Dresses"},
    "footwear": {"Formal Shoes", "Heels", "Flats", "Booties"},
}
INTERVIEW_ARTICLES = {
    "topwear": {"Shirts", "Blazers"},
    "bottomwear": {"Trousers", "Skirts"},
    "one_piece": {"Dresses"},
    "footwear": {"Formal Shoes", "Heels", "Flats", "Booties"},
}


def json_safe_item(item):
    """Convert missing CSV cells to JSON null before returning an outfit."""
    return {
        key: None if pd.isna(value) or (
            isinstance(value, float) and not math.isfinite(value)
        ) else value
        for key, value in item.items()
    }

class RecommendationRequest(BaseModel):
    gender: str = Field(min_length=1)
    season: str = Field(min_length=1)
    occasion: str = Field(min_length=1)
    time: str = Field(min_length=1)
    style: str = Field(min_length=1)
    color: str | None = None
    cv_profile: dict | None = None
    limit: int = Field(default=3, ge=1, le=10)

@app.get("/health")
def health():
    return {"status": "UP", "service": "recommendation-engine"}

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        df = catalog_filtering.load_dataset()
        if "image_exists" in df:
            df = df[df["image_exists"].astype(str).str.lower() == "true"]
        # Some catalog rows label children's garments as Men/Women/Unisex.
        df = df[~df["productDisplayName"].fillna("").str.contains(
            r"\b(?:kids?|boys?|girls?|bab(?:y|ies)|infants?|juniors?)\b",
            case=False, regex=True,
        )]

        occasion = OCCASION_TAGS.get(request.occasion, request.occasion)
        if request.occasion == "Date" and request.style == "Traditional":
            occasion = "ethnic_event"
        season = "Fall" if request.season == "Monsoon" else request.season
        preferences = {
            "preferred_colour": request.color,
            "preferred_style": STYLE_TAGS.get(request.style, request.style),
            "preferred_formality": FORMALITY.get(request.style),
            "preferred_time": request.time,
        }
        cv_features = get_default_cv_features()
        if request.cv_profile:
            for key in ("face_shape", "skin_tone", "skin_undertone", "hair_color"):
                value = request.cv_profile.get(key)
                if isinstance(value, str) and value != "unknown":
                    cv_features[key] = value
            confidence = request.cv_profile.get("confidence", {})
            if isinstance(confidence, dict):
                for key in cv_features["confidence"]:
                    value = confidence.get(key)
                    if isinstance(value, (int, float)) and 0 <= value <= 1:
                        cv_features["confidence"][key] = float(value)

        candidates, fallback_categories = {}, []
        for category in ("topwear", "bottomwear", "one_piece", "footwear"):
            # The catalog has no interview-tagged Unisex items. For that choice,
            # consider adult items from both sections, then apply interview rules.
            catalog_gender = None if request.gender == "Unisex" and request.occasion == "Interview" else request.gender
            items = catalog_filtering.filter_items(
                df, gender=catalog_gender, occasion=occasion, category=category
            )
            items = _context_items(items, request, category)
            if items.empty and request.occasion != "Interview":
                items = catalog_filtering.filter_items(
                    df, gender=request.gender, occasion="casual_outing", category=category
                )
                items = _context_items(items, request, category)
                if not items.empty:
                    fallback_categories.append(category)
            candidates[category] = items

        candidates = select_all_candidates(
            candidates,
            requested_occasion=occasion,
            requested_season=season,
            limit=80
        )

        # The generic candidate shortlist does not know the selected style,
        # colour, time or CV profile. Re-rank before building combinations.
        for category, items in candidates.items():
            if items.empty:
                continue
            items = items.copy()
            items["personalized_score"] = items.apply(
                lambda row: row["item_score"]
                + 16 * preference_score(row, preferences)
                + 8 * cv_item_score(row, cv_features),
                axis=1,
            )
            candidates[category] = items.sort_values(
                by="personalized_score", ascending=False, kind="stable"
            ).head(10).reset_index(drop=True)

        outfits = generate_outfits(candidates)

        scored_outfits = []
        for outfit in outfits:
            scores = score_outfit(
                outfit,
                occasion,
                season,
                preferences,
                cv_features
            )
            scored_outfits.append({**outfit, "scores": scores})

        scored_outfits.sort(
            key=lambda item: item["scores"]["score"],
            reverse=True
        )

        selected = select_diverse_top_outfits(
            scored_outfits,
            limit=request.limit
        )

        results = []
        for index, outfit in enumerate(selected, start=1):
            results.append({
                "outfitId": index,
                "score": outfit["scores"]["score"],
                "items": {
                    category: _public_item(outfit[category])
                    for category in [
                        "topwear", "bottomwear", "one_piece", "footwear"
                    ]
                    if category in outfit
                }
            })

        return {
            "status": "success",
            "recommendationType": "existing-scoring-engine",
            "matchedOccasionTag": occasion,
            "matchedSeason": season,
            "fallbackCategories": fallback_categories,
            "resultCount": len(results),
            "outfits": results
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {exc}"
        ) from exc


def _public_item(item):
    public = json_safe_item(item)
    filename = public.get("image_path")
    if isinstance(filename, str) and filename.endswith(".jpg") and "/" not in filename and "\\" not in filename:
        public["image_url"] = "/catalog-images/" + filename
    return public


def _context_items(items, request, category):
    if request.occasion == "Date":
        allowed = TRADITIONAL_DATE_ARTICLES if request.style == "Traditional" else DATE_ARTICLES
    elif request.occasion == "Party":
        allowed = PARTY_ARTICLES
    elif request.occasion == "Wedding":
        allowed = WEDDING_ARTICLES
    elif request.occasion == "Interview":
        allowed = INTERVIEW_ARTICLES
    else:
        return items
    article_types = allowed[category]
    if request.occasion == "Date" and category == "footwear" and request.time in {"Evening", "Night"}:
        article_types = article_types - {"Casual Shoes"}
    return items[items["articleType"].isin(article_types)]
