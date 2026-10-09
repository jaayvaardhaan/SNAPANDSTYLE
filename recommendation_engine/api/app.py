from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENGINE_DIR = PROJECT_ROOT / "recommendation_engine" / "recommendation"
DATASET_PATH = PROJECT_ROOT / "datasets" / "snapstyle_items_tagged.csv"

sys.path.insert(0, str(ENGINE_DIR))

import filtering
from candidate_scoring import select_all_candidates
from combination import generate_outfits
from scoring import score_outfit, select_diverse_top_outfits

filtering.DATASET_PATH = str(DATASET_PATH)

app = FastAPI(title="SnapAndStyle Recommendation API")

class RecommendationRequest(BaseModel):
    gender: str = "Men"
    season: str = "Fall"
    occasion: str = "college"
    limit: int = Field(default=3, ge=1, le=10)

@app.get("/health")
def health():
    return {"status": "UP", "service": "recommendation-engine"}

@app.post("/recommend")
def recommend(request: RecommendationRequest):
    try:
        df = filtering.load_dataset()

        candidates = {
            category: filtering.filter_items(
                df,
                gender=request.gender,
                occasion=request.occasion,
                category=category
            )
            for category in ["topwear", "bottomwear", "one_piece", "footwear"]
        }

        candidates = select_all_candidates(
            candidates,
            requested_occasion=request.occasion,
            requested_season=request.season,
            limit=10
        )

        outfits = generate_outfits(candidates)

        scored_outfits = []
        for outfit in outfits:
            scores = score_outfit(
                outfit,
                request.occasion,
                request.season,
                {},
                {}
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
                    category: outfit[category]
                    for category in [
                        "topwear", "bottomwear", "one_piece", "footwear"
                    ]
                    if category in outfit
                }
            })

        return {
            "status": "success",
            "recommendationType": "existing-scoring-engine",
            "resultCount": len(results),
            "outfits": results
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Recommendation failed: {exc}"
        ) from exc
