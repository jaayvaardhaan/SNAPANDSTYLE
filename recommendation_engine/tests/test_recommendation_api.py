"""Regression tests for real catalog rows with missing CSV cells."""
import sys
import unittest
from pathlib import Path

from starlette.responses import JSONResponse
from pydantic import ValidationError

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from recommendation_engine.api.recommendation_api import RecommendationRequest, json_safe_item, recommend


class RecommendationSerializationTests(unittest.TestCase):
    def test_formal_unisex_interview_uses_adult_interview_clothes(self):
        response = recommend(RecommendationRequest(
            gender="Unisex", season="Monsoon", occasion="Interview",
            time="Afternoon", style="Formal", color="Neutral", limit=4,
        ))
        self.assertEqual(response["fallbackCategories"], [])
        self.assertEqual(response["resultCount"], 4)
        allowed = {
            "topwear": {"Shirts", "Blazers"},
            "bottomwear": {"Trousers", "Skirts"},
            "one_piece": {"Dresses"},
            "footwear": {"Formal Shoes", "Heels", "Flats", "Booties"},
        }
        for outfit in response["outfits"]:
            for category, item in outfit["items"].items():
                self.assertIn(item["articleType"], allowed[category])
                self.assertNotRegex(item["productDisplayName"].lower(),
                                    r"\b(?:kids?|boys?|girls?|bab(?:y|ies)|infants?|juniors?)\b")

    def test_traditional_spring_night_date_avoids_casual_clothes(self):
        response = recommend(RecommendationRequest(
            gender="Men", season="Spring", occasion="Date",
            time="Night", style="Traditional", color="Neutral", limit=4,
        ))
        self.assertEqual(response["resultCount"], 4)
        self.assertEqual(len({outfit["items"]["topwear"]["id"]
                              for outfit in response["outfits"]}), 4)
        for outfit in response["outfits"]:
            self.assertEqual(outfit["items"]["topwear"]["articleType"], "Kurtas")
            self.assertEqual(outfit["items"]["bottomwear"]["articleType"], "Churidar")
            self.assertEqual(outfit["items"]["footwear"]["articleType"], "Formal Shoes")

    def test_five_style_choices_are_required(self):
        for omitted in ("gender", "season", "occasion", "time", "style"):
            with self.subTest(omitted=omitted), self.assertRaises(ValidationError):
                values = {"gender": "Men", "season": "Spring", "occasion": "Date",
                          "time": "Night", "style": "Traditional"}
                del values[omitted]
                RecommendationRequest(**values)

    def test_selected_preferences_and_cv_profile_affect_catalog_score(self):
        base = dict(gender="Women", occasion="Date", season="Spring",
                    style="Elegant", color="Black", time="Night", limit=1)
        without_cv = recommend(RecommendationRequest(**base))
        profile = {
            "face_shape": "oval", "skin_tone": "medium",
            "skin_undertone": "warm", "hair_color": "black",
            "confidence": {"face_shape": 0.9, "skin_tone": 0.8,
                           "skin_undertone": 0.8, "hair_color": 0.9},
        }
        with_cv = recommend(RecommendationRequest(**base, cv_profile=profile))
        changed_preferences = recommend(RecommendationRequest(
            **{**base, "style": "Sporty", "color": "Pink", "time": "Morning"},
            cv_profile=profile,
        ))
        self.assertEqual(with_cv["resultCount"], 1)
        self.assertNotEqual(without_cv["outfits"][0]["score"],
                            with_cv["outfits"][0]["score"])
        self.assertNotEqual(with_cv["outfits"][0]["score"],
                            changed_preferences["outfits"][0]["score"])
        for item in with_cv["outfits"][0]["items"].values():
            self.assertTrue(item["image_url"].startswith("/catalog-images/"))

    def test_missing_catalog_values_are_json_null(self):
        self.assertIsNone(json_safe_item({"season": float("nan")})["season"])
        for gender, season in (
            ("Unisex", "Winter"),
            ("Men", "Monsoon"),
            ("Women", "Monsoon"),
            ("Unisex", "Monsoon"),
        ):
            with self.subTest(gender=gender, season=season):
                response = recommend(RecommendationRequest(
                    gender=gender, occasion="sports", season=season,
                    time="Morning", style="Sporty", limit=10,
                ))
                self.assertGreater(response["resultCount"], 0)
                encoded = JSONResponse(response).body.decode("utf-8")
                self.assertNotIn("NaN", encoded)
                self.assertNotIn("Infinity", encoded)
                self.assertTrue(response["outfits"])

    def test_recommendation_with_required_choices_serializes(self):
        response = recommend(RecommendationRequest(
            gender="Men", season="Summer", occasion="College",
            time="Morning", style="Casual",
        ))
        self.assertEqual(response["resultCount"], 3)
        self.assertTrue(JSONResponse(response).body)


if __name__ == "__main__":
    unittest.main()
