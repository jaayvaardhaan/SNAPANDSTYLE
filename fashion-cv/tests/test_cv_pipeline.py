import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi import Response

CV_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CV_DIR))
sys.path.insert(0, str(CV_DIR.parent / "backend" / "python"))

from backend_server import _model_status, decode_jpeg, health
from face_shape_analysis import classify_face_geometry
from face_segmentation import _decode_logits
from hair_color_analysis import classify_hair_pixels
from appearance_pipeline import _unknown_hair, build_profile
from skin_tone_analysis import _ciede2000


class CvContractTests(unittest.TestCase):
    def test_models_are_present(self):
        self.assertTrue(all(item["present"] for item in _model_status().values()))

    def test_project_health_includes_catalog(self):
        response = Response()
        status = health(response)
        self.assertEqual(status["status"], "ready")
        self.assertTrue(all(status["catalog"].values()))

    def test_jpeg_decoder_accepts_valid_image(self):
        source = np.full((300, 400, 3), 127, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", source)
        self.assertTrue(ok)
        decoded = decode_jpeg(encoded.tobytes())
        self.assertEqual(decoded.shape, source.shape)

    def test_jpeg_decoder_rejects_wrong_format_and_small_image(self):
        ok, png = cv2.imencode(".png", np.zeros((300, 300, 3), dtype=np.uint8))
        self.assertTrue(ok)
        with self.assertRaisesRegex(ValueError, "not a JPEG"):
            decode_jpeg(png.tobytes())
        ok, tiny = cv2.imencode(".jpg", np.zeros((128, 128, 3), dtype=np.uint8))
        self.assertTrue(ok)
        with self.assertRaisesRegex(ValueError, "too small"):
            decode_jpeg(tiny.tobytes())

    def test_face_geometry_returns_supported_label(self):
        label, ranking = classify_face_geometry(1.31, 0.85, 0.79)
        self.assertEqual(label, "oval")
        self.assertEqual(len(ranking), 6)

    def test_parser_decodes_nchw_and_nhwc(self):
        nchw = np.zeros((1, 19, 4, 5), dtype=np.float32)
        nchw[:, 17] = 1
        self.assertTrue(np.all(_decode_logits(nchw) == 17))
        nhwc = np.zeros((1, 4, 5, 19), dtype=np.float32)
        nhwc[..., 1] = 1
        self.assertTrue(np.all(_decode_logits(nhwc) == 1))

    def test_ciede2000_reference_pair(self):
        value = _ciede2000([50, 2.6772, -79.7751], [50, 0, -82.7485])
        self.assertAlmostEqual(value, 2.0425, places=4)

    def test_uniform_dark_hair_is_classifiable(self):
        pixels = np.tile(np.array([[15, 18, 20]], dtype=np.uint8), (300, 1))
        result = classify_hair_pixels(pixels)
        self.assertIn(result["hair_color"], {"black", "dark_brown"})

    def test_unknown_hair_preserves_profile_contract(self):
        face = {"face_shape": "oval", "confidence": 0.6, "quality_warnings": [], "sampling": {}}
        parsing = {"statistics": {"hair_pixels": 0}}
        skin = {
            "skin_tone": "medium", "skin_undertone": "neutral",
            "confidence": {"skin_tone": 0.5, "skin_undertone": 0.4},
            "quality_warnings": [],
        }
        profile = build_profile(face, parsing, _unknown_hair("no hair"), skin, 640, 480)
        self.assertEqual(profile["hair_color"], "unknown")
        self.assertEqual(profile["confidence"]["hair_color"], 0.0)


if __name__ == "__main__":
    unittest.main()
