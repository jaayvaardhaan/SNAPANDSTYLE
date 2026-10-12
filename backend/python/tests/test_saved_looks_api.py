import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import saved_looks_api


class SavedLooksTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "archive.json"
        self.override = patch.object(saved_looks_api, "ARCHIVE_PATH", self.path)
        self.override.start()
        self.look = {"id": "favorite-1", "title": "Interview Look", "source": "catalog",
                     "items": [{"name": "White shirt", "role": "Top", "image": "/catalog-images/123.jpg"}]}

    def tearDown(self):
        self.override.stop()
        self.directory.cleanup()

    def test_save_is_readable_by_a_new_client_from_disk(self):
        saved_looks_api.save_look(self.look)
        reply = saved_looks_api.list_saved_looks()["looks"]
        self.assertEqual(reply[0]["items"][0]["image"], "/catalog-images/123.jpg")
        self.assertEqual(json.loads(self.path.read_text())[0]["id"], "favorite-1")

    def test_replacing_favorite_keeps_one_entry(self):
        saved_looks_api.save_look(self.look)
        self.look["title"] = "Updated Look"
        saved_looks_api.save_look(self.look)
        looks = saved_looks_api.list_saved_looks()["looks"]
        self.assertEqual(len(looks), 1)
        self.assertEqual(looks[0]["title"], "Updated Look")

    def test_disk_failure_reports_failure_and_preserves_archive(self):
        saved_looks_api.save_look(self.look)
        before = self.path.read_bytes()
        with patch.object(saved_looks_api, "_write", side_effect=OSError("disk full")), self.assertRaises(HTTPException) as failure:
            saved_looks_api.save_look({**self.look, "title": "Not saved"})
        self.assertEqual(failure.exception.status_code, 500)
        self.assertEqual(self.path.read_bytes(), before)

    def test_camera_data_is_not_archived(self):
        self.look["capturedPhoto"] = "data:image/jpeg;base64,secret"
        self.look["preferences"] = {"category": "Men", "capturedPhoto": "secret"}
        self.look["items"][0]["image"] = "data:image/jpeg;base64,secret"
        saved_looks_api.save_look(self.look)
        self.assertNotIn("secret", self.path.read_text())


if __name__ == "__main__":
    unittest.main()
