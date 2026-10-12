"""Durable local outfit favorites. Camera photos are never stored here."""
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException

ARCHIVE_PATH = Path(__file__).resolve().parents[2] / "data" / "saved_looks.json"
router = APIRouter(prefix="/api/saved-looks")
_lock = threading.Lock()


def _read():
    if not ARCHIVE_PATH.exists():
        return []
    value = json.loads(ARCHIVE_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Invalid saved-look archive")
    return value


def _write(looks):
    ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    filename = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=ARCHIVE_PATH.parent,
                                         delete=False) as handle:
            filename = handle.name
            json.dump(looks, handle, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(filename, ARCHIVE_PATH)
    finally:
        if filename and os.path.exists(filename):
            os.unlink(filename)


def _snapshot(value):
    if not isinstance(value.get("title"), str) or not value["title"].strip():
        raise HTTPException(422, detail="A saved look needs a title")
    items = value.get("items")
    if not isinstance(items, list) or not items or len(items) > 12:
        raise HTTPException(422, detail="A saved look needs clothing items")
    # Save only outfit display data, never arbitrary profile or photo fields.
    allowed = ("title", "subtitle", "score", "reason", "palette", "source",
               "savedAt", "outfitId", "recreatedFrom")
    result = {key: value[key] for key in allowed if key in value}
    preferences = value.get("preferences")
    if isinstance(preferences, dict):
        result["preferences"] = {key: preferences[key] for key in
            ("occasion", "season", "time", "category", "style", "color")
            if isinstance(preferences.get(key), str)}
    result["id"] = str(value.get("id") or uuid4())
    result["items"] = []
    for item in items:
        if not isinstance(item, dict):
            raise HTTPException(422, detail="Invalid clothing item")
        piece = {key: item[key] for key in ("role", "name", "description", "icon") if key in item}
        image = item.get("image")
        piece["image"] = image if isinstance(image, str) and re.fullmatch(r"/catalog-images/\d+\.jpg", image) else None
        result["items"].append(piece)
    if len(json.dumps(result)) > 64000:
        raise HTTPException(422, detail="Saved look is too large")
    return result


@router.get("")
def list_saved_looks():
    try:
        with _lock:
            return {"looks": _read()}
    except (OSError, ValueError) as exc:
        raise HTTPException(500, detail="Could not read saved looks from disk") from exc


@router.post("")
def save_look(value: dict):
    look = _snapshot(value)
    try:
        with _lock:
            looks = _read()
            looks = [old for old in looks if str(old.get("id")) != look["id"]]
            looks.insert(0, look)
            _write(looks)
        return {"look": look}
    except (OSError, ValueError) as exc:
        raise HTTPException(500, detail="Could not save this look to disk") from exc
