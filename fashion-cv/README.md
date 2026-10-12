# Snap&Style computer vision

This module accepts one approved face photo and estimates four appearance features:

- face shape: `oval`, `round`, `square`, `oblong`, `heart`, or `diamond`
- apparent skin tone: `deep`, `medium-deep`, `medium`, `medium-light`, or `light`
- apparent skin undertone: `warm`, `cool`, or `neutral`
- apparent hair color: `black`, `dark_brown`, `brown`, `light_brown`, `blonde`, `red_or_auburn`, `gray_or_white`, or `unknown`

These are heuristic estimates from one camera image. The confidence fields are relative reliability scores, not calibrated probabilities.

## Install

From `SNAPANDSTYLE/fashion-cv`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The two model files must remain in `models/`. Their expected hashes are recorded in `models/SHA256SUMS`.

## Validate

```powershell
python cv_preflight.py
python cv_preflight.py --deep
python -m unittest discover -s tests -v
```

Run in-memory inference on a real local photo without writing the photo, masks, or JSON:

```powershell
python cv_preflight.py --image "C:\path\to\photo.jpg"
```

## Run the browser flow

Use `start_project.bat` in the project root, or start the backend from the project root:

```powershell
python backend/python/backend_server.py
```

Then open `http://localhost:8000/index.html`. The same FastAPI process serves the frontend and CV endpoint.

## API contract

`POST /api/analyze` accepts JPEG bytes with `Content-Type: image/jpeg`. The compressed limit is 8 MB and both image sides must be at least 256 pixels.

```json
{
  "profile": {
    "face_shape": "oval",
    "skin_tone": "medium",
    "skin_undertone": "neutral",
    "hair_color": "black",
    "confidence": {
      "face_shape": 0.61,
      "skin_tone": 0.54,
      "skin_undertone": 0.39,
      "hair_color": 0.66
    },
    "quality_warnings": []
  }
}
```

Status codes: `400` empty request, `413` size limit, `415` wrong content type, `422` malformed or unsuitable photo, `429` busy, and `503` unavailable model/runtime.

`GET /api/health` checks that both model files, the tagged catalog CSV, and the image directory are present. It also reports whether each CV runtime has loaded. Use `cv_preflight.py --deep` to prove model initialization.

## Storage and privacy

`/api/analyze` processes the approved JPEG in memory and does not save photos, masks, or JSON. The separate `appearance_pipeline.py` command writes diagnostic outputs and is intended for deliberate local debugging.

The service binds to `127.0.0.1` by default. Add authentication, HTTPS, and rate limiting before exposing it publicly.

## Integration

The frontend sends the selected occasion, season, time, category, style, colour, and optional four-feature CV profile to `POST /api/recommendations/recommend`. The recommendation engine uses the CV confidence scores during outfit scoring; unknown fields contribute no CV signal. The response contains ranked catalog outfits and local `/catalog-images/` URLs. The same CV server hosts this recommendation route, so `start_project.bat` starts the complete browser-to-recommendation flow.

See `MODEL_CARD.md` for preprocessing, provenance gaps, and limitations. Model loading does not establish accuracy; evaluate on consented, labeled real photos before reporting accuracy.
