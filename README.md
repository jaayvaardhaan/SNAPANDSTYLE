# SNAPANDSTYLE
AN AI BASED ENGINE FOR OUTFIT SUGGESTIONS

## Run locally on Windows

From this project folder, install the Python packages with `python -m pip install -r backend/python/requirements.txt`, then run `start_project.bat`. The launcher starts one local server at `http://localhost:8000/index.html` and waits until the CV models and clothing catalog are present.

The browser sends an approved JPEG to `POST /api/analyze`. It then sends the selected preferences and the returned appearance profile to `POST /api/recommendations/recommend`. The recommendation engine ranks clothing from `datasets/snapstyle_items_tagged.csv`; product photos are served from `datasets/kaggle_fashion/images/`.

The separate Java catalog API under `backend/` is available for other clients, but the current browser flow uses the recommendation engine mounted in the local Python server. Restart `start_project.bat` after changing server files.

## Folder layout

```text
SNAPANDSTYLE/
  start_project.bat          Windows launcher
  backend/
    python/                 Web/CV server, saved-look API, dependencies and tests
    src/                    Java Spring Boot catalog API and tests
    pom.xml                 Java build configuration
  frontend/                 Browser page and automatic capture JavaScript
  fashion-cv/               CV analysis code, models, checks, input/output and tests
  recommendation_engine/    Recommendation API, filtering/scoring, dataset scripts and tests
  datasets/
    kaggle_fashion/          Active product images and original styles.csv
    processed/              Derived DeepFashion data and crop samples
    archive/                Preserved duplicate downloads, metadata copies and Mac archive files
    *.csv                   Active cleaned/tagged clothing catalogs
  data/                     Persistent saved looks (created/updated by the server)
  docs/archive/             Historical project notes
  tools/audit/              Audit utilities, historical reports and verification scripts
```

The runtime server lives in `backend/python/backend_server.py`; the CV module itself stays in `fashion-cv/`. Run the backend directly with `python backend/python/backend_server.py`. Paths resolve from the project location, so launching from another working directory is supported.

Saved outfits are stored in `data/saved_looks.json`. The archive includes clothing details and catalog image paths, not camera photos. Existing browser favorites migrate when the frontend connects to the server.

## Checks

```powershell
python -m unittest discover -s fashion-cv/tests -v
python -m unittest discover -s backend/python/tests -v
python -m unittest discover -s recommendation_engine/tests -v
node tools/audit/check_frontend.cjs
```
