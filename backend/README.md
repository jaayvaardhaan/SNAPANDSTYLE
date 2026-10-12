# Backend

`python/` contains the local FastAPI application (`backend_server.py`) and the saved-look archive API (`saved_looks_api.py`). It calls the separate `fashion-cv/` module and mounts the recommendation engine, frontend and catalog images.

From the project root:

```powershell
python -m pip install -r backend/python/requirements.txt
python backend/python/backend_server.py
python -m unittest discover -s backend/python/tests -v
```

The server uses port 8000 by default. `SNAPSTYLE_HOST` and `SNAPSTYLE_PORT` can override its address. Favorites are stored at project `data/saved_looks.json`.

`src/` and `pom.xml` contain the separate Java Spring Boot catalog API. Its standard Maven layout stays intact. Run it from this backend folder with `mvn spring-boot:run` when needed; the current browser flow uses the Python application.
