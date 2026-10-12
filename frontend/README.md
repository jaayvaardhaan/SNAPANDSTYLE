# Frontend

`index.html` is the AngularJS application. `camera_auto_capture.js` handles guided face capture. The Python backend serves both files and the API routes from the same address.

Use the project-root `start_project.bat`, then open `http://localhost:8000/index.html`. A standalone static server cannot provide CV, recommendations or saved-look storage.
