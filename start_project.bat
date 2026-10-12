@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
if not exist "%ROOT%backend\python\backend_server.py" (
    echo Missing backend\python\backend_server.py. Extract the project files into the correct folders.
    pause
    exit /b 1
)
echo Starting SnapStyle frontend, backend, CV, and recommendation engine...
start "SnapStyle Backend" /D "%ROOT%backend\python" cmd /k "python backend_server.py"

echo Waiting for CV models and clothing catalog...
for /L %%I in (1,1,90) do (
    powershell -NoProfile -Command "try { $r = Invoke-RestMethod 'http://127.0.0.1:8000/api/health' -TimeoutSec 2; if ($r.status -eq 'ready') { exit 0 } } catch {} ; exit 1" >nul 2>&1
    if not errorlevel 1 goto READY
    timeout /t 1 /nobreak >nul
)
echo Project API did not become ready. Check the service window for missing packages, models, or catalog files.
pause
exit /b 1

:READY
start "" "http://localhost:8000/index.html"
echo SnapStyle started. Keep the backend command window open.
endlocal
