@echo off
title SnapStyle Project Launcher

set "PROJECT=%~dp0"

echo ==========================================
echo       Starting SNAP AND STYLE
echo ==========================================

echo [1/3] Starting frontend server...

start "SnapStyle Frontend" /D "%PROJECT%frontend" cmd /k "python -m http.server 8000"

echo [2/3] Starting CV worker...

start "SnapStyle CV Worker" /D "%PROJECT%fashion-cv" cmd /k "python cv_worker.py"

echo [3/3] Opening website...

timeout /t 3 /nobreak >nul

start "" "http://localhost:8000/frontend.html"

echo ==========================================
echo       SnapStyle launched!
echo ==========================================

exit