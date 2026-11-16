@echo off
title Multimodal AI Research Server
cd /d "%~dp0"

echo =====================================================================
echo   Starting Multimodal AI Forensics ^& Image Generation Server
echo   Project Directory: %~dp0
echo   Local Dashboard:   http://127.0.0.1:8000
echo   API Docs:          http://127.0.0.1:8000/docs
echo =====================================================================
echo.

:: Check if port 8000 is currently occupied and free it if needed
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo [Info] Port 8000 is occupied by PID %%a. Terminating previous server instance...
    taskkill /f /pid %%a >nul 2>&1
)

echo [Server] Launching application on http://127.0.0.1:8000...
echo [Server] Press CTRL+C at any time to stop the server.
echo.

python backend\main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Error] Server stopped with error code %ERRORLEVEL%.
    pause
)
