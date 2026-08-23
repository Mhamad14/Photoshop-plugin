@echo off
title AI Retouching Server - FastAPI + Simple-LaMa
cd /d "%~dp0"
echo =======================================================
echo  Starting AI Retouching Local Server (FastAPI + Simple-LaMa)
echo =======================================================

py -3.13 run_server.py
if %errorlevel% neq 0 (
    echo.
    echo [!] py -3.13 failed with code %errorlevel%. Trying python...
    python run_server.py
)

echo.
echo =======================================================
echo Server process exited.
echo =======================================================
pause
