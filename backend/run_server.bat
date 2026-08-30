@echo off
title AI Retouching Server - FastAPI + Simple-LaMa
cd /d "%~dp0"
echo =======================================================
echo  Starting AI Retouching Local Server (FastAPI + Simple-LaMa)
echo =======================================================

:: Python 3.14 is NOT supported yet (simple-lama-inpainting pins numpy 1.x,
:: which has no 3.14 prebuilt wheel and fails to compile from source).
:: Prefer 3.13, then 3.12/3.11.

py -3.13 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] Using Python 3.13
    py -3.13 run_server.py
    goto :done
)

py -3.12 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] Using Python 3.12
    py -3.12 run_server.py
    goto :done
)

py -3.11 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] Using Python 3.11
    py -3.11 run_server.py
    goto :done
)

:: Fallback to python in PATH
echo [*] Trying system python...
python run_server.py

:done
echo.
echo =======================================================
echo Server process exited.
echo =======================================================
pause
