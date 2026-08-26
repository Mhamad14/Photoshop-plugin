@echo off
title AI Retouching Server - FastAPI + Simple-LaMa
cd /d "%~dp0"
echo =======================================================
echo  Starting AI Retouching Local Server (FastAPI + Simple-LaMa)
echo =======================================================

:: Try Python 3.11 launcher first (optimal for PyTorch/CUDA/ONNX)
py -3.11 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] Using Python 3.11 launcher...
    py -3.11 run_server.py
    goto :done
)

:: Try standard py launcher
py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [*] Using py -3 launcher...
    py -3 run_server.py
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

