@echo off
:: Check for administrative rights
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Requesting Administrator privileges to copy into Photoshop Scripts folder...
    powershell -Command "Start-Process cmd -ArgumentList '/c \"\"%~dpnx0\"\"' -Verb RunAs"
    exit /b
)

title Install AI Blemish Remover to Photoshop 2020
cd /d "%~dp0"
echo =======================================================
echo  Installing AI Blemish Remover to Photoshop 2020 Scripts
echo =======================================================

set DEST=C:\Program Files\Adobe\Adobe Photoshop 2020\Presets\Scripts\AI_Blemish_Remover.jsx

copy /Y "AI_Blemish_Remover.jsx" "%DEST%"

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] Script installed successfully to:
    echo %DEST%
    echo.
    echo Now in Photoshop, simply go to:
    echo    File ^> Scripts ^> AI_Blemish_Remover
) else (
    echo.
    echo [!] Failed to copy script. Please check permissions.
)

echo =======================================================
pause
