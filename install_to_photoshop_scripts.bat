@echo off
:: Check for administrative rights
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Requesting Administrator privileges to copy into Photoshop Scripts folder...
    powershell -Command "Start-Process cmd -ArgumentList '/c \"\"%~dpnx0\"\"' -Verb RunAs"
    exit /b
)

title Install AI Blemish Remover to Photoshop Scripts
cd /d "%~dp0"
echo =======================================================
echo  Installing AI Retouch Scripts to Photoshop Versions
echo =======================================================

setlocal enabledelayedexpansion
set FOUND_PS=0

for /d %%P in ("C:\Program Files\Adobe\Adobe Photoshop *") do (
    if exist "%%P\Presets\Scripts" (
        echo [*] Discovered: %%P
        copy /Y "AI_Blemish_Remover.jsx" "%%P\Presets\Scripts\AI_Blemish_Remover.jsx" >nul
        copy /Y "AI_Blemish_Remover_Dialog.jsx" "%%P\Presets\Scripts\AI_Blemish_Remover_Dialog.jsx" >nul
        if !errorlevel! equ 0 (
            echo     [+] Successfully copied scripts to %%P\Presets\Scripts\
            set FOUND_PS=1
        ) else (
            echo     [!] Failed copying to %%P
        )
    )
)

for /d %%P in ("C:\Program Files (x86)\Adobe\Adobe Photoshop *") do (
    if exist "%%P\Presets\Scripts" (
        echo [*] Discovered: %%P
        copy /Y "AI_Blemish_Remover.jsx" "%%P\Presets\Scripts\AI_Blemish_Remover.jsx" >nul
        copy /Y "AI_Blemish_Remover_Dialog.jsx" "%%P\Presets\Scripts\AI_Blemish_Remover_Dialog.jsx" >nul
        if !errorlevel! equ 0 (
            echo     [+] Successfully copied scripts to %%P\Presets\Scripts\
            set FOUND_PS=1
        )
    )
)

if %FOUND_PS% equ 0 (
    echo [!] No Adobe Photoshop installations found in standard program folders.
    echo     You can still run the script directly inside Photoshop via:
    echo     File ^> Scripts ^> Browse... ^> AI_Blemish_Remover.jsx
) else (
    echo.
    echo =======================================================
    echo [SUCCESS] Scripts installed into Photoshop!
    echo.
    echo To use the AI plugin in Photoshop:
    echo 1. Launch the backend: run backend\run_server.bat
    echo 2. In Photoshop, open a portrait photo
    echo 3. Go to: File ^> Scripts ^> AI_Blemish_Remover
    echo =======================================================
)

pause

