@echo off
title Install AI Retouch Plugin to Photoshop
cd /d "%~dp0"
echo =======================================================
echo  Installing AI Retouch Plugin (Direct Mode)
echo =======================================================

set TARGET_USER=%APPDATA%\Adobe\UXP\Plugins\External\com.antigravity.airetouch
set TARGET_SYSTEM=%COMMONPROGRAMFILES%\Adobe\UXP\extensions\com.antigravity.airetouch

echo [*] Copying plugin files to User UXP folder:
echo     %TARGET_USER%
if not exist "%TARGET_USER%" mkdir "%TARGET_USER%"
xcopy /E /I /Y "uxp-plugin\*" "%TARGET_USER%\"

echo.
echo [*] Copying plugin files to System UXP extensions folder:
echo     %TARGET_SYSTEM%
if not exist "%TARGET_SYSTEM%" mkdir "%TARGET_SYSTEM%"
xcopy /E /I /Y "uxp-plugin\*" "%TARGET_SYSTEM%\" 2>nul

echo.
echo =======================================================
echo [SUCCESS] Plugin files installed!
echo.
echo Next steps:
echo 1. Open Photoshop.
echo 2. Go to Edit ^> Preferences ^> Plugins.
echo 3. Ensure 'Enable Developer Mode' is CHECKED.
echo 4. Restart Photoshop if it was already open.
echo 5. Check the 'Plugins' menu at the top of Photoshop!
echo =======================================================
pause
