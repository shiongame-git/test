@echo off
setlocal
cd /d "%~dp0"

where powershell.exe >nul 2>&1
if errorlevel 1 (
    echo PowerShell was not found.
    echo Windows PowerShell is required to start the packing3d GUI.
    echo.
    pause
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0demo\windows_launcher.ps1"
if errorlevel 1 (
    echo The packing3d GUI launcher could not be started.
    echo See README.md for troubleshooting information.
    echo.
    pause
    exit /b 1
)

exit /b 0
