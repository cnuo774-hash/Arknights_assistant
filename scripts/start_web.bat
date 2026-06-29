@echo off
chcp 65001 >nul
title Arknights Assistant - Web Server

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

set "PYTHON_EXE="

where python >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_EXE=python"
    goto :found
)

where python3 >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_EXE=python3"
    goto :found
)

echo [ERROR] Cannot find Python. Please install Python 3.11+ and add it to PATH.
pause
exit /b 1

:found
echo Using Python:
"%PYTHON_EXE%" --version
echo.
echo If this is the first run, install dependencies with:
echo   "%PYTHON_EXE%" -m pip install -r requirements_web.txt
echo.
echo ========================================
echo   Arknights Assistant
echo   Open: http://localhost:8090
echo   Press Ctrl+C to stop
echo ========================================
"%PYTHON_EXE%" -m uvicorn api:app --host 0.0.0.0 --port 8090 --reload
pause
