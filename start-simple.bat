@echo off
chcp 65001 >nul 2>&1
title AIFX Trading Signals - Direct Python Start

echo.
echo 🚀 AIFX Trading Signals - Simple Direct Start
echo ===========================================
echo This version runs directly with Python (no Docker needed)
echo.

REM Check current directory
echo Current directory: %CD%
echo.

REM Check if we're in the right directory
if not exist "src\main\python\web_interface.py" (
    echo ❌ Error: web_interface.py not found
    echo Please run this script from the AIFX project directory
    echo Expected path: C:\Users\butte\OneDrive\桌面\AIFX_CLAUDE
    echo Current path: %CD%
    echo.
    pause
    exit /b 1
)

echo ✅ Found web_interface.py
echo.

REM Check if Python is installed
echo Checking Python installation...
python --version 2>nul
if errorlevel 1 (
    echo ❌ Python is not installed or not accessible
    echo.
    echo Please:
    echo 1. Install Python from https://python.org/downloads/
    echo 2. Make sure Python is added to PATH
    echo 3. Restart this script
    echo.
    pause
    exit /b 1
)

echo ✅ Python is available
echo.

REM Check if requirements file exists
if not exist "requirements-web.txt" (
    echo ❌ requirements-web.txt not found
    echo Creating minimal requirements file...

    echo fastapi==0.104.1> requirements-web.txt
    echo uvicorn[standard]==0.24.0>> requirements-web.txt
    echo websockets==12.0>> requirements-web.txt
    echo jinja2==3.1.2>> requirements-web.txt
    echo pandas==2.1.3>> requirements-web.txt
    echo numpy==1.25.2>> requirements-web.txt
    echo scikit-learn==1.3.2>> requirements-web.txt
    echo xgboost==2.0.2>> requirements-web.txt
    echo yfinance==0.2.28>> requirements-web.txt
    echo python-dateutil==2.8.2>> requirements-web.txt
    echo pytz==2023.3>> requirements-web.txt
    echo psutil==5.9.6>> requirements-web.txt
    echo aiofiles==23.2.1>> requirements-web.txt

    echo ✅ Created requirements-web.txt
)

REM Install dependencies
echo Installing Python dependencies...
echo This may take a few minutes...
echo.

pip install -r requirements-web.txt

if errorlevel 1 (
    echo.
    echo ❌ Failed to install dependencies
    echo.
    echo Common solutions:
    echo 1. Run as Administrator
    echo 2. Try: pip install --user -r requirements-web.txt
    echo 3. Create virtual environment first
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ Dependencies installed successfully
echo.

REM Create necessary directories
echo Creating necessary directories...
if not exist logs mkdir logs
if not exist "data\cache" mkdir data\cache
echo ✅ Directories created
echo.

REM Start the application
echo.
echo 🚀 Starting AIFX Trading Signals Web Interface...
echo.
echo Access your dashboard at: http://localhost:8080
echo Press Ctrl+C to stop the service
echo.

REM Set environment variable
set PYTHONPATH=%CD%

REM Start with detailed output
echo Starting Python application...
python -m src.main.python.web_interface

if errorlevel 1 (
    echo.
    echo ❌ Failed to start the application
    echo.
    echo Common solutions:
    echo 1. Check if port 8080 is already in use
    echo 2. Make sure all dependencies are installed
    echo 3. Run: pip install --upgrade -r requirements-web.txt
    echo.
    echo Press any key to see the error details...
    pause >nul
)

echo.
echo Application stopped.
echo Press any key to exit...
pause >nul