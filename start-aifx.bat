@echo off
chcp 65001 >nul 2>&1
title AIFX Trading System - Unified Launcher
REM AIFX Trading System - 統一啟動器

echo.
echo 🚀 AIFX Trading System - Unified Launcher
echo ==========================================
echo Choose your preferred launch mode:
echo 選擇您偏好的啟動模式：
echo.
echo 🎛️  選項 1: 直接啟動模式 (Direct Python)
echo ==========================
echo 1. simple     - Direct Python startup (no Docker)
echo 2. web        - Web interface (direct Python)
echo 3. demo       - Demo trading mode
echo.
echo 🐳 選項 2: Docker 容器模式
echo ==========================
echo 4. docker-dev - Docker development mode
echo 5. docker-prod - Docker production mode
echo 6. docker-optimized - Docker optimized mode
echo.
echo 🛠️  選項 3: 系統工具
echo ====================
echo 7. diagnose   - System diagnostics
echo 8. quit       - Exit launcher
echo.

set /p "choice=Enter your choice (1-8): "

if "%choice%"=="1" goto simple
if "%choice%"=="2" goto web
if "%choice%"=="3" goto demo
if "%choice%"=="4" goto docker_dev
if "%choice%"=="5" goto docker_prod
if "%choice%"=="6" goto docker_optimized
if "%choice%"=="7" goto diagnose
if "%choice%"=="8" goto quit
if "%choice%"=="simple" goto simple
if "%choice%"=="web" goto web
if "%choice%"=="demo" goto demo
if "%choice%"=="docker-dev" goto docker_dev
if "%choice%"=="docker-prod" goto docker_prod
if "%choice%"=="docker-optimized" goto docker_optimized
if "%choice%"=="diagnose" goto diagnose
if "%choice%"=="quit" goto quit

echo Invalid choice. Please try again.
echo 無效選擇。請重試。
pause
goto start

:simple
echo.
echo 🚀 Starting AIFX in Simple Mode (Direct Python)...
echo.
if not exist "src\main\python\web_interface.py" (
    echo ❌ Error: web_interface.py not found
    echo Please run this script from the AIFX project directory
    pause
    exit /b 1
)

python --version 2>nul
if errorlevel 1 (
    echo ❌ Python is not installed or not accessible
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo Creating necessary directories...
if not exist logs mkdir logs
if not exist "data\cache" mkdir data\cache

set PYTHONPATH=%CD%
echo Starting AIFX Web Interface...
python -m src.main.python.web_interface
goto end

:web
echo.
echo 🌐 Starting AIFX Web Interface...
echo.
goto simple

:demo
echo.
echo 🎮 Starting AIFX Demo Trading Mode...
echo.
if not exist "run_trading_demo.py" (
    echo ❌ Error: run_trading_demo.py not found
    pause
    exit /b 1
)

python --version 2>nul
if errorlevel 1 (
    echo ❌ Python is not installed or not accessible
    pause
    exit /b 1
)

pip install -r requirements.txt
set PYTHONPATH=%CD%
python run_trading_demo.py --mode demo
goto end

:docker_dev
echo.
echo 🐳 Starting AIFX with Docker (Development Mode)...
echo.
docker --version 2>nul
if errorlevel 1 (
    echo ❌ Docker is not installed or not accessible
    pause
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running
    pause
    exit /b 1
)

docker-compose down --remove-orphans >nul 2>&1
echo Building and starting services...
docker-compose up -d --build
goto end

:docker_prod
echo.
echo 🏭 Starting AIFX with Docker (Production Mode)...
echo.
docker --version 2>nul
if errorlevel 1 (
    echo ❌ Docker is not installed or not accessible
    pause
    exit /b 1
)

docker-compose -f docker-compose.prod.yml down --remove-orphans >nul 2>&1
echo Building and starting production services...
docker-compose -f docker-compose.prod.yml up -d --build
goto end

:docker_optimized
echo.
echo ⚡ Starting AIFX with Docker (Optimized Mode)...
echo.
docker --version 2>nul
if errorlevel 1 (
    echo ❌ Docker is not installed or not accessible
    pause
    exit /b 1
)

docker-compose -f docker-compose.optimized.yml down --remove-orphans >nul 2>&1
echo Building and starting optimized services...
docker-compose -f docker-compose.optimized.yml up -d --build
goto end

:diagnose
echo.
echo 🛠️  Running AIFX System Diagnostics...
echo.
echo Checking Python installation...
python --version
echo.
echo Checking required directories...
if exist "src\main\python" (echo ✅ Source directory found) else (echo ❌ Source directory missing)
if exist "requirements.txt" (echo ✅ Requirements file found) else (echo ❌ Requirements file missing)
echo.
echo Checking Docker...
docker --version 2>nul
if errorlevel 1 (echo ❌ Docker not found) else (echo ✅ Docker found)
echo.
echo Checking Python packages...
python -c "import pandas, numpy, sklearn; print('✅ Core packages available')" 2>nul || echo "❌ Some packages missing"
echo.
echo Diagnostics complete!
pause
goto start

:quit
echo.
echo 👋 Goodbye! 再見！
echo.
exit /b 0

:end
echo.
echo 🎉 AIFX launched successfully!
echo Access your trading interface at: http://localhost:8080
echo.
echo Press any key to exit...
pause