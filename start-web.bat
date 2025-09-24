@echo off
REM AIFX Simplified Web Trading Signals - Windows Start Script
REM AIFX 簡化網頁交易信號 - Windows啟動腳本

echo.
echo 🚀 AIFX Trading Signals - Quick Start
echo ====================================
echo.

REM Colors (limited in CMD)
set GREEN=[92m
set RED=[91m
set YELLOW=[93m
set NC=[0m

REM Check if Docker is installed and running
echo Checking Docker installation...
docker --version >nul 2>&1
if errorlevel 1 (
    echo %RED%❌ Docker is not installed or not running%NC%
    echo Please install Docker Desktop and make sure it's running
    echo Visit: https://docs.docker.com/get-docker/
    echo.
    pause
    exit /b 1
)

echo %GREEN%✅ Docker is available%NC%

REM Check if Docker Compose is available
docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo %RED%❌ Docker Compose is not available%NC%
    echo Please make sure Docker Desktop is fully started
    echo.
    pause
    exit /b 1
)

echo %GREEN%✅ Docker Compose is available%NC%
echo.

REM Create necessary directories
echo Creating necessary directories...
if not exist logs mkdir logs
if not exist data\cache mkdir data\cache
echo %GREEN%✅ Directories created%NC%

REM Stop any existing containers
echo Stopping any existing containers...
docker-compose -f docker-compose.web.yml down --remove-orphans >nul 2>&1

REM Build and start services
echo.
echo %YELLOW%⚡ Building and starting AIFX Trading Signals...%NC%
echo This may take a few minutes on first run...
echo.

docker-compose -f docker-compose.web.yml up -d --build

if errorlevel 1 (
    echo.
    echo %RED%❌ Failed to start services%NC%
    echo Please check the error messages above
    echo.
    pause
    exit /b 1
)

REM Wait for services to be ready
echo.
echo %YELLOW%⏳ Waiting for services to be ready...%NC%
timeout /t 15 /nobreak >nul

REM Check if services are running
echo.
echo Checking service status...
docker-compose -f docker-compose.web.yml ps

echo.
echo %GREEN%🎉 AIFX Trading Signals is now running!%NC%
echo.
echo 📊 Access your trading signals dashboard at:
echo    🌐 Main Interface: http://localhost:8080/
echo    ❤️  Health Check:  http://localhost:8080/api/health
echo    📊 Signals API:    http://localhost:8080/api/signals
echo.
echo 📋 Useful commands:
echo    View logs:    docker-compose -f docker-compose.web.yml logs -f
echo    Stop service: docker-compose -f docker-compose.web.yml down
echo    Restart:      docker-compose -f docker-compose.web.yml restart
echo.

REM Try to open browser (optional)
set /p "OPEN=Open browser automatically? (y/N): "
if /i "%OPEN%"=="y" (
    start http://localhost:8080
)

echo.
echo %GREEN%✅ Setup complete! Your 24/7 trading signals are now active. 🚀%NC%
echo Press any key to exit...
pause >nul