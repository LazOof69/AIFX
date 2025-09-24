@echo off
chcp 65001 >nul 2>&1
REM AIFX Simplified Web Trading Signals - Windows Start Script
REM AIFX 簡化網頁交易信號 - Windows啟動腳本

title AIFX Trading Signals Startup

echo.
echo 🚀 AIFX Trading Signals - Quick Start
echo ====================================
echo.

REM Check current directory
echo Current directory: %CD%
echo.

REM Check if we're in the right directory
if not exist "docker-compose.web.yml" (
    echo ❌ Error: docker-compose.web.yml not found
    echo Please run this script from the AIFX project directory
    echo Expected path: C:\Users\butte\OneDrive\桌面\AIFX_CLAUDE
    echo Current path: %CD%
    echo.
    pause
    exit /b 1
)

echo ✅ Found docker-compose.web.yml
echo.

REM Check if Docker is installed and running
echo Checking Docker installation...
docker --version 2>nul
if errorlevel 1 (
    echo ❌ Docker is not installed or not accessible
    echo.
    echo Please:
    echo 1. Install Docker Desktop from https://docs.docker.com/get-docker/
    echo 2. Make sure Docker Desktop is running
    echo 3. Restart this script
    echo.
    pause
    exit /b 1
)

echo ✅ Docker is installed
echo.

REM Check if Docker is running
echo Checking if Docker is running...
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running
    echo.
    echo Please:
    echo 1. Start Docker Desktop
    echo 2. Wait for it to fully start
    echo 3. Restart this script
    echo.
    pause
    exit /b 1
)

echo ✅ Docker is running
echo.

REM Check if Docker Compose is available
echo Checking Docker Compose...
docker-compose --version 2>nul
if errorlevel 1 (
    echo ❌ Docker Compose is not available
    echo Please make sure Docker Desktop is fully started
    echo.
    pause
    exit /b 1
)

echo ✅ Docker Compose is available
echo.

REM Create necessary directories
echo Creating necessary directories...
if not exist logs mkdir logs
if not exist data\cache mkdir data\cache
echo %GREEN%✅ Directories created%NC%

REM Stop any existing containers first
echo Stopping any existing containers...
docker-compose -f docker-compose.web.yml down --remove-orphans >nul 2>&1
echo ✅ Previous containers stopped
echo.

REM Build and start services
echo ⚡ Building and starting AIFX Trading Signals...
echo This may take a few minutes on first run...
echo Please wait...
echo.

REM Show the actual command being run
echo Running: docker-compose -f docker-compose.web.yml up -d --build
echo.

docker-compose -f docker-compose.web.yml up -d --build

if errorlevel 1 (
    echo.
    echo ❌ Failed to start services
    echo.
    echo Common solutions:
    echo 1. Make sure Docker Desktop is fully started
    echo 2. Check if port 8080 is already in use
    echo 3. Try running: docker system prune -f
    echo 4. Restart Docker Desktop
    echo.
    echo Press any key to see detailed logs...
    pause >nul
    echo.
    echo === DETAILED LOGS ===
    docker-compose -f docker-compose.web.yml logs
    echo.
    pause
    exit /b 1
)

REM Wait for services to be ready
echo.
echo ⏳ Waiting for services to initialize...
echo Please wait 30 seconds...

REM Count down timer
for /l %%i in (30,-1,1) do (
    echo Waiting... %%i seconds remaining
    timeout /t 1 /nobreak >nul
)

echo.
echo Checking service status...
docker-compose -f docker-compose.web.yml ps

echo.
echo 🎉 AIFX Trading Signals should now be running!
echo.
echo 📊 Access your trading signals dashboard at:
echo    🌐 Main Interface: http://localhost:8080/
echo    ❤️  Health Check:  http://localhost:8080/api/health
echo    📊 Signals API:    http://localhost:8080/api/signals
echo.
echo 📋 Useful commands for later:
echo    View logs:    docker-compose -f docker-compose.web.yml logs -f
echo    Stop service: docker-compose -f docker-compose.web.yml down
echo    Restart:      docker-compose -f docker-compose.web.yml restart
echo.

REM Test the health endpoint
echo Testing connection...
curl -s http://localhost:8080/api/health >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Service might still be starting up
    echo If you can't access http://localhost:8080 after a few minutes:
    echo 1. Check logs: docker-compose -f docker-compose.web.yml logs
    echo 2. Try restarting: docker-compose -f docker-compose.web.yml restart
) else (
    echo ✅ Service is responding correctly!
)

echo.
REM Try to open browser (optional)
set /p "OPEN=Open browser automatically? (Y/n): "
if /i not "%OPEN%"=="n" (
    echo Opening browser...
    start http://localhost:8080
)

echo.
echo ✅ Setup complete!
echo.
echo If the browser doesn't open, manually visit: http://localhost:8080
echo.
echo Press any key to exit (the service will keep running)...
pause >nul