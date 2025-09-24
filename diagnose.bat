@echo off
chcp 65001 >nul 2>&1
title AIFX System Diagnosis

echo.
echo 🔧 AIFX System Diagnosis Tool
echo =============================
echo.

REM Check current directory
echo 1. Checking current directory...
echo    Current path: %CD%

if exist "docker-compose.web.yml" (
    echo    ✅ docker-compose.web.yml found
) else (
    echo    ❌ docker-compose.web.yml NOT found
)

if exist "src\main\python\web_interface.py" (
    echo    ✅ web_interface.py found
) else (
    echo    ❌ web_interface.py NOT found
)

echo.

REM Check Python
echo 2. Checking Python...
python --version 2>nul
if errorlevel 1 (
    echo    ❌ Python is NOT available
    echo    Please install Python from https://python.org
) else (
    echo    ✅ Python is available
)

echo.

REM Check Docker
echo 3. Checking Docker...
docker --version 2>nul
if errorlevel 1 (
    echo    ❌ Docker is NOT installed
    echo    Install from: https://docs.docker.com/get-docker/
) else (
    echo    ✅ Docker is installed

    REM Check if Docker is running
    docker info >nul 2>&1
    if errorlevel 1 (
        echo    ❌ Docker is NOT running
        echo    Please start Docker Desktop
    ) else (
        echo    ✅ Docker is running
    )
)

echo.

REM Check Docker Compose
echo 4. Checking Docker Compose...
docker-compose --version 2>nul
if errorlevel 1 (
    echo    ❌ Docker Compose is NOT available
) else (
    echo    ✅ Docker Compose is available
)

echo.

REM Check port availability
echo 5. Checking port 8080...
netstat -an | find "8080" >nul 2>&1
if errorlevel 1 (
    echo    ✅ Port 8080 appears to be free
) else (
    echo    ⚠️  Port 8080 is already in use
    echo    Something else might be running on this port
)

echo.

REM Check if containers are already running
echo 6. Checking existing containers...
docker ps -q --filter "name=aifx" >nul 2>&1
if errorlevel 1 (
    echo    ✅ No AIFX containers running
) else (
    echo    ⚠️  AIFX containers might be running
    echo    Current containers:
    docker ps --filter "name=aifx"
)

echo.

REM Summary and recommendations
echo 📋 RECOMMENDATIONS:
echo ==================

if not exist "docker-compose.web.yml" (
    echo ❌ You're not in the AIFX directory!
    echo    Navigate to: C:\Users\butte\OneDrive\桌面\AIFX_CLAUDE
    echo.
)

docker info >nul 2>&1
if errorlevel 1 (
    echo 🐳 Docker Issues:
    echo    1. Make sure Docker Desktop is installed and running
    echo    2. Wait for Docker to fully start up
    echo    3. Try restarting Docker Desktop
    echo.
)

python --version >nul 2>&1
if errorlevel 1 (
    echo 🐍 Python Issues:
    echo    1. Install Python from https://python.org
    echo    2. Make sure to check "Add Python to PATH" during installation
    echo.
)

echo 🚀 STARTUP OPTIONS:
echo ===================
echo Option 1 - Docker (Recommended):
echo    1. Make sure Docker Desktop is running
echo    2. Double-click start-web.bat
echo.
echo Option 2 - Direct Python (Alternative):
echo    1. Double-click start-simple.bat
echo.
echo Option 3 - Manual Docker:
echo    1. Open Command Prompt in this directory
echo    2. Run: docker-compose -f docker-compose.web.yml up -d
echo.

echo Press any key to exit...
pause >nul