@echo off
chcp 65001 >nul 2>&1
title AIFX Cloud Deployment - Windows
REM AIFX 雲端部署 - Windows

echo.
echo 🚀 AIFX Cloud Deployment Script (Windows)
echo AIFX 雲端部署腳本 (Windows)
echo ===========================================
echo.

REM Check if Docker is installed
echo 📦 Checking Docker Installation... | 檢查 Docker 安裝...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not installed or not accessible
    echo ❌ Docker 未安裝或無法訪問
    echo.
    echo Please install Docker Desktop:
    echo 請安裝 Docker Desktop：
    echo https://docs.docker.com/get-docker/
    echo.
    pause
    exit /b 1
)

docker-compose --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker Compose is not installed
    echo ❌ Docker Compose 未安裝
    pause
    exit /b 1
)

echo ✅ Docker and Docker Compose are available
echo ✅ Docker 和 Docker Compose 可用
echo.

REM Check if Docker is running
echo 🔍 Checking if Docker is running... | 檢查 Docker 是否運行...
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running
    echo ❌ Docker 未運行
    echo.
    echo Please start Docker Desktop first
    echo 請先啟動 Docker Desktop
    echo.
    pause
    exit /b 1
)

echo ✅ Docker is running
echo ✅ Docker 正在運行
echo.

REM Setup environment file
echo ⚙️  Setting up environment... | 設置環境...
if not exist ".env" (
    if exist ".env.cloud" (
        copy .env.cloud .env >nul
        echo ✅ Environment file created from template
        echo ✅ 從模板創建環境文件
    ) else (
        echo ❌ No environment template found
        echo ❌ 找不到環境模板
        pause
        exit /b 1
    )
) else (
    echo ✅ Environment file already exists
    echo ✅ 環境文件已存在
)
echo.

REM Stop existing containers
echo 🛑 Stopping existing containers... | 停止現有容器...
docker-compose -f docker-compose.cloud.yml down --remove-orphans >nul 2>&1
echo ✅ Existing containers stopped
echo ✅ 現有容器已停止
echo.

REM Build and start services
echo 🚀 Building and starting services... | 建置並啟動服務...
echo This may take a few minutes... | 這可能需要幾分鐘...
echo.

docker-compose -f docker-compose.cloud.yml up -d --build

if errorlevel 1 (
    echo.
    echo ❌ Deployment failed! | 部署失敗！
    echo.
    echo Common solutions | 常見解決方案：
    echo 1. Make sure Docker Desktop is running | 確保 Docker Desktop 正在運行
    echo 2. Check if port 8080 is available | 檢查端口 8080 是否可用
    echo 3. Try: docker system prune -f | 嘗試：docker system prune -f
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ Services started successfully! | 服務啟動成功！
echo.

REM Health check
echo 🏥 Performing health check... | 執行健康檢查...
echo Waiting for services to be ready... | 等待服務準備就緒...
echo.

REM Wait 30 seconds for services to start
timeout /t 30 /nobreak >nul

REM Check container status
docker-compose -f docker-compose.cloud.yml ps

echo.
echo 🎉 AIFX Cloud Deployment Completed! | AIFX 雲端部署完成！
echo ==========================================
echo.

REM Read port from .env file
set AIFX_WEB_PORT=8080
for /f "tokens=2 delims==" %%i in ('findstr "AIFX_WEB_PORT" .env 2^>nul') do set AIFX_WEB_PORT=%%i

echo 📱 Access Information | 訪問信息：
echo    🌐 Web Interface | 網頁介面: http://localhost:%AIFX_WEB_PORT%
echo    ❤️  Health Check | 健康檢查: http://localhost:%AIFX_WEB_PORT%/api/health
echo    📊 API Docs | API文件: http://localhost:%AIFX_WEB_PORT%/docs
echo.

echo 🛠️  Management Commands | 管理命令：
echo    View logs | 查看日誌:
echo      docker-compose -f docker-compose.cloud.yml logs -f
echo.
echo    Stop service | 停止服務:
echo      docker-compose -f docker-compose.cloud.yml down
echo.
echo    Restart service | 重新啟動服務:
echo      docker-compose -f docker-compose.cloud.yml restart
echo.

REM Test the health endpoint
echo 🔍 Testing connection... | 測試連接...
curl -s http://localhost:%AIFX_WEB_PORT%/api/health >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Service might still be starting up | 服務可能仍在啟動中
    echo If you can't access the service after 2-3 minutes:
    echo 如果 2-3 分鐘後無法訪問服務：
    echo 1. Check logs: docker-compose -f docker-compose.cloud.yml logs
    echo 1. 檢查日誌: docker-compose -f docker-compose.cloud.yml logs
) else (
    echo ✅ Service is responding correctly! | 服務正常響應！
)

echo.
set /p "OPEN=Open web interface in browser? (Y/n) | 在瀏覽器中打開網頁介面？(Y/n): "
if /i not "%OPEN%"=="n" (
    echo Opening browser... | 打開瀏覽器...
    start http://localhost:%AIFX_WEB_PORT%
)

echo.
echo ✅ Deployment completed successfully! | 部署成功完成！
echo.
echo Press any key to exit... | 按任意鍵退出...
pause >nul