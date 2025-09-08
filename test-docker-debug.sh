#!/bin/bash
# AIFX Docker Debug Script - 調試版本
# 這個版本會顯示完整錯誤信息且不會自動關閉

echo "🔧 AIFX Docker Debug Test"
echo "=========================="

echo "📍 Current directory: $(pwd)"
echo "👤 Current user: $(whoami)"
echo ""

# Check Docker installation
echo "🐳 Checking Docker installation..."
if command -v docker &> /dev/null; then
    echo "✅ Docker command found: $(which docker)"
    echo "📦 Docker version: $(docker --version)"
else
    echo "❌ Docker command not found!"
    echo "請確保 Docker Desktop 已安裝"
    read -p "按任意鍵繼續..."
    exit 1
fi

echo ""
echo "🔍 Checking Docker daemon status..."

# Test Docker connection with detailed output
if docker info &> /dev/null; then
    echo "✅ Docker daemon is running!"
    echo "📊 Docker system info:"
    docker info | head -10
else
    echo "❌ Docker daemon is NOT running!"
    echo ""
    echo "錯誤詳情："
    docker info 2>&1 | head -5
    echo ""
    echo "解決方法："
    echo "1. 👆 開啟 Docker Desktop (從開始選單)"
    echo "2. ⏳ 等待 Docker 啟動 (系統匣圖示變綠色)"
    echo "3. 🔄 重新執行此腳本"
    echo ""
    echo "Docker Desktop 狀態檢查："
    echo "- Windows: 檢查系統匣是否有 Docker 圖示"
    echo "- 圖示應該是綠色 (表示運行中)"
    echo "- 如果是灰色或紅色，請點擊啟動"
    echo ""
    read -p "按 Enter 鍵結束..."
    exit 1
fi

echo ""
echo "🧪 Testing Docker Compose..."
if docker-compose --version &> /dev/null; then
    echo "✅ Docker Compose available: $(docker-compose --version)"
else
    echo "❌ Docker Compose not available"
    read -p "按 Enter 鍵結束..."
    exit 1
fi

echo ""
echo "📁 Checking required files..."
required_files=("Dockerfile" "docker-compose-free.yml" ".env")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file missing"
    fi
done

echo ""
echo "🚀 All checks passed! Docker is ready for deployment."
echo "🚀 所有檢查通過！Docker 準備就緒可以部署。"
echo ""
echo "執行部署命令："
echo "docker-compose -f docker-compose-free.yml up --build -d"
echo ""
read -p "按 Enter 鍵開始部署，或 Ctrl+C 取消..."

# If we get here, Docker is running, so let's try deployment
echo ""
echo "🏗️ Starting deployment..."
echo "🏗️ 開始部署..."

# Clean previous containers
echo "🧹 Cleaning previous containers..."
docker-compose -f docker-compose-free.yml down -v 2>/dev/null || true

# Start deployment
echo "🚀 Building and starting services..."
if docker-compose -f docker-compose-free.yml up --build -d; then
    echo ""
    echo "✅ Deployment started successfully!"
    echo "✅ 部署成功啟動！"
    echo ""
    echo "服務狀態："
    docker-compose -f docker-compose-free.yml ps
    echo ""
    echo "可用服務："
    echo "📱 AIFX App: http://localhost:8000"
    echo "📊 Grafana: http://localhost:3000"
    echo "🗄️ PostgreSQL: localhost:5432"
    echo ""
    echo "查看日誌: docker-compose -f docker-compose-free.yml logs -f"
    echo "停止服務: docker-compose -f docker-compose-free.yml down"
else
    echo ""
    echo "❌ Deployment failed!"
    echo "❌ 部署失敗！"
    echo ""
    echo "查看錯誤日誌："
    docker-compose -f docker-compose-free.yml logs
fi

echo ""
read -p "按 Enter 鍵結束..."