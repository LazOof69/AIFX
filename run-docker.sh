#!/bin/bash
# AIFX Docker Run Script
# AIFX Docker 運行腳本

set -e

echo "🐳 AIFX Docker Deployment Script"
echo "🐳 AIFX Docker 部署腳本"
echo "=================================="

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker is not running!"
    echo "❌ Docker 沒有運行！"
    echo ""
    echo "Please start Docker Desktop first:"
    echo "請先啟動 Docker Desktop："
    echo "1. Open Docker Desktop from Start Menu"
    echo "2. Wait for Docker to start (green icon in system tray)"
    echo "3. Run this script again"
    echo ""
    exit 1
fi

echo "✅ Docker is running"
echo "✅ Docker 正在運行"

# Choose deployment option
echo ""
echo "Choose deployment option | 選擇部署選項:"
echo "1. Simple deployment (app + database + cache) | 簡單部署 (應用 + 資料庫 + 緩存)"
echo "2. Full development stack (all services) | 完整開發堆棧 (所有服務)"
echo "3. Testing environment | 測試環境"

read -p "Enter your choice [1-3]: " choice

case $choice in
    1)
        echo "🚀 Starting simple deployment..."
        echo "🚀 啟動簡單部署..."
        docker-compose -f docker-compose-free.yml up --build -d
        ;;
    2)
        echo "🚀 Starting full development stack..."
        echo "🚀 啟動完整開發堆棧..."
        docker-compose up --build -d
        ;;
    3)
        echo "🧪 Starting testing environment..."
        echo "🧪 啟動測試環境..."
        docker build --target testing -t aifx-testing .
        docker run --rm -v $(pwd):/workspace aifx-testing
        ;;
    *)
        echo "❌ Invalid choice"
        echo "❌ 無效選擇"
        exit 1
        ;;
esac

if [ $choice -ne 3 ]; then
    echo ""
    echo "🎉 Deployment completed!"
    echo "🎉 部署完成！"
    echo ""
    echo "Services available at | 服務可用地址:"
    echo "📱 AIFX App: http://localhost:8000"
    echo "📊 Grafana: http://localhost:3000 (admin/admin123 or aifx_grafana_password)"
    
    if [ $choice -eq 2 ]; then
        echo "🔍 Kibana: http://localhost:5601"
        echo "📈 Prometheus: http://localhost:9090"
        echo "🗄️ PostgreSQL: localhost:5432"
        echo "🟥 Redis: localhost:6379"
        echo "🍃 MongoDB: localhost:27017"
        echo "🔍 Elasticsearch: http://localhost:9200"
    else
        echo "🗄️ PostgreSQL: localhost:5432"
        echo "🟥 Redis: localhost:6379"
    fi
    
    echo ""
    echo "To stop services | 停止服務:"
    if [ $choice -eq 1 ]; then
        echo "docker-compose -f docker-compose-free.yml down"
    else
        echo "docker-compose down"
    fi
    
    echo ""
    echo "To view logs | 查看日誌:"
    if [ $choice -eq 1 ]; then
        echo "docker-compose -f docker-compose-free.yml logs -f"
    else
        echo "docker-compose logs -f"
    fi
fi