#!/bin/bash
# AIFX Docker Fixed Deployment Test Script
# AIFX Docker 修復部署測試腳本

set -e

echo "🔧 AIFX Docker Fixed Deployment"
echo "🔧 AIFX Docker 修復部署"
echo "================================"

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

# Clean any previous containers
echo "🧹 Cleaning previous containers..."
echo "🧹 清理之前的容器..."
docker-compose -f docker-compose-free.yml down -v 2>/dev/null || true

# Clean Docker cache
echo "🗑️ Cleaning Docker cache..."
echo "🗑️ 清理 Docker 緩存..."
docker system prune -f

# Build and start services
echo "🚀 Building and starting AIFX services..."
echo "🚀 構建並啟動 AIFX 服務..."
docker-compose -f docker-compose-free.yml up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
echo "⏳ 等待服務準備就緒..."
sleep 30

# Check service status
echo "📊 Service Status:"
docker-compose -f docker-compose-free.yml ps

# Test basic connectivity
echo ""
echo "🧪 Testing basic connectivity..."
echo "🧪 測試基本連接..."

# Test PostgreSQL
if docker-compose -f docker-compose-free.yml exec -T db pg_isready -U aifx; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL connection failed"
fi

# Test Redis
if docker-compose -f docker-compose-free.yml exec -T redis redis-cli ping; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis connection failed"
fi

# Test AIFX app (basic HTTP check)
sleep 5
if curl -f http://localhost:8000/health 2>/dev/null; then
    echo "✅ AIFX application is responding"
elif curl -f http://localhost:8000 2>/dev/null; then
    echo "✅ AIFX application is running (no health endpoint)"
else
    echo "⚠️ AIFX application may still be starting..."
    echo "   Check logs: docker-compose -f docker-compose-free.yml logs aifx-web"
fi

echo ""
echo "🎉 Deployment completed!"
echo "🎉 部署完成！"
echo ""
echo "Services available at:"
echo "📱 AIFX App: http://localhost:8000"
echo "📊 Grafana: http://localhost:3000 (admin/admin123)"
echo "🗄️ PostgreSQL: localhost:5432 (aifx/password)"
echo "🟥 Redis: localhost:6379"
echo ""
echo "To view logs: docker-compose -f docker-compose-free.yml logs -f"
echo "To stop: docker-compose -f docker-compose-free.yml down"
echo ""
echo "Next steps:"
echo "1. Visit http://localhost:8000 to access AIFX"
echo "2. Visit http://localhost:3000 for monitoring (Grafana)"
echo "3. Run tests with: docker-compose -f docker-compose-free.yml exec aifx-web python test_phase1_complete.py"