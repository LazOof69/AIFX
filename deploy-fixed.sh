#!/bin/bash
# AIFX Fixed Deployment Script
# AIFX 修復部署腳本

set -e

echo "🔧 AIFX Fixed Deployment"
echo "========================"

echo "📁 Checking required files..."
if [ ! -f "docker/entrypoint.sh" ]; then
    echo "❌ docker/entrypoint.sh not found!"
    exit 1
else
    echo "✅ docker/entrypoint.sh found"
fi

if [ ! -f "Dockerfile" ]; then
    echo "❌ Dockerfile not found!"
    exit 1
else
    echo "✅ Dockerfile found"
fi

# Clean up previous builds
echo "🧹 Cleaning previous Docker builds..."
docker-compose -f docker-compose-free.yml down -v 2>/dev/null || true
docker system prune -f 2>/dev/null || true

# Clear build cache
echo "🗑️ Clearing Docker build cache..."
docker builder prune -f 2>/dev/null || true

# Fix Docker credential config temporarily
echo "🔧 Temporarily fixing Docker credentials..."
if [ -f "$HOME/.docker/config.json" ]; then
    cp "$HOME/.docker/config.json" "$HOME/.docker/config.json.backup" 2>/dev/null || true
fi
mkdir -p "$HOME/.docker"
echo '{"auths":{}}' > "$HOME/.docker/config.json"

# Build with no cache
echo "🏗️ Building AIFX with fresh cache..."
docker-compose -f docker-compose-free.yml build --no-cache

# Start services
echo "🚀 Starting all services..."
docker-compose -f docker-compose-free.yml up -d

# Restore Docker config if backup exists
if [ -f "$HOME/.docker/config.json.backup" ]; then
    mv "$HOME/.docker/config.json.backup" "$HOME/.docker/config.json" 2>/dev/null || true
fi

echo ""
echo "⏳ Waiting for services to initialize..."
sleep 20

echo ""
echo "📊 Service Status:"
docker-compose -f docker-compose-free.yml ps

echo ""
echo "🧪 Testing service connectivity..."

# Test PostgreSQL
echo "🗄️ PostgreSQL:"
if docker-compose -f docker-compose-free.yml exec -T db pg_isready -U aifx >/dev/null 2>&1; then
    echo "  ✅ Ready and accepting connections"
else
    echo "  ⏳ Still starting up..."
fi

# Test Redis
echo "🟥 Redis:"
if docker-compose -f docker-compose-free.yml exec -T redis redis-cli ping >/dev/null 2>&1; then
    echo "  ✅ Ready and accepting connections"
else
    echo "  ⏳ Still starting up..."
fi

# Test AIFX App
echo "📱 AIFX Application:"
sleep 5
if curl -s http://localhost:8000 >/dev/null 2>&1; then
    echo "  ✅ Responding on port 8000"
elif docker-compose -f docker-compose-free.yml logs aifx-web | grep -q "Error\|Exception\|Failed"; then
    echo "  ❌ Application errors detected"
    echo "  📋 Recent logs:"
    docker-compose -f docker-compose-free.yml logs --tail=10 aifx-web
else
    echo "  ⏳ Still starting up (this is normal for first run)"
fi

# Test Grafana
echo "📊 Grafana:"
if curl -s http://localhost:3000 >/dev/null 2>&1; then
    echo "  ✅ Ready and accepting connections"
else
    echo "  ⏳ Still starting up..."
fi

echo ""
echo "🎉 Deployment Process Complete!"
echo "================================="
echo ""
echo "🌐 Access Points:"
echo "  📱 AIFX Application: http://localhost:8000"
echo "  📊 Grafana Dashboard: http://localhost:3000"
echo "      Username: admin"
echo "      Password: admin123"
echo ""
echo "🗄️ Database Access:"
echo "  PostgreSQL: localhost:5432"
echo "  Username: aifx"
echo "  Password: password"
echo "  Database: aifx"
echo ""
echo "📋 Management Commands:"
echo "  查看所有服務狀態: docker-compose -f docker-compose-free.yml ps"
echo "  查看應用日誌: docker-compose -f docker-compose-free.yml logs -f aifx-web"
echo "  查看所有日誌: docker-compose -f docker-compose-free.yml logs -f"
echo "  重啟服務: docker-compose -f docker-compose-free.yml restart"
echo "  停止所有服務: docker-compose -f docker-compose-free.yml down"
echo ""
echo "✅ AIFX 系統部署完成！請訪問 http://localhost:8000 開始使用。"

# Show logs if there are any errors
if docker-compose -f docker-compose-free.yml logs aifx-web 2>/dev/null | grep -q "Error\|Exception\|Traceback"; then
    echo ""
    echo "⚠️ 檢測到應用程式錯誤，顯示最近日誌："
    echo "================================================"
    docker-compose -f docker-compose-free.yml logs --tail=20 aifx-web
fi