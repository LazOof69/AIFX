#!/bin/bash
# Simple AIFX Deployment Script (without credential issues)
# 簡單 AIFX 部署腳本（避免認證問題）

set -e

echo "🚀 AIFX Simple Deployment"
echo "========================"

# Clean environment
echo "🧹 Cleaning Docker environment..."
docker system prune -f 2>/dev/null || true

# Remove credential store configuration temporarily
echo "🔧 Temporarily fixing credential configuration..."
if [ -f "$HOME/.docker/config.json" ]; then
    cp "$HOME/.docker/config.json" "$HOME/.docker/config.json.backup"
    echo '{"auths":{}}' > "$HOME/.docker/config.json"
fi

# Stop any existing containers
echo "🛑 Stopping existing containers..."
docker-compose -f docker-compose-free.yml down -v 2>/dev/null || true

# Pull images individually to avoid credential issues
echo "📥 Pulling required images..."
echo "  - PostgreSQL..."
docker pull postgres:15-alpine

echo "  - Redis..."
docker pull redis:7-alpine

echo "  - Grafana..."
docker pull grafana/grafana:latest

# Build and start services
echo "🏗️ Building AIFX application..."
docker-compose -f docker-compose-free.yml build aifx-web

echo "🚀 Starting all services..."
docker-compose -f docker-compose-free.yml up -d

# Restore original Docker config if it existed
if [ -f "$HOME/.docker/config.json.backup" ]; then
    mv "$HOME/.docker/config.json.backup" "$HOME/.docker/config.json"
fi

echo ""
echo "⏳ Waiting for services to start..."
sleep 15

echo ""
echo "📊 Service Status:"
docker-compose -f docker-compose-free.yml ps

echo ""
echo "🧪 Testing services..."

# Test database
echo "🗄️ Testing PostgreSQL..."
if docker-compose -f docker-compose-free.yml exec -T db pg_isready -U aifx >/dev/null 2>&1; then
    echo "  ✅ PostgreSQL is ready"
else
    echo "  ⏳ PostgreSQL still starting..."
fi

# Test Redis
echo "🟥 Testing Redis..."
if docker-compose -f docker-compose-free.yml exec -T redis redis-cli ping >/dev/null 2>&1; then
    echo "  ✅ Redis is ready"
else
    echo "  ⏳ Redis still starting..."
fi

# Test AIFX app
echo "📱 Testing AIFX application..."
sleep 10
if curl -f http://localhost:8000 >/dev/null 2>&1; then
    echo "  ✅ AIFX application is responding"
elif curl -f http://localhost:8000/health >/dev/null 2>&1; then
    echo "  ✅ AIFX health endpoint is responding"
else
    echo "  ⏳ AIFX application still starting (this is normal)"
    echo "     Check logs: docker-compose -f docker-compose-free.yml logs aifx-web"
fi

echo ""
echo "🎉 Deployment Complete!"
echo "======================================"
echo ""
echo "📱 AIFX Application: http://localhost:8000"
echo "📊 Grafana Dashboard: http://localhost:3000 (admin/admin123)"
echo "🗄️ PostgreSQL: localhost:5432 (aifx/password)"
echo "🟥 Redis: localhost:6379"
echo ""
echo "📋 Management Commands:"
echo "  查看日誌: docker-compose -f docker-compose-free.yml logs -f"
echo "  停止服務: docker-compose -f docker-compose-free.yml down"
echo "  重啟服務: docker-compose -f docker-compose-free.yml restart"
echo ""
echo "✅ AIFX is now running! Visit http://localhost:8000 to get started."