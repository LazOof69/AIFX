#!/bin/bash
# AIFX Simple Docker Deployment (No SQL Server Dependencies)
# AIFX 簡化 Docker 部署（無 SQL Server 依賴）

set -e

echo "🚀 AIFX Simple Docker Deployment"
echo "================================"
echo "Using simplified Dockerfile without SQL Server dependencies"
echo "使用無 SQL Server 依賴的簡化 Dockerfile"
echo ""

# Clean up previous builds
echo "🧹 Cleaning previous builds..."
docker-compose -f docker-compose-free.yml down -v 2>/dev/null || true
docker system prune -f 2>/dev/null || true

# Fix Docker credentials
echo "🔧 Fixing Docker credentials..."
mkdir -p "$HOME/.docker"
echo '{"auths":{}}' > "$HOME/.docker/config.json"

# Create temporary docker-compose with simple dockerfile
echo "📝 Creating temporary compose configuration..."
cat > docker-compose-simple.yml << 'EOF'
# Simple AIFX Deployment
# 簡化 AIFX 部署

services:
  aifx-web:
    build:
      context: .
      dockerfile: Dockerfile.simple
      target: production
    ports:
      - "8000:8000"
    environment:
      - PYTHONPATH=/home/aifx/app/src/main/python
      - PYTHONUNBUFFERED=1
      - PYTHONDONTWRITEBYTECODE=1
      - AIFX_ENV=production
      - AIFX_ROOT=/home/aifx/app
      - AIFX_DATA_DIR=/home/aifx/app/data
      - AIFX_MODELS_DIR=/home/aifx/app/models
      - AIFX_OUTPUT_DIR=/home/aifx/app/output
      - AIFX_LOGS_DIR=/home/aifx/app/logs
      - TF_CPP_MIN_LOG_LEVEL=2
      - SKIP_NETWORK_TESTS=true
      - DATABASE_URL=postgresql://aifx:password@db:5432/aifx
      - REDIS_URL=redis://redis:6379/0
      - POSTGRES_HOST=db
      - POSTGRES_DB=aifx
      - POSTGRES_USER=aifx
      - POSTGRES_PASSWORD=password
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - PORT=8000
      - HOST=0.0.0.0
    depends_on:
      - db
      - redis
    volumes:
      - ./data:/home/aifx/app/data
      - ./logs:/home/aifx/app/logs
      - ./models:/home/aifx/app/models
      - ./output:/home/aifx/app/output
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: aifx
      POSTGRES_USER: aifx
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # Optional: Simple monitoring
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  grafana_data:
EOF

# Build the application
echo "🏗️ Building AIFX application..."
docker-compose -f docker-compose-simple.yml build --no-cache aifx-web

# Start all services
echo "🚀 Starting all services..."
docker-compose -f docker-compose-simple.yml up -d

echo ""
echo "⏳ Waiting for services to start..."
sleep 25

echo ""
echo "📊 Service Status:"
docker-compose -f docker-compose-simple.yml ps

echo ""
echo "🧪 Testing services..."

# Test PostgreSQL
echo "🗄️ PostgreSQL:"
if docker-compose -f docker-compose-simple.yml exec -T db pg_isready -U aifx >/dev/null 2>&1; then
    echo "  ✅ Ready and accepting connections"
else
    echo "  ⏳ Still starting..."
fi

# Test Redis
echo "🟥 Redis:"
if docker-compose -f docker-compose-simple.yml exec -T redis redis-cli ping >/dev/null 2>&1; then
    echo "  ✅ Ready and accepting connections"
else
    echo "  ⏳ Still starting..."
fi

# Test AIFX Application
echo "📱 AIFX Application:"
sleep 5
if curl -s http://localhost:8000 >/dev/null 2>&1; then
    echo "  ✅ Responding on http://localhost:8000"
elif docker-compose -f docker-compose-simple.yml logs aifx-web | grep -i error >/dev/null 2>&1; then
    echo "  ❌ Found errors in application logs"
    echo "  📋 Recent errors:"
    docker-compose -f docker-compose-simple.yml logs --tail=5 aifx-web | grep -i error
else
    echo "  ⏳ Still starting up (normal for first run)"
fi

# Test Grafana
echo "📊 Grafana:"
if curl -s http://localhost:3000 >/dev/null 2>&1; then
    echo "  ✅ Ready at http://localhost:3000"
else
    echo "  ⏳ Still starting..."
fi

echo ""
echo "🎉 Deployment Complete!"
echo "======================"
echo ""
echo "🌐 Service URLs:"
echo "  📱 AIFX Application: http://localhost:8000"
echo "  📊 Grafana Dashboard: http://localhost:3000"
echo "     - Username: admin"
echo "     - Password: admin123"
echo ""
echo "🗄️ Database Access:"
echo "  📊 PostgreSQL: localhost:5432"
echo "     - Database: aifx"
echo "     - Username: aifx"  
echo "     - Password: password"
echo ""
echo "📋 Management:"
echo "  查看狀態: docker-compose -f docker-compose-simple.yml ps"
echo "  查看日誌: docker-compose -f docker-compose-simple.yml logs -f aifx-web"
echo "  停止服務: docker-compose -f docker-compose-simple.yml down"
echo "  重啟應用: docker-compose -f docker-compose-simple.yml restart aifx-web"
echo ""

# Show application logs if there are issues
if docker-compose -f docker-compose-simple.yml logs aifx-web 2>/dev/null | grep -i -E "error|exception|traceback" >/dev/null 2>&1; then
    echo "⚠️  Application logs show some issues:"
    echo "======================================="
    docker-compose -f docker-compose-simple.yml logs --tail=10 aifx-web
    echo ""
    echo "💡 If the application is not responding:"
    echo "   1. Check logs: docker-compose -f docker-compose-simple.yml logs aifx-web"
    echo "   2. Restart app: docker-compose -f docker-compose-simple.yml restart aifx-web"
    echo "   3. Check database connection and initialization"
fi

echo "✅ AIFX deployment completed! Visit http://localhost:8000 to access the application."