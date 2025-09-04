#!/bin/bash
# AIFX - Production Entrypoint Script
# AIFX - 生產環境入口腳本

set -e

# ============================================================================
# Environment Setup | 環境設置
# ============================================================================
echo "🚀 Starting AIFX Production Environment..."
echo "🚀 啟動AIFX生產環境..."

# Set default environment if not provided | 設置默認環境（如果未提供）
export AIFX_ENV=${AIFX_ENV:-production}
export PYTHONPATH=${PYTHONPATH:-/home/aifx/app}

echo "📋 Environment: $AIFX_ENV"
echo "📋 Python Path: $PYTHONPATH"

# ============================================================================
# Pre-flight Checks | 啟動前檢查
# ============================================================================
echo "🔍 Performing pre-flight checks..."
echo "🔍 執行啟動前檢查..."

# Check if required directories exist | 檢查必需目錄是否存在
for dir in logs data models output; do
    if [ ! -d "$dir" ]; then
        echo "📁 Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

# Check Python dependencies | 檢查Python依賴項
echo "🐍 Checking Python environment..."
python --version
pip list --format=freeze > /tmp/requirements_check.txt
echo "✅ Python environment ready"

# ============================================================================
# Database Connection Checks | 資料庫連接檢查
# ============================================================================
if [ "$AIFX_ENV" = "production" ]; then
    echo "🗄️  Checking database connections..."
    
    # Wait for PostgreSQL | 等待PostgreSQL
    if [ -n "$POSTGRES_HOST" ]; then
        echo "⏳ Waiting for PostgreSQL at $POSTGRES_HOST..."
        while ! nc -z "$POSTGRES_HOST" 5432; do
            sleep 1
        done
        echo "✅ PostgreSQL is ready"
    fi
    
    # Wait for Redis | 等待Redis
    if [ -n "$REDIS_HOST" ]; then
        echo "⏳ Waiting for Redis at $REDIS_HOST..."
        while ! nc -z "$REDIS_HOST" 6379; do
            sleep 1
        done
        echo "✅ Redis is ready"
    fi
    
    # Wait for MongoDB | 等待MongoDB
    if [ -n "$MONGODB_HOST" ]; then
        echo "⏳ Waiting for MongoDB at $MONGODB_HOST..."
        while ! nc -z "$MONGODB_HOST" 27017; do
            sleep 1
        done
        echo "✅ MongoDB is ready"
    fi
fi

# ============================================================================
# Application Initialization | 應用程式初始化
# ============================================================================
echo "⚙️  Initializing application..."
echo "⚙️  初始化應用程式..."

# Run database migrations if needed | 如需要運行資料庫遷移
if [ "$AIFX_ENV" = "production" ] && [ -f "scripts/migrate.py" ]; then
    echo "🔄 Running database migrations..."
    python scripts/migrate.py
    echo "✅ Database migrations completed"
fi

# Load AI models if they exist | 如果存在則加載AI模型
if [ -d "models/trained" ] && [ "$(ls -A models/trained)" ]; then
    echo "🧠 Loading trained AI models..."
    echo "✅ AI models loaded"
else
    echo "⚠️  No trained models found, will use default models"
fi

# ============================================================================
# Performance Tuning | 性能調優
# ============================================================================
echo "🔧 Applying performance tuning..."

# Set memory limits | 設置內存限制
export MALLOC_TRIM_THRESHOLD_=100000
export MALLOC_TOP_PAD_=100000

# Python optimizations | Python優化
export PYTHONOPTIMIZE=1
export PYTHONHASHSEED=random

echo "✅ Performance tuning applied"

# ============================================================================
# Monitoring Setup | 監控設置
# ============================================================================
if [ "$AIFX_ENV" = "production" ]; then
    echo "📊 Setting up monitoring..."
    
    # Create prometheus multiproc directory | 創建prometheus多進程目錄
    mkdir -p /tmp/prometheus_multiproc
    export PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus_multiproc
    
    # Start background health check process | 啟動後台健康檢查進程
    python scripts/health_monitor.py &
    
    echo "✅ Monitoring setup completed"
fi

# ============================================================================
# Security Setup | 安全設置
# ============================================================================
echo "🔐 Applying security configurations..."

# Set secure permissions | 設置安全權限
chmod -R 750 /home/aifx/app
chmod -R 640 /home/aifx/app/config

# Remove sensitive environment variables from process list | 從進程列表中移除敏感環境變數
unset POSTGRES_PASSWORD 2>/dev/null || true
unset MONGO_ROOT_PASSWORD 2>/dev/null || true

echo "✅ Security configurations applied"

# ============================================================================
# Application Startup | 應用程式啟動
# ============================================================================
echo "🎯 Starting AIFX application..."
echo "🎯 啟動AIFX應用程式..."

# Log startup information | 記錄啟動信息
echo "$(date '+%Y-%m-%d %H:%M:%S') - AIFX application starting with PID $$" >> logs/startup.log

# Execute the main command | 執行主命令
echo "▶️  Executing: $@"
echo "▶️  執行: $@"

# Trap signals for graceful shutdown | 捕獲信號以優雅關閉
trap 'echo "🛑 Received shutdown signal, stopping AIFX..."; kill -TERM $PID; wait $PID' INT TERM

# Start the application | 啟動應用程式
exec "$@" &
PID=$!

# Wait for the application to start | 等待應用程式啟動
wait $PID

# ============================================================================
# Cleanup on Exit | 退出時清理
# ============================================================================
cleanup() {
    echo "🧹 Performing cleanup..."
    echo "🧹 執行清理..."
    
    # Clean up temporary files | 清理臨時文件
    rm -rf /tmp/prometheus_multiproc/* 2>/dev/null || true
    rm -f /tmp/requirements_check.txt 2>/dev/null || true
    
    # Log shutdown | 記錄關閉
    echo "$(date '+%Y-%m-%d %H:%M:%S') - AIFX application stopped" >> logs/startup.log
    
    echo "✅ Cleanup completed"
}

# Register cleanup function | 註冊清理函數
trap cleanup EXIT

echo "🏁 AIFX application has stopped"
echo "🏁 AIFX應用程式已停止"