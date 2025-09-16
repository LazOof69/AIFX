#!/bin/bash
# AIFX 24/7 USD/JPY Trading System - Production Deployment Script
# AIFX 24/7 美元/日圓交易系統 - 生產部署腳本

set -e

echo "🚀 AIFX 24/7 USD/JPY Trading System Deployment"
echo "🚀 AIFX 24/7 美元/日圓交易系統部署"
echo "=================================================="

# ============================================================================
# SYSTEM CHECKS | 系統檢查
# ============================================================================

echo ""
echo "🔍 Performing system checks..."
echo "🔍 執行系統檢查..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose >/dev/null 2>&1; then
    echo "❌ Docker Compose not found. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker environment ready"

# ============================================================================
# CREDENTIAL SETUP CHECK | 憑證設置檢查
# ============================================================================

echo ""
echo "🔐 Checking credentials setup..."
echo "🔐 檢查憑證設置..."

if [ ! -d "secrets" ]; then
    echo "⚠️ Secrets directory not found. Running credential setup..."
    echo "⚠️ 未找到秘密目錄。運行憑證設置..."
    ./setup_credentials.sh
fi

# Check required secret files
required_secrets=(
    "secrets/db_password.txt"
    "secrets/ig_api_key.txt"
    "secrets/ig_username.txt"
    "secrets/ig_password.txt"
    "secrets/ig_account_id.txt"
    "secrets/grafana_password.txt"
)

missing_secrets=()
for secret in "${required_secrets[@]}"; do
    if [ ! -f "$secret" ]; then
        missing_secrets+=("$secret")
    fi
done

if [ ${#missing_secrets[@]} -gt 0 ]; then
    echo "❌ Missing required secret files:"
    for secret in "${missing_secrets[@]}"; do
        echo "   - $secret"
    done
    echo ""
    echo "Please run: ./setup_credentials.sh"
    exit 1
fi

echo "✅ All credentials configured"

# ============================================================================
# CONFIGURATION VALIDATION | 配置驗證
# ============================================================================

echo ""
echo "⚙️ Validating configuration..."
echo "⚙️ 驗證配置..."

# Check if configuration files exist
if [ ! -f "docker-compose-24x7-usdjpy.yml" ]; then
    echo "❌ Docker Compose configuration not found: docker-compose-24x7-usdjpy.yml"
    exit 1
fi

if [ ! -f "database/init/01_init_trading_db.sql" ]; then
    echo "❌ Database initialization script not found"
    exit 1
fi

echo "✅ Configuration validation passed"

# ============================================================================
# RISK WARNING | 風險警告
# ============================================================================

echo ""
echo "🚨 IMPORTANT RISK WARNING | 重要風險警告"
echo "========================================="
echo ""
echo "⚠️ You are about to start LIVE TRADING with real money!"
echo "⚠️ 您即將開始使用真實資金的實盤交易！"
echo ""
echo "🔴 This system will:"
echo "   • Trade USD/JPY with your real IG Markets account"
echo "   • Use real money for all trades"
echo "   • Run continuously 24/7"
echo "   • Execute trades automatically based on AI signals"
echo ""
echo "🔴 此系統將："
echo "   • 使用您的真實IG Markets帳戶交易美元/日圓"
echo "   • 所有交易使用真實資金"
echo "   • 24/7持續運行"
echo "   • 基於AI信號自動執行交易"
echo ""
echo "💡 Monitor your system at:"
echo "   • Trading Dashboard: http://localhost:8088"
echo "   • Grafana Monitor: http://localhost:3000"
echo ""

read -p "Do you understand the risks and want to continue? (type 'YES' to confirm): " risk_confirmation

if [ "$risk_confirmation" != "YES" ]; then
    echo "❌ Deployment cancelled for safety. Use demo mode instead."
    echo "❌ 為了安全起見取消部署。請使用演示模式。"
    exit 1
fi

# ============================================================================
# FINAL DEPLOYMENT | 最終部署
# ============================================================================

echo ""
echo "🚀 Starting 24/7 USD/JPY Trading System..."
echo "🚀 啟動24/7美元/日圓交易系統..."

# Stop any existing services
echo "🛑 Stopping any existing trading services..."
docker-compose -f docker-compose-24x7-usdjpy.yml down --remove-orphans >/dev/null 2>&1 || true

# Build and start all services
echo "🔧 Building and starting all services..."
docker-compose -f docker-compose-24x7-usdjpy.yml up -d --build

# Wait for services to start
echo "⏳ Waiting for services to initialize..."
sleep 30

# ============================================================================
# HEALTH CHECKS | 健康檢查
# ============================================================================

echo ""
echo "🏥 Performing health checks..."
echo "🏥 執行健康檢查..."

# Check service status
services_status=$(docker-compose -f docker-compose-24x7-usdjpy.yml ps --services)
running_services=$(docker-compose -f docker-compose-24x7-usdjpy.yml ps --filter "status=running" --services)

echo "📊 Service Status:"
for service in $services_status; do
    if echo "$running_services" | grep -q "^$service$"; then
        echo "   ✅ $service: Running"
    else
        echo "   ❌ $service: Not Running"
    fi
done

# Check trading system health
echo ""
echo "🔍 Checking trading system health..."
sleep 5

if curl -f http://localhost:8088/health >/dev/null 2>&1; then
    echo "✅ Trading system is healthy"
else
    echo "⚠️ Trading system health check failed"
fi

# Check database connectivity
echo "🗄️ Testing database connectivity..."
if docker-compose -f docker-compose-24x7-usdjpy.yml exec -T postgres-db psql -U aifx -d aifx_trading_24x7 -c "SELECT 1;" >/dev/null 2>&1; then
    echo "✅ Database is accessible"
else
    echo "⚠️ Database connectivity issues"
fi

# ============================================================================
# DEPLOYMENT SUMMARY | 部署摘要
# ============================================================================

echo ""
echo "🎉 24/7 USD/JPY TRADING SYSTEM DEPLOYMENT COMPLETE!"
echo "🎉 24/7美元/日圓交易系統部署完成！"
echo "======================================================"
echo ""
echo "📊 System Status:"
echo "   • Live Trading: ACTIVE (USD/JPY only)"
echo "   • Mode: 24/7 Continuous Operation"
echo "   • Auto-restart: Enabled"
echo "   • Backups: Every 6 hours"
echo "   • Monitoring: Active"
echo ""
echo "🌐 Access Points:"
echo "   • Trading Dashboard: http://localhost:8088"
echo "   • Grafana Monitor: http://localhost:3000 (admin/your_password)"
echo "   • Database: localhost:5432"
echo ""
echo "📋 Management Commands:"
echo "   • View logs: docker-compose -f docker-compose-24x7-usdjpy.yml logs -f aifx-trading"
echo "   • Stop system: docker-compose -f docker-compose-24x7-usdjpy.yml down"
echo "   • Restart: docker-compose -f docker-compose-24x7-usdjpy.yml restart aifx-trading"
echo "   • Status: docker-compose -f docker-compose-24x7-usdjpy.yml ps"
echo ""
echo "⚠️ REMEMBER: This is LIVE TRADING with real money!"
echo "📈 Monitor your positions and performance regularly."
echo "🛑 You can stop the system anytime with Ctrl+C or the stop command above."
echo ""
echo "✅ Your 24/7 USD/JPY trading system is now running!"
echo "✅ 您的24/7美元/日圓交易系統現已運行！"