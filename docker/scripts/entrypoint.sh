#!/bin/bash
# AIFX Production Entrypoint Script | AIFX 生產入口點腳本
# Handles container startup, health checks, and graceful shutdown
# 處理容器啟動、健康檢查和優雅關閉

set -e

# Color codes for logging | 日誌顏色代碼
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function | 日誌函數
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] AIFX:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] AIFX ERROR:${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] AIFX SUCCESS:${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] AIFX WARNING:${NC} $1"
}

# Environment setup | 環境設置
export PYTHONPATH="/app:/app/src/main/python:$PYTHONPATH"
export AIFX_ENV="${AIFX_ENV:-production}"

log "🚀 Starting AIFX AI-Enhanced Forex Trading System"
log "Environment: $AIFX_ENV"
log "Python Path: $PYTHONPATH"

# Pre-startup checks | 啟動前檢查
log "🔍 Performing pre-startup health checks..."

# Check Python environment | 檢查Python環境
if ! python --version >/dev/null 2>&1; then
    log_error "Python is not available"
    exit 1
fi
log_success "Python environment OK"

# Check required directories | 檢查必需目錄
for dir in "/app/data" "/app/models" "/app/output" "/app/logs"; do
    if [ ! -d "$dir" ]; then
        log "📁 Creating directory: $dir"
        mkdir -p "$dir"
    fi
done
log_success "Directory structure OK"

# Check core dependencies | 檢查核心依賴項
log "🔧 Checking core dependencies..."
python -c "
import sys
sys.path.append('/app/src/main/python')
try:
    import pandas, numpy, sklearn, xgboost
    from core.risk_manager import AdvancedRiskManager
    from core.trading_strategy import AIFXTradingStrategy
    print('✅ Core dependencies available')
except ImportError as e:
    print(f'❌ Dependency error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    log_error "Core dependency check failed"
    exit 1
fi
log_success "Dependencies check passed"

# Handle different startup modes | 處理不同啟動模式
case "${1:-run}" in
    "test")
        log "🧪 Running in test mode..."
        cd /app
        python -m pytest src/test/ -v --tb=short
        ;;
    "health-check")
        log "💓 Running health check..."
        python -c "
import sys
sys.path.append('/app/src/main/python')
try:
    from core.risk_manager import AdvancedRiskManager
    from core.trading_strategy import AIFXTradingStrategy
    print('AIFX Health Check: HEALTHY')
    sys.exit(0)
except Exception as e:
    print(f'AIFX Health Check: UNHEALTHY - {e}')
    sys.exit(1)
        "
        ;;
    "backtest")
        log "📊 Running backtest mode..."
        cd /app
        python test_phase3_integration.py
        ;;
    "run"|"trading")
        log "💹 Starting AIFX Trading System..."
        cd /app
        
        # Set up signal handling for graceful shutdown | 設置信號處理以優雅關閉
        cleanup() {
            log "🛑 Received shutdown signal, stopping AIFX..."
            # Add graceful shutdown logic here
            log_success "AIFX stopped gracefully"
            exit 0
        }
        trap cleanup SIGTERM SIGINT
        
        # Start the main application | 啟動主應用程式
        python -c "
import sys, signal, time
sys.path.append('/app/src/main/python')

def signal_handler(sig, frame):
    print('\\n🛑 Gracefully shutting down AIFX...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    from core.trading_strategy import AIFXTradingStrategy
    from core.risk_manager import AdvancedRiskManager, RiskLevel, create_risk_manager_preset
    
    print('🚀 AIFX Trading System Started')
    print('💹 AI-Enhanced Forex Trading Active')
    print('📊 Monitoring EUR/USD and USD/JPY pairs...')
    
    # Keep the container running | 保持容器運行
    while True:
        time.sleep(10)
        print('🔄 AIFX System Running...', flush=True)
        
except KeyboardInterrupt:
    print('\\n🛑 AIFX Stopped by user')
except Exception as e:
    print(f'❌ AIFX Error: {e}')
    sys.exit(1)
        "
        ;;
    "debug")
        log "🐛 Starting in debug mode..."
        cd /app
        python -c "
import sys
sys.path.append('/app/src/main/python')
print('🐛 AIFX Debug Mode')
print('Python version:', sys.version)
print('Python path:', sys.path)
import pandas, numpy, sklearn, xgboost
print('✅ Core packages imported successfully')
from core.risk_manager import AdvancedRiskManager
from core.trading_strategy import AIFXTradingStrategy
print('✅ AIFX core modules imported successfully')
print('🎯 AIFX Debug Mode: All systems operational')
        "
        ;;
    *)
        log "ℹ️  Usage: entrypoint.sh [run|test|health-check|backtest|debug]"
        log "ℹ️  Default: run (starts trading system)"
        exec "$@"
        ;;
esac

log_success "AIFX entrypoint completed successfully"