#!/bin/bash
# AIFX Cloud Deployment Script
# AIFX 雲端部署腳本

set -e  # Exit on any error

echo "🚀 AIFX Cloud Deployment Script | AIFX 雲端部署腳本"
echo "=================================================="

# Color definitions for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Check if Docker is installed
check_docker() {
    print_header "📦 Checking Docker Installation | 檢查 Docker 安裝"

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        print_error "Docker 未安裝。請先安裝 Docker。"
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        print_error "Docker Compose 未安裝。請先安裝 Docker Compose。"
        exit 1
    fi

    print_status "Docker and Docker Compose are installed ✅"
    print_status "Docker 和 Docker Compose 已安裝 ✅"
}

# Check system requirements
check_system() {
    print_header "🖥️  Checking System Requirements | 檢查系統需求"

    # Check available memory
    MEMORY_GB=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$MEMORY_GB" -lt 2 ]; then
        print_warning "System has less than 2GB RAM. Performance may be affected."
        print_warning "系統記憶體少於 2GB。性能可能受到影響。"
    else
        print_status "Memory: ${MEMORY_GB}GB ✅"
        print_status "記憶體：${MEMORY_GB}GB ✅"
    fi

    # Check disk space
    DISK_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$DISK_SPACE" -lt 10 ]; then
        print_warning "Less than 10GB disk space available."
        print_warning "可用磁碟空間少於 10GB。"
    else
        print_status "Disk space: ${DISK_SPACE}GB available ✅"
        print_status "磁碟空間：${DISK_SPACE}GB 可用 ✅"
    fi
}

# Setup environment
setup_environment() {
    print_header "⚙️  Setting up Environment | 設置環境"

    # Copy environment file if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.cloud" ]; then
            cp .env.cloud .env
            print_status "Environment file created from template ✅"
            print_status "從模板創建環境文件 ✅"
        else
            print_error "No environment template found!"
            print_error "找不到環境模板！"
            exit 1
        fi
    else
        print_status "Environment file already exists ✅"
        print_status "環境文件已存在 ✅"
    fi
}

# Deploy application
deploy_application() {
    print_header "🚀 Deploying Application | 部署應用程式"

    # Stop existing containers if running
    print_status "Stopping existing containers... | 停止現有容器..."
    docker-compose -f docker-compose.cloud.yml down --remove-orphans || true

    # Build and start services
    print_status "Building and starting services... | 建置並啟動服務..."
    docker-compose -f docker-compose.cloud.yml up -d --build

    if [ $? -eq 0 ]; then
        print_status "Application deployed successfully! ✅"
        print_status "應用程式部署成功！✅"
    else
        print_error "Deployment failed! ❌"
        print_error "部署失敗！❌"
        exit 1
    fi
}

# Health check
health_check() {
    print_header "🏥 Performing Health Check | 執行健康檢查"

    print_status "Waiting for services to be ready... | 等待服務準備就緒..."
    sleep 30

    # Check if container is running
    if docker-compose -f docker-compose.cloud.yml ps | grep -q "Up"; then
        print_status "Container is running ✅"
        print_status "容器正在運行 ✅"
    else
        print_error "Container is not running ❌"
        print_error "容器未運行 ❌"
        return 1
    fi

    # Check HTTP endpoint
    PORT=$(grep AIFX_WEB_PORT .env | cut -d'=' -f2 || echo "8080")
    if curl -f -s "http://localhost:${PORT}/api/health" > /dev/null; then
        print_status "Health check passed ✅"
        print_status "健康檢查通過 ✅"
    else
        print_warning "Health check failed - service may still be starting"
        print_warning "健康檢查失敗 - 服務可能仍在啟動中"
    fi
}

# Show status and access info
show_info() {
    print_header "📊 Deployment Information | 部署信息"

    PORT=$(grep AIFX_WEB_PORT .env | cut -d'=' -f2 || echo "8080")

    echo ""
    echo "🎉 AIFX is now running! | AIFX 現在正在運行！"
    echo ""
    echo "📱 Access URLs | 訪問網址："
    echo "   🌐 Web Interface: http://localhost:${PORT}"
    echo "   🌐 網頁介面：http://localhost:${PORT}"
    echo "   ❤️  Health Check: http://localhost:${PORT}/api/health"
    echo "   ❤️  健康檢查：http://localhost:${PORT}/api/health"
    echo "   📊 API Documentation: http://localhost:${PORT}/docs"
    echo "   📊 API 文件：http://localhost:${PORT}/docs"
    echo ""
    echo "🛠️  Management Commands | 管理命令："
    echo "   View logs | 查看日誌：        docker-compose -f docker-compose.cloud.yml logs -f"
    echo "   Stop service | 停止服務：       docker-compose -f docker-compose.cloud.yml down"
    echo "   Restart service | 重新啟動服務： docker-compose -f docker-compose.cloud.yml restart"
    echo "   Update service | 更新服務：     docker-compose -f docker-compose.cloud.yml pull && docker-compose -f docker-compose.cloud.yml up -d"
    echo ""
}

# Main deployment process
main() {
    check_docker
    check_system
    setup_environment
    deploy_application
    health_check
    show_info
}

# Run main function
main

print_status "Deployment completed! | 部署完成！"