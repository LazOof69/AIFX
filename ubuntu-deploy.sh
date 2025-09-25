#!/bin/bash
# AIFX Ubuntu Server One-Click Deployment Script
# AIFX Ubuntu 伺服器一鍵部署腳本
#
# Usage: curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/AIFX/main/ubuntu-deploy.sh | bash
# 使用方法: curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/AIFX/main/ubuntu-deploy.sh | bash

set -e  # Exit on any error

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
GITHUB_REPO="https://github.com/LazOof69/AIFX.git"  # 請替換為您的實際倉庫地址
PROJECT_DIR="AIFX"
MIN_MEMORY_GB=2
MIN_DISK_GB=10

# Print colored output functions
print_header() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Welcome message
show_welcome() {
    clear
    echo -e "${BLUE}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   🚀 AIFX Ubuntu Server One-Click Deployment Script 🚀      ║
║   🚀 AIFX Ubuntu 伺服器一鍵部署腳本 🚀                      ║
║                                                              ║
║   Professional quantitative trading system deployment       ║
║   專業量化交易系統部署                                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
${NC}"
    echo ""
    print_info "This script will automatically install and deploy AIFX on your Ubuntu server"
    print_info "此腳本將在您的 Ubuntu 伺服器上自動安裝和部署 AIFX"
    echo ""
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_error "This script should not be run as root for security reasons"
        print_error "出於安全考慮，此腳本不應以 root 身份運行"
        print_info "Please run as a regular user with sudo privileges"
        print_info "請以具有 sudo 權限的普通用戶身份運行"
        exit 1
    fi
}

# Check system requirements
check_system_requirements() {
    print_step "Checking system requirements | 檢查系統需求"

    # Check Ubuntu version
    if ! lsb_release -d | grep -q "Ubuntu"; then
        print_error "This script is designed for Ubuntu systems"
        print_error "此腳本專為 Ubuntu 系統設計"
        exit 1
    fi

    local ubuntu_version=$(lsb_release -rs)
    print_success "Ubuntu version: $ubuntu_version"

    # Check memory
    local memory_gb=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$memory_gb" -lt "$MIN_MEMORY_GB" ]; then
        print_warning "System has ${memory_gb}GB RAM (minimum ${MIN_MEMORY_GB}GB recommended)"
        print_warning "系統有 ${memory_gb}GB 記憶體（建議最少 ${MIN_MEMORY_GB}GB）"
    else
        print_success "Memory: ${memory_gb}GB ✅"
    fi

    # Check disk space
    local disk_space=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
    if [ "$disk_space" -lt "$MIN_DISK_GB" ]; then
        print_error "Insufficient disk space: ${disk_space}GB (minimum ${MIN_DISK_GB}GB required)"
        print_error "磁碟空間不足：${disk_space}GB（需要最少 ${MIN_DISK_GB}GB）"
        exit 1
    else
        print_success "Disk space: ${disk_space}GB available ✅"
    fi

    # Check internet connectivity
    if ! ping -c 1 google.com &> /dev/null; then
        print_error "No internet connectivity detected"
        print_error "未檢測到互聯網連接"
        exit 1
    else
        print_success "Internet connectivity ✅"
    fi

    echo ""
}

# Update system
update_system() {
    print_step "Updating system packages | 更新系統套件"

    sudo apt update -y
    sudo apt upgrade -y

    # Install essential packages
    sudo apt install -y curl wget git unzip software-properties-common \
        apt-transport-https ca-certificates gnupg lsb-release jq htop

    print_success "System updated successfully"
    echo ""
}

# Install Docker
install_docker() {
    print_step "Installing Docker and Docker Compose | 安裝 Docker 和 Docker Compose"

    # Check if Docker is already installed
    if command -v docker &> /dev/null; then
        print_info "Docker is already installed"
        docker --version
    else
        print_info "Installing Docker..."

        # Remove old versions
        sudo apt remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true

        # Add Docker GPG key
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

        # Add Docker repository
        echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

        # Update and install Docker
        sudo apt update
        sudo apt install -y docker-ce docker-ce-cli containerd.io

        # Add user to docker group
        sudo usermod -aG docker $USER

        # Start and enable Docker
        sudo systemctl start docker
        sudo systemctl enable docker

        print_success "Docker installed successfully"
    fi

    # Install Docker Compose
    if command -v docker-compose &> /dev/null; then
        print_info "Docker Compose is already installed"
        docker-compose --version
    else
        print_info "Installing Docker Compose..."

        sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose

        print_success "Docker Compose installed successfully"
    fi

    # Verify installation
    docker --version
    docker-compose --version

    echo ""
}

# Download AIFX from GitHub
download_aifx() {
    print_step "Downloading AIFX from GitHub | 從 GitHub 下載 AIFX"

    # Remove existing directory if it exists
    if [ -d "$PROJECT_DIR" ]; then
        print_warning "Removing existing AIFX directory"
        rm -rf "$PROJECT_DIR"
    fi

    # Clone repository
    print_info "Cloning AIFX repository..."
    if git clone "$GITHUB_REPO" "$PROJECT_DIR"; then
        print_success "AIFX repository cloned successfully"
    else
        print_error "Failed to clone AIFX repository"
        print_error "Please check the repository URL and your internet connection"
        exit 1
    fi

    # Navigate to project directory
    cd "$PROJECT_DIR"

    # Verify project structure
    if [ ! -d "cloud-deployment" ]; then
        print_error "Cloud deployment directory not found in the project"
        print_error "Please ensure you're using the latest version of AIFX"
        exit 1
    fi

    print_success "Project structure verified ✅"
    echo ""
}

# Configure environment
configure_environment() {
    print_step "Configuring environment | 配置環境"

    cd cloud-deployment

    # Copy environment template
    if [ -f ".env.cloud" ]; then
        cp .env.cloud .env
        print_success "Environment file created from template"
    else
        print_error "Environment template not found"
        exit 1
    fi

    # Make deployment scripts executable
    chmod +x deploy-cloud.sh 2>/dev/null || true

    # Detect server IP for display
    SERVER_IP=$(curl -s ifconfig.me 2>/dev/null || curl -s ipinfo.io/ip 2>/dev/null || echo "localhost")

    print_success "Environment configured successfully"
    print_info "Server IP detected: $SERVER_IP"
    echo ""
}

# Deploy AIFX
deploy_aifx() {
    print_step "Deploying AIFX Trading System | 部署 AIFX 交易系統"

    # Check if we need to use sudo for docker commands
    if ! docker ps &>/dev/null; then
        print_warning "Need to refresh docker group membership"
        print_info "You may need to logout and login again after deployment"
        DOCKER_CMD="sudo docker-compose"
    else
        DOCKER_CMD="docker-compose"
    fi

    # Stop any existing containers
    $DOCKER_CMD -f docker-compose.cloud.yml down --remove-orphans 2>/dev/null || true

    # Build and start services
    print_info "Building and starting services (this may take a few minutes)..."

    if $DOCKER_CMD -f docker-compose.cloud.yml up -d --build; then
        print_success "AIFX deployed successfully! 🎉"
    else
        print_error "Deployment failed!"
        print_info "Checking for common issues..."

        # Try to diagnose the issue
        if ! docker ps &>/dev/null; then
            print_error "Docker permission issue. Please run: sudo usermod -aG docker $USER"
            print_error "Then logout and login again, or run: newgrp docker"
        fi

        exit 1
    fi

    echo ""
}

# Perform health check
health_check() {
    print_step "Performing health check | 執行健康檢查"

    # Wait for services to start
    print_info "Waiting for services to initialize..."
    sleep 30

    # Check container status
    if docker-compose -f docker-compose.cloud.yml ps | grep -q "Up"; then
        print_success "Container is running ✅"
    else
        print_error "Container is not running properly"
        docker-compose -f docker-compose.cloud.yml ps
        docker-compose -f docker-compose.cloud.yml logs --tail=20
        return 1
    fi

    # Check health endpoint
    local port=$(grep AIFX_WEB_PORT .env | cut -d'=' -f2 2>/dev/null || echo "8080")

    print_info "Testing health endpoint..."
    for i in {1..5}; do
        if curl -f -s "http://localhost:${port}/api/health" > /dev/null; then
            print_success "Health check passed ✅"
            return 0
        fi
        print_info "Attempt $i/5 - waiting..."
        sleep 10
    done

    print_warning "Health check timeout - service may still be starting"
    echo ""
}

# Setup firewall (optional)
setup_firewall() {
    print_step "Setting up basic firewall | 設置基本防火牆"

    if command -v ufw &> /dev/null; then
        print_info "Configuring UFW firewall..."

        # Install UFW if not present
        sudo apt install -y ufw

        # Configure basic rules
        sudo ufw default deny incoming
        sudo ufw default allow outgoing
        sudo ufw allow ssh
        sudo ufw allow 8080/tcp

        # Enable firewall (with auto-confirm)
        echo "y" | sudo ufw enable

        print_success "Firewall configured ✅"
    else
        print_info "UFW not available, skipping firewall setup"
    fi

    echo ""
}

# Show deployment results
show_results() {
    local server_ip=$(curl -s ifconfig.me 2>/dev/null || curl -s ipinfo.io/ip 2>/dev/null || echo "YOUR_SERVER_IP")
    local port=$(grep AIFX_WEB_PORT .env | cut -d'=' -f2 2>/dev/null || echo "8080")

    print_header "🎉 AIFX Deployment Completed Successfully! | AIFX 部署成功完成！"

    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗"
    echo -e "║                    🚀 ACCESS INFORMATION 🚀                  ║"
    echo -e "║                    🚀 訪問信息 🚀                           ║"
    echo -e "╠══════════════════════════════════════════════════════════════╣"
    echo -e "║                                                              ║"
    echo -e "║  🌐 Web Interface | 網頁介面:                                ║"
    echo -e "║     http://${server_ip}:${port}                                       ║"
    echo -e "║                                                              ║"
    echo -e "║  📊 API Documentation | API 文件:                            ║"
    echo -e "║     http://${server_ip}:${port}/docs                                  ║"
    echo -e "║                                                              ║"
    echo -e "║  ❤️  Health Check | 健康檢查:                                ║"
    echo -e "║     http://${server_ip}:${port}/api/health                           ║"
    echo -e "║                                                              ║"
    echo -e "║  📈 Trading Signals | 交易信號:                              ║"
    echo -e "║     http://${server_ip}:${port}/api/signals                          ║"
    echo -e "║                                                              ║"
    echo -e "╚══════════════════════════════════════════════════════════════╝${NC}"

    echo ""
    echo -e "${BLUE}🛠️  Management Commands | 管理命令:${NC}"
    echo ""
    echo -e "${CYAN}  View logs | 查看日誌:${NC}"
    echo "    cd $HOME/$PROJECT_DIR/cloud-deployment"
    echo "    docker-compose -f docker-compose.cloud.yml logs -f"
    echo ""
    echo -e "${CYAN}  Stop service | 停止服務:${NC}"
    echo "    docker-compose -f docker-compose.cloud.yml down"
    echo ""
    echo -e "${CYAN}  Restart service | 重新啟動服務:${NC}"
    echo "    docker-compose -f docker-compose.cloud.yml restart"
    echo ""
    echo -e "${CYAN}  Update service | 更新服務:${NC}"
    echo "    cd $HOME/$PROJECT_DIR && git pull origin main"
    echo "    cd cloud-deployment && docker-compose -f docker-compose.cloud.yml down"
    echo "    docker-compose -f docker-compose.cloud.yml up -d --build"
    echo ""

    echo -e "${YELLOW}📋 Next Steps | 後續步驟:${NC}"
    echo "  1. Access the web interface using the URL above"
    echo "  1. 使用上面的網址訪問網頁介面"
    echo "  2. Configure your trading parameters if needed"
    echo "  2. 如需要，配置您的交易參數"
    echo "  3. Review the logs to ensure everything is working properly"
    echo "  3. 檢查日誌確保一切正常運行"
    echo ""

    if ! docker ps &>/dev/null; then
        echo -e "${RED}⚠️  Important Note | 重要提醒:${NC}"
        echo "  You may need to logout and login again to use docker commands without sudo"
        echo "  您可能需要登出後重新登入才能不使用 sudo 執行 docker 命令"
        echo "  Or run: newgrp docker"
        echo ""
    fi
}

# Main deployment process
main() {
    show_welcome

    # Ask for confirmation
    echo -e "${YELLOW}This script will install Docker, download AIFX, and deploy it on your server.${NC}"
    echo -e "${YELLOW}此腳本將安裝 Docker，下載 AIFX 並在您的伺服器上部署。${NC}"
    echo ""
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Deployment cancelled by user"
        exit 0
    fi

    echo ""

    # Run deployment steps
    check_root
    check_system_requirements
    update_system
    install_docker
    download_aifx
    configure_environment
    deploy_aifx
    health_check
    setup_firewall
    show_results

    print_success "🎉 AIFX deployment completed successfully!"
    print_success "🎉 AIFX 部署成功完成！"
}

# Handle script interruption
trap 'print_error "Script interrupted by user"; exit 1' INT TERM

# Run main function
main "$@"