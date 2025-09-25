# AIFX Ubuntu Server Deployment Master Guide | AIFX Ubuntu 伺服器部署主要指南

> **🎯 Complete deployment guide for Ubuntu servers | Ubuntu 伺服器完整部署指南**
> **Version | 版本**: 3.0
> **Last Updated | 最後更新**: 2025-09-25
> **System Status | 系統狀態**: ✅ Production Ready | 生產就緒

## 📋 Table of Contents | 目錄

1. [🚀 Quick Deployment (Recommended) | 快速部署（建議）](#quick-deployment)
2. [📋 Prerequisites & Requirements | 前置需求](#prerequisites)
3. [🛠️ Manual Step-by-Step Installation | 手動逐步安裝](#manual-installation)
4. [⚙️ Advanced Configuration | 進階配置](#advanced-configuration)
5. [🔧 Management Commands | 管理命令](#management-commands)
6. [🚨 Troubleshooting | 故障排除](#troubleshooting)
7. [📊 Performance Optimization | 性能優化](#performance-optimization)
8. [🔒 Security Configuration | 安全配置](#security-configuration)
9. [📈 Monitoring & Maintenance | 監控與維護](#monitoring-maintenance)
10. [🆘 Support & Resources | 支援與資源](#support-resources)

---

## 🚀 Quick Deployment (Recommended) | 快速部署（建議） {#quick-deployment}

### ⚡ One-Command Deployment | 一鍵部署

**Copy and paste this single command into your Ubuntu terminal:**
**複製並貼上此命令到您的 Ubuntu 終端：**

```bash
curl -fsSL https://raw.githubusercontent.com/LazOof69/AIFX/main/ubuntu-deploy.sh | bash
```

**This command will automatically:**
**此命令將自動：**

- ✅ Check system requirements and compatibility | 檢查系統需求和兼容性
- ✅ Install Docker and Docker Compose | 安裝 Docker 和 Docker Compose
- ✅ Download AIFX from GitHub | 從 GitHub 下載 AIFX
- ✅ Configure environment for production | 配置生產環境
- ✅ Deploy and start all services | 部署並啟動所有服務
- ✅ Setup basic firewall security | 設置基本防火牆安全
- ✅ Perform comprehensive health checks | 執行全面健康檢查
- ✅ Display access information and management commands | 顯示訪問信息和管理命令

### 🌐 Access Your System | 訪問您的系統

After successful deployment, access at:
部署成功後，在以下地址訪問：

- **Web Interface | 網頁介面**: `http://YOUR_SERVER_IP:8080`
- **API Documentation | API 文檔**: `http://YOUR_SERVER_IP:8080/docs`
- **Health Check | 健康檢查**: `http://YOUR_SERVER_IP:8080/api/health`
- **Trading Signals | 交易信號**: `http://YOUR_SERVER_IP:8080/api/signals`

Replace `YOUR_SERVER_IP` with your actual server IP address.
將 `YOUR_SERVER_IP` 替換為您實際的伺服器 IP 地址。

---

## 📋 Prerequisites & Requirements | 前置需求 {#prerequisites}

### 🖥️ System Requirements | 系統需求

- **Operating System | 操作系統**: Ubuntu 18.04+ (20.04 LTS recommended | 建議)
- **Memory | 記憶體**: Minimum 2GB RAM (4GB recommended | 建議)
- **Storage | 儲存空間**: Minimum 10GB free disk space | 最少 10GB 可用磁碟空間
- **Network | 網路**: Internet connection for downloads | 下載用的網路連接
- **Permissions | 權限**: Sudo privileges OR root access | Sudo 權限或 root 訪問權限

### 👥 **User Support | 用戶支援**

**✅ Regular Users with Sudo**
- Standard deployment with sudo privilege requirement
- Docker group membership automatically configured
- Best practice for shared systems

**✅ Root Users** *(Fully Supported)*
- Enhanced security mode with automatic permission handling
- Direct Docker access without group membership requirements
- Optimal for dedicated servers and VPS environments
- Advanced system access and control capabilities

### 🔗 Network Requirements | 網路需求

**Outbound connections required | 需要的外部連接：**
- Port 80/443: Package downloads and Git clone | 套件下載和 Git 克隆
- Port 22: SSH access (if remote) | SSH 訪問（如果是遠程）

**Inbound connections | 入站連接：**
- Port 8080: AIFX Web Interface (default) | AIFX 網頁介面（預設）
- Port 22: SSH management | SSH 管理

---

## 🛠️ Manual Step-by-Step Installation | 手動逐步安裝 {#manual-installation}

### Step 1: System Preparation | 步驟 1：系統準備

```bash
# Update system packages | 更新系統套件
sudo apt update && sudo apt upgrade -y

# Install essential packages | 安裝基本套件
sudo apt install -y curl wget git unzip software-properties-common \
    apt-transport-https ca-certificates gnupg lsb-release
```

### Step 2: Install Docker | 步驟 2：安裝 Docker

```bash
# Remove old Docker versions | 移除舊版本 Docker
sudo apt remove -y docker docker-engine docker.io containerd runc

# Add Docker GPG key | 添加 Docker GPG 密鑰
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository | 添加 Docker 儲存庫
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker | 安裝 Docker
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Add user to docker group | 將用戶加入 docker 群組
sudo usermod -aG docker $USER

# Start and enable Docker | 啟動並啟用 Docker
sudo systemctl start docker
sudo systemctl enable docker
```

### Step 3: Install Docker Compose | 步驟 3：安裝 Docker Compose

```bash
# Download and install Docker Compose | 下載並安裝 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# Make executable | 設為可執行
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation | 驗證安裝
docker --version
docker-compose --version
```

### Step 4: Download AIFX | 步驟 4：下載 AIFX

```bash
# Clone AIFX repository | 克隆 AIFX 儲存庫
git clone https://github.com/LazOof69/AIFX.git

# Navigate to project directory | 導航到專案目錄
cd AIFX

# Verify project structure | 驗證專案結構
ls -la cloud-deployment/
```

### Step 5: Configure Environment | 步驟 5：配置環境

```bash
# Navigate to deployment directory | 導航到部署目錄
cd cloud-deployment

# Copy server environment template | 複製伺服器環境模板
cp .env.server .env

# Make deployment script executable | 設置部署腳本為可執行
chmod +x deploy-cloud.sh

# Optional: Edit configuration | 可選：編輯配置
nano .env
```

### Step 6: Deploy AIFX | 步驟 6：部署 AIFX

```bash
# Build and start services | 建構並啟動服務
docker-compose -f docker-compose.cloud.yml up -d --build

# Verify deployment | 驗證部署
docker-compose -f docker-compose.cloud.yml ps
```

### Step 7: Validation | 步驟 7：驗證

```bash
# Navigate back to main directory | 導航回主目錄
cd ..

# Run validation script | 運行驗證腳本
chmod +x validate-deployment.sh
./validate-deployment.sh
```

---

## ⚙️ Advanced Configuration | 進階配置 {#advanced-configuration}

### 🔧 Environment Variables | 環境變數

**Key configuration options in `.env` file:**
**`.env` 文件中的關鍵配置選項：**

```bash
# Application Settings | 應用程式設定
AIFX_WEB_PORT=8080
AIFX_LOG_LEVEL=INFO
AIFX_DEBUG=false
AIFX_ENVIRONMENT=production

# Performance Settings | 性能設定
WORKERS=2                    # 2-4GB RAM servers | 2-4GB 記憶體伺服器
MAX_CONNECTIONS=100
CACHE_TTL=600

# For high-performance servers (8GB+ RAM) | 高性能伺服器（8GB+ 記憶體）
# WORKERS=6
# MAX_CONNECTIONS=300
# CACHE_TTL=1200

# Security Settings | 安全設定
ALLOWED_ORIGINS=*           # Change for production | 生產環境請更改
API_KEY_REQUIRED=false      # Enable for production | 生產環境請啟用
RATE_LIMIT_ENABLED=true

# Trading Configuration | 交易配置
TRADING_ENABLED=false       # Set to true for live trading | 設為 true 進行實盤交易
TRADING_MODE=demo
DEFAULT_CURRENCY_PAIRS=EUR/USD,USD/JPY,GBP/USD
```

### 🐳 Docker Resource Limits | Docker 資源限制

**Modify `docker-compose.cloud.yml` for your server specs:**
**根據您的伺服器規格修改 `docker-compose.cloud.yml`：**

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'      # Adjust based on your server | 根據您的伺服器調整
      memory: 2G       # Adjust based on available RAM | 根據可用記憶體調整
    reservations:
      cpus: '1.0'
      memory: 1G
```

### 📊 Database Configuration | 資料庫配置

**For production with PostgreSQL:**
**生產環境使用 PostgreSQL：**

```bash
# Add to .env file | 添加到 .env 文件
DATABASE_URL=postgresql://username:password@localhost:5432/aifx
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20
```

---

## 🔧 Management Commands | 管理命令 {#management-commands}

### 📊 Status & Monitoring | 狀態與監控

```bash
# Check service status | 檢查服務狀態
docker-compose -f cloud-deployment/docker-compose.cloud.yml ps

# View live logs | 查看即時日誌
docker-compose -f cloud-deployment/docker-compose.cloud.yml logs -f

# Check container resource usage | 檢查容器資源使用
docker stats

# Health check | 健康檢查
curl http://localhost:8080/api/health

# System status | 系統狀態
curl http://localhost:8080/api/status
```

### 🔄 Service Management | 服務管理

```bash
# Start services | 啟動服務
docker-compose -f cloud-deployment/docker-compose.cloud.yml up -d

# Stop services | 停止服務
docker-compose -f cloud-deployment/docker-compose.cloud.yml down

# Restart services | 重啟服務
docker-compose -f cloud-deployment/docker-compose.cloud.yml restart

# Rebuild and restart | 重建並重啟
docker-compose -f cloud-deployment/docker-compose.cloud.yml down
docker-compose -f cloud-deployment/docker-compose.cloud.yml up -d --build
```

### 🔄 Updates | 更新

```bash
# Update AIFX | 更新 AIFX
cd AIFX
git pull origin main

# Redeploy with updates | 使用更新重新部署
cd cloud-deployment
docker-compose -f docker-compose.cloud.yml down
docker-compose -f docker-compose.cloud.yml up -d --build
```

---

## 🚨 Troubleshooting | 故障排除 {#troubleshooting}

### ❌ Common Issues & Solutions | 常見問題與解決方案

#### 1. Docker Permission Errors | Docker 權限錯誤

```bash
# Problem: "permission denied" when using docker commands
# 問題：使用 docker 命令時出現「權限被拒絕」

# Solution | 解決方案：
sudo usermod -aG docker $USER
newgrp docker
# Or logout and login again | 或登出後重新登入
```

#### 2. Port 8080 Already in Use | 端口 8080 已被使用

```bash
# Check what's using the port | 檢查什麼在使用該端口
sudo netstat -tulpn | grep :8080
sudo lsof -i :8080

# Solution 1: Kill the process | 解決方案 1：終止進程
sudo kill -9 <PID>

# Solution 2: Change AIFX port | 解決方案 2：更改 AIFX 端口
echo "AIFX_WEB_PORT=8081" >> cloud-deployment/.env
docker-compose -f cloud-deployment/docker-compose.cloud.yml up -d
```

#### 3. Service Won't Start | 服務無法啟動

```bash
# Check logs for errors | 查看日誌錯誤
docker-compose -f cloud-deployment/docker-compose.cloud.yml logs

# Clean up and restart | 清理並重啟
docker-compose -f cloud-deployment/docker-compose.cloud.yml down
docker system prune -f
docker-compose -f cloud-deployment/docker-compose.cloud.yml up -d --build
```

#### 4. Out of Memory Issues | 記憶體不足問題

```bash
# Reduce workers | 減少工作進程
echo "WORKERS=1" >> cloud-deployment/.env
echo "MAX_CONNECTIONS=50" >> cloud-deployment/.env
docker-compose -f cloud-deployment/docker-compose.cloud.yml restart

# Check memory usage | 檢查記憶體使用
free -h
docker stats --no-stream
```

#### 5. Network Connectivity Issues | 網路連接問題

```bash
# Test internet connectivity | 測試網路連接
ping -c 4 google.com
curl -I https://github.com

# Test local service | 測試本地服務
curl -v http://localhost:8080/api/health

# Check firewall | 檢查防火牆
sudo ufw status
sudo iptables -L
```

---

## 📊 Performance Optimization | 性能優化 {#performance-optimization}

### 🚀 Server Specifications Based Configuration | 基於伺服器規格的配置

#### 💻 Small Server (2-4GB RAM) | 小型伺服器

```bash
# Optimized settings for small servers | 小型伺服器優化設定
cat >> cloud-deployment/.env << EOF
WORKERS=1
MAX_CONNECTIONS=50
CACHE_TTL=300
GC_THRESHOLD=50000
MAX_MEMORY_MB=768
THREAD_POOL_SIZE=10
EOF
```

#### 🖥️ Medium Server (4-8GB RAM) | 中型伺服器

```bash
# Optimized settings for medium servers | 中型伺服器優化設定
cat >> cloud-deployment/.env << EOF
WORKERS=2
MAX_CONNECTIONS=100
CACHE_TTL=600
GC_THRESHOLD=100000
MAX_MEMORY_MB=1536
THREAD_POOL_SIZE=20
EOF
```

#### 🏢 Large Server (8GB+ RAM) | 大型伺服器

```bash
# Optimized settings for large servers | 大型伺服器優化設定
cat >> cloud-deployment/.env << EOF
WORKERS=4
MAX_CONNECTIONS=200
CACHE_TTL=900
GC_THRESHOLD=200000
MAX_MEMORY_MB=3072
THREAD_POOL_SIZE=40
EOF
```

### 📈 Performance Monitoring | 性能監控

```bash
# Monitor system resources | 監控系統資源
htop
iotop
nethogs

# Monitor Docker containers | 監控 Docker 容器
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# Application performance metrics | 應用程式性能指標
curl http://localhost:8080/api/metrics
```

---

## 🔒 Security Configuration | 安全配置 {#security-configuration}

### 🛡️ Firewall Setup | 防火牆設置

```bash
# Install and configure UFW | 安裝並配置 UFW
sudo apt install -y ufw

# Default policies | 預設政策
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow essential services | 允許基本服務
sudo ufw allow ssh
sudo ufw allow 8080/tcp

# Enable firewall | 啟用防火牆
sudo ufw enable

# Check status | 檢查狀態
sudo ufw status verbose
```

### 🔐 SSL/TLS with Nginx (Production) | 使用 Nginx 的 SSL/TLS（生產環境）

```bash
# Install Nginx and Certbot | 安裝 Nginx 和 Certbot
sudo apt install -y nginx certbot python3-certbot-nginx

# Create Nginx configuration | 創建 Nginx 配置
sudo tee /etc/nginx/sites-available/aifx << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

# Enable site | 啟用站點
sudo ln -s /etc/nginx/sites-available/aifx /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get SSL certificate | 獲取 SSL 憑證
sudo certbot --nginx -d your-domain.com
```

### 🔑 API Security | API 安全

```bash
# Generate secure API key | 生成安全 API 密鑰
API_KEY=$(openssl rand -hex 32)
echo "API_KEY=$API_KEY" >> cloud-deployment/.env
echo "API_KEY_REQUIRED=true" >> cloud-deployment/.env

# Generate JWT secret | 生成 JWT 密鑰
JWT_SECRET=$(openssl rand -hex 64)
echo "JWT_SECRET=$JWT_SECRET" >> cloud-deployment/.env

# Restart services | 重啟服務
docker-compose -f cloud-deployment/docker-compose.cloud.yml restart
```

---

## 📈 Monitoring & Maintenance | 監控與維護 {#monitoring-maintenance}

### 📊 System Monitoring | 系統監控

```bash
# Install monitoring tools | 安裝監控工具
sudo apt install -y htop iotop nethogs ncdu

# System health check script | 系統健康檢查腳本
cat > ~/health-check.sh << 'EOF'
#!/bin/bash
echo "=== System Health Check ==="
echo "Date: $(date)"
echo "Uptime: $(uptime)"
echo "Memory: $(free -h | grep ^Mem | awk '{print $3 "/" $2}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')"
echo "Docker containers: $(docker ps --format 'table {{.Names}}\t{{.Status}}')"
echo "AIFX Health: $(curl -s http://localhost:8080/api/health | grep -o '"status":"[^"]*"' || echo 'Service down')"
EOF

chmod +x ~/health-check.sh
```

### 🔄 Automated Maintenance | 自動維護

```bash
# Setup cron jobs for maintenance | 設置維護定時任務
(crontab -l 2>/dev/null; cat << 'EOF'
# AIFX Maintenance Tasks
0 2 * * * docker system prune -f > /dev/null 2>&1
0 3 * * 0 cd /home/$USER/AIFX && git pull origin main > /dev/null 2>&1
30 1 * * * /home/$USER/health-check.sh >> /var/log/aifx-health.log
EOF
) | crontab -
```

### 📋 Log Management | 日誌管理

```bash
# Setup log rotation | 設置日誌輪轉
sudo tee /etc/logrotate.d/aifx << EOF
/var/log/aifx-health.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 $USER $USER
}
EOF

# View logs | 查看日誌
tail -f /var/log/aifx-health.log
docker-compose -f cloud-deployment/docker-compose.cloud.yml logs -f --tail=100
```

---

## 🔄 Backup and Recovery | 備份與恢復

### 💾 Data Backup | 數據備份

```bash
# Create backup directory | 創建備份目錄
mkdir -p ~/aifx-backups

# Backup application data | 備份應用程式數據
docker run --rm -v aifx_data:/data -v ~/aifx-backups:/backup alpine tar czf /backup/aifx_data_$(date +%Y%m%d_%H%M%S).tar.gz -C /data .

# Backup configuration | 備份配置
tar czf ~/aifx-backups/aifx_config_$(date +%Y%m%d_%H%M%S).tar.gz -C ~/AIFX cloud-deployment/.env

# Automated backup script | 自動備份腳本
cat > ~/backup-aifx.sh << 'EOF'
#!/bin/bash
BACKUP_DIR=~/aifx-backups
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup data volumes
docker run --rm -v aifx_data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/aifx_data_$DATE.tar.gz -C /data .

# Backup configuration
tar czf $BACKUP_DIR/aifx_config_$DATE.tar.gz -C ~/AIFX cloud-deployment/.env

# Remove backups older than 7 days
find $BACKUP_DIR -name "aifx_*" -mtime +7 -delete

echo "Backup completed: $DATE"
EOF

chmod +x ~/backup-aifx.sh
```

### 🔄 Data Recovery | 數據恢復

```bash
# Stop services | 停止服務
docker-compose -f ~/AIFX/cloud-deployment/docker-compose.cloud.yml down

# Restore data from backup | 從備份恢復數據
BACKUP_FILE=~/aifx-backups/aifx_data_YYYYMMDD_HHMMSS.tar.gz
docker run --rm -v aifx_data:/data -v ~/aifx-backups:/backup alpine tar xzf /backup/$(basename $BACKUP_FILE) -C /data

# Restore configuration | 恢復配置
tar xzf ~/aifx-backups/aifx_config_YYYYMMDD_HHMMSS.tar.gz -C ~/AIFX

# Start services | 啟動服務
docker-compose -f ~/AIFX/cloud-deployment/docker-compose.cloud.yml up -d
```

---

## 🆘 Support & Resources | 支援與資源 {#support-resources}

### 📚 Documentation | 文件

- **Quick Start Guide | 快速入門指南**: `UBUNTU_QUICK_START.md`
- **Detailed Setup | 詳細設置**: `UBUNTU_SERVER_DEPLOYMENT.md`
- **Project Rules | 專案規則**: `CLAUDE.md`
- **System Status | 系統狀態**: `SYSTEM_STATUS.md`

### 🔧 Diagnostic Tools | 診斷工具

```bash
# Run comprehensive validation | 運行全面驗證
./validate-deployment.sh

# System diagnosis | 系統診斷
./diagnose.bat  # On Windows
./diagnose.sh   # On Linux (if available)

# Check all services | 檢查所有服務
curl http://localhost:8080/api/health
curl http://localhost:8080/api/status
```

### 📞 Getting Help | 獲取幫助

**If you encounter issues | 如果遇到問題：**

1. **Check logs first | 首先檢查日誌**:
   ```bash
   docker-compose -f cloud-deployment/docker-compose.cloud.yml logs
   ```

2. **Run validation script | 運行驗證腳本**:
   ```bash
   ./validate-deployment.sh
   ```

3. **Check system resources | 檢查系統資源**:
   ```bash
   htop
   df -h
   free -h
   ```

4. **Review documentation | 檢查文件**:
   - Check prerequisite requirements | 檢查前置需求
   - Review troubleshooting section | 查看故障排除部分
   - Verify configuration settings | 驗證配置設定

### 🌟 Advanced Support | 進階支援

**For advanced configuration and customization | 進階配置和自定義：**

- **Environment Configuration | 環境配置**: Modify `cloud-deployment/.env`
- **Docker Settings | Docker 設定**: Adjust `cloud-deployment/docker-compose.cloud.yml`
- **Performance Tuning | 性能調優**: See performance optimization section
- **Security Hardening | 安全加固**: Follow security configuration guide

---

## ✅ Deployment Checklist | 部署檢查清單

**Before going live | 上線前檢查：**

- [ ] ✅ System meets minimum requirements | 系統滿足最低需求
- [ ] ✅ All dependencies installed correctly | 所有依賴正確安裝
- [ ] ✅ Services start without errors | 服務啟動無錯誤
- [ ] ✅ Health checks pass | 健康檢查通過
- [ ] ✅ Web interface accessible | 網頁介面可訪問
- [ ] ✅ API endpoints responding | API 端點響應
- [ ] ✅ Firewall configured | 防火牆已配置
- [ ] ✅ SSL certificate installed (production) | SSL 憑證已安裝（生產環境）
- [ ] ✅ Backup system configured | 備份系統已配置
- [ ] ✅ Monitoring system active | 監控系統已啟動

---

**🎯 This master guide provides everything you need for successful AIFX deployment on Ubuntu servers.**
**📈 Follow the quick deployment for fastest setup, or use manual steps for custom configurations.**
**🚀 Your professional quantitative trading system awaits!**

**🎯 本主要指南提供了在 Ubuntu 伺服器上成功部署 AIFX 所需的一切。**
**📈 遵循快速部署以實現最快設置，或使用手動步驟進行自定義配置。**
**🚀 您的專業量化交易系統等著您！**