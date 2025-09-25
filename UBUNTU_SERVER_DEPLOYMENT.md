# AIFX Ubuntu Server Deployment Guide | AIFX Ubuntu 伺服器部署指南

> **🚀 Complete deployment guide for Ubuntu servers using Git and Docker**
> **🚀 使用 Git 和 Docker 的 Ubuntu 伺服器完整部署指南**

## 📋 Quick Start (One-Command Deployment) | 快速開始（一鍵部署）

```bash
# Download and run the one-click deployment script
# 下載並運行一鍵部署腳本
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/AIFX/main/ubuntu-deploy.sh | bash
```

## 🔧 Manual Deployment Steps | 手動部署步驟

### Step 1: System Preparation | 步驟一：系統準備

```bash
# Update system packages | 更新系統套件
sudo apt update && sudo apt upgrade -y

# Install essential packages | 安裝基本套件
sudo apt install -y curl wget git unzip software-properties-common apt-transport-https ca-certificates gnupg lsb-release

# Check system requirements | 檢查系統需求
echo "🔍 System Information:"
echo "OS: $(lsb_release -d | cut -f2)"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $4}') available"
echo "CPU: $(nproc) cores"
```

### Step 2: Install Docker | 步驟二：安裝 Docker

```bash
# Remove old Docker versions | 移除舊版 Docker
sudo apt remove -y docker docker-engine docker.io containerd runc

# Add Docker GPG key | 添加 Docker GPG 密鑰
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository | 添加 Docker 倉庫
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package index | 更新套件索引
sudo apt update

# Install Docker Engine | 安裝 Docker 引擎
sudo apt install -y docker-ce docker-ce-cli containerd.io

# Add current user to docker group | 將當前用戶添加到 docker 組
sudo usermod -aG docker $USER

# Start and enable Docker | 啟動並啟用 Docker
sudo systemctl start docker
sudo systemctl enable docker

# Install Docker Compose | 安裝 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation | 驗證安裝
docker --version
docker-compose --version
```

### Step 3: Download AIFX from GitHub | 步驟三：從 GitHub 下載 AIFX

```bash
# Clone the AIFX repository | 克隆 AIFX 倉庫
git clone https://github.com/YOUR_USERNAME/AIFX.git

# Navigate to project directory | 導航到項目目錄
cd AIFX

# Check project structure | 檢查項目結構
ls -la

# Verify cloud deployment package | 驗證雲端部署包
ls -la cloud-deployment/
```

### Step 4: Configure Environment | 步驟四：配置環境

```bash
# Navigate to cloud deployment directory | 導航到雲端部署目錄
cd cloud-deployment

# Copy environment template | 複製環境模板
cp .env.cloud .env

# Edit environment configuration (optional) | 編輯環境配置（可選）
nano .env

# Make deployment script executable | 使部署腳本可執行
chmod +x deploy-cloud.sh
```

### Step 5: Deploy AIFX | 步驟五：部署 AIFX

```bash
# Run the deployment script | 運行部署腳本
./deploy-cloud.sh

# OR manually deploy | 或手動部署
docker-compose -f docker-compose.cloud.yml up -d --build
```

### Step 6: Verify Deployment | 步驟六：驗證部署

```bash
# Check container status | 檢查容器狀態
docker-compose -f docker-compose.cloud.yml ps

# Check logs | 查看日誌
docker-compose -f docker-compose.cloud.yml logs -f

# Test health endpoint | 測試健康端點
curl -f http://localhost:8080/api/health

# Test web interface | 測試網頁介面
curl -I http://localhost:8080
```

## 🔥 One-Click Deployment Script | 一鍵部署腳本

Create and run the optimized deployment script:
創建並運行優化的部署腳本：

```bash
# Download the optimized deployment script | 下載優化的部署腳本
wget https://raw.githubusercontent.com/YOUR_USERNAME/AIFX/main/ubuntu-deploy.sh

# Make it executable | 使其可執行
chmod +x ubuntu-deploy.sh

# Run deployment | 運行部署
./ubuntu-deploy.sh
```

## 🌐 Accessing AIFX | 訪問 AIFX

After successful deployment:
部署成功後：

- **Web Interface | 網頁介面**: `http://YOUR_SERVER_IP:8080`
- **API Documentation | API 文件**: `http://YOUR_SERVER_IP:8080/docs`
- **Health Check | 健康檢查**: `http://YOUR_SERVER_IP:8080/api/health`
- **Trading Signals | 交易信號**: `http://YOUR_SERVER_IP:8080/api/signals`

## 🛠️ Management Commands | 管理命令

```bash
# View logs | 查看日誌
cd AIFX/cloud-deployment
docker-compose -f docker-compose.cloud.yml logs -f

# Stop service | 停止服務
docker-compose -f docker-compose.cloud.yml down

# Restart service | 重新啟動服務
docker-compose -f docker-compose.cloud.yml restart

# Update service | 更新服務
git pull origin main
docker-compose -f docker-compose.cloud.yml down
docker-compose -f docker-compose.cloud.yml up -d --build

# Scale service | 擴展服務
docker-compose -f docker-compose.cloud.yml up -d --scale aifx-trading-system=2
```

## 🔒 Security Configuration | 安全配置

### Firewall Setup | 防火牆設置

```bash
# Install and configure UFW | 安裝並配置 UFW
sudo apt install -y ufw

# Set default policies | 設置預設策略
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH | 允許 SSH
sudo ufw allow ssh

# Allow AIFX web interface | 允許 AIFX 網頁介面
sudo ufw allow 8080/tcp

# Enable firewall | 啟用防火牆
sudo ufw enable

# Check status | 檢查狀態
sudo ufw status
```

### SSL/HTTPS Setup with Nginx | 使用 Nginx 設置 SSL/HTTPS

```bash
# Install Nginx | 安裝 Nginx
sudo apt install -y nginx

# Install Certbot for Let's Encrypt | 安裝 Let's Encrypt 的 Certbot
sudo apt install -y certbot python3-certbot-nginx

# Create Nginx configuration | 創建 Nginx 配置
sudo tee /etc/nginx/sites-available/aifx << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# Enable the site | 啟用站點
sudo ln -s /etc/nginx/sites-available/aifx /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get SSL certificate | 獲取 SSL 證書
sudo certbot --nginx -d your-domain.com
```

## 📊 Monitoring & Maintenance | 監控與維護

### System Monitoring | 系統監控

```bash
# Install monitoring tools | 安裝監控工具
sudo apt install -y htop iotop netstat-nat

# Monitor system resources | 監控系統資源
htop

# Monitor Docker containers | 監控 Docker 容器
docker stats

# Check disk usage | 檢查磁碟使用
df -h

# Check memory usage | 檢查記憶體使用
free -h
```

### Log Management | 日誌管理

```bash
# Configure log rotation | 配置日誌輪轉
sudo tee /etc/logrotate.d/aifx << 'EOF'
/var/lib/docker/containers/*/*-json.log {
    rotate 7
    daily
    compress
    size 10M
    missingok
    delaycompress
    copytruncate
}
EOF

# View application logs | 查看應用程式日誌
cd AIFX/cloud-deployment
docker-compose -f docker-compose.cloud.yml logs --tail=100 -f
```

### Automated Backups | 自動備份

```bash
# Create backup script | 創建備份腳本
sudo tee /usr/local/bin/aifx-backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/opt/aifx-backups"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Backup Docker volumes | 備份 Docker 卷
docker run --rm -v aifx_data:/data -v $BACKUP_DIR:/backup alpine tar czf /backup/aifx_data_$DATE.tar.gz -C /data .

# Keep only last 7 backups | 僅保留最後 7 個備份
find $BACKUP_DIR -name "aifx_data_*.tar.gz" -mtime +7 -delete

echo "Backup completed: aifx_data_$DATE.tar.gz"
EOF

chmod +x /usr/local/bin/aifx-backup.sh

# Set up daily backup cron job | 設置每日備份 cron 作業
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/aifx-backup.sh") | crontab -
```

## 🚨 Troubleshooting | 故障排除

### Common Issues | 常見問題

#### Docker Permission Denied | Docker 權限被拒
```bash
# Add user to docker group and restart | 將用戶添加到 docker 組並重啟
sudo usermod -aG docker $USER
newgrp docker
# OR logout and login again | 或登出後重新登入
```

#### Port 8080 Already in Use | 端口 8080 被佔用
```bash
# Check what's using the port | 檢查什麼在使用該端口
sudo netstat -tulpn | grep :8080
sudo lsof -i :8080

# Kill the process or change port in .env | 結束進程或在 .env 中更改端口
echo "AIFX_WEB_PORT=8081" >> .env
```

#### Container Won't Start | 容器無法啟動
```bash
# Check container logs | 檢查容器日誌
docker-compose -f docker-compose.cloud.yml logs

# Remove and rebuild | 移除並重建
docker-compose -f docker-compose.cloud.yml down
docker system prune -f
docker-compose -f docker-compose.cloud.yml up -d --build
```

#### Out of Disk Space | 磁碟空間不足
```bash
# Clean up Docker system | 清理 Docker 系統
docker system prune -a --volumes

# Remove old containers and images | 移除舊容器和映像
docker container prune -f
docker image prune -a -f
```

### Performance Optimization | 性能優化

```bash
# For high-performance servers | 針對高性能伺服器
echo "WORKERS=4" >> .env
echo "MAX_CONNECTIONS=200" >> .env

# For low-resource servers | 針對低資源伺服器
echo "WORKERS=1" >> .env
echo "MAX_CONNECTIONS=50" >> .env

# Restart to apply changes | 重啟以應用更改
docker-compose -f docker-compose.cloud.yml restart
```

## ✅ Deployment Checklist | 部署檢查清單

- [ ] System updated and packages installed | 系統更新且套件已安裝
- [ ] Docker and Docker Compose installed | Docker 和 Docker Compose 已安裝
- [ ] AIFX repository cloned | AIFX 倉庫已克隆
- [ ] Environment configured | 環境已配置
- [ ] Service deployed and running | 服務已部署且運行中
- [ ] Health check passes | 健康檢查通過
- [ ] Web interface accessible | 網頁介面可訪問
- [ ] Firewall configured | 防火牆已配置
- [ ] SSL certificate installed (if needed) | SSL 證書已安裝（如需要）
- [ ] Monitoring and logging set up | 監控和日誌已設置
- [ ] Backup system configured | 備份系統已配置

---

**🎯 Your AIFX trading system is now ready for production on Ubuntu server!**
**🎯 您的 AIFX 交易系統現在已準備好在 Ubuntu 伺服器上投入生產！**