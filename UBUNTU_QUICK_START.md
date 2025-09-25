# AIFX Ubuntu Quick Start Guide | AIFX Ubuntu 快速入門指南

> **🚀 Get AIFX running on your Ubuntu server in under 5 minutes!**
> **🚀 讓 AIFX 在您的 Ubuntu 伺服器上 5 分鐘內運行！**

## 📋 Prerequisites | 前置需求

- Ubuntu 18.04+ server | Ubuntu 18.04+ 伺服器
- Minimum 2GB RAM, 10GB disk space | 最少 2GB 記憶體，10GB 磁碟空間
- Internet connection | 網路連接
- **Sudo privileges OR root access** | **Sudo 權限或 root 訪問權限**

### 🔑 **User Requirements | 用戶需求**
- **Regular User**: Must have sudo privileges | **普通用戶**：必須具有 sudo 權限
- **Root User**: ✅ **FULLY SUPPORTED** with enhanced security mode | **Root 用戶**：✅ **完全支持**配合增強安全模式

## 🚀 One-Command Deployment | 一鍵部署

Copy and paste this single command into your Ubuntu terminal:
複製並貼上此命令到您的 Ubuntu 終端：

```bash
curl -fsSL https://raw.githubusercontent.com/LazOof69/AIFX/main/ubuntu-deploy.sh | bash
```

**That's it! The script will:**
**就是這樣！腳本將會：**

1. ✅ Check system requirements | 檢查系統需求
2. ✅ Install Docker & Docker Compose | 安裝 Docker 和 Docker Compose
3. ✅ Download AIFX from GitHub | 從 GitHub 下載 AIFX
4. ✅ Configure environment | 配置環境
5. ✅ Deploy and start services | 部署並啟動服務
6. ✅ Setup basic firewall | 設置基本防火牆
7. ✅ Perform health checks | 執行健康檢查

## 🌐 Access Your AIFX System | 訪問您的 AIFX 系統

After successful deployment, access your system at:
部署成功後，在以下地址訪問您的系統：

- **Web Interface | 網頁介面**: `http://YOUR_SERVER_IP:8080`
- **API Documentation | API 文檔**: `http://YOUR_SERVER_IP:8080/docs`
- **Health Check | 健康檢查**: `http://YOUR_SERVER_IP:8080/api/health`

Replace `YOUR_SERVER_IP` with your actual server IP address.
將 `YOUR_SERVER_IP` 替換為您實際的伺服器 IP 地址。

## 🛠️ Manual Step-by-Step (if needed) | 手動步驟（如需要）

If you prefer manual installation or the one-click script fails:
如果您偏好手動安裝或一鍵腳本失敗：

### Step 1: Update System | 步驟 1：更新系統
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget
```

### Step 2: Install Docker | 步驟 2：安裝 Docker
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

### Step 3: Download AIFX | 步驟 3：下載 AIFX
```bash
git clone https://github.com/LazOof69/AIFX.git
cd AIFX/cloud-deployment
```

### Step 4: Deploy | 步驟 4：部署
```bash
cp .env.cloud .env
chmod +x deploy-cloud.sh
./deploy-cloud.sh
```

## 🔧 Common Management Commands | 常用管理命令

### View Logs | 查看日誌
```bash
cd AIFX/cloud-deployment
docker-compose -f docker-compose.cloud.yml logs -f
```

### Stop Service | 停止服務
```bash
docker-compose -f docker-compose.cloud.yml down
```

### Restart Service | 重啟服務
```bash
docker-compose -f docker-compose.cloud.yml restart
```

### Update AIFX | 更新 AIFX
```bash
cd AIFX
git pull origin main
cd cloud-deployment
docker-compose -f docker-compose.cloud.yml down
docker-compose -f docker-compose.cloud.yml up -d --build
```

### Check Status | 檢查狀態
```bash
docker-compose -f docker-compose.cloud.yml ps
curl http://localhost:8080/api/health
```

## 🚨 Troubleshooting | 故障排除

### Permission Error with Docker | Docker 權限錯誤
```bash
sudo usermod -aG docker $USER
newgrp docker
# OR logout and login again | 或登出後重新登入
```

### Port 8080 Already in Use | 端口 8080 被佔用
```bash
# Check what's using the port | 檢查什麼在使用該端口
sudo netstat -tulpn | grep :8080

# Change port in .env file | 在 .env 文件中更改端口
echo "AIFX_WEB_PORT=8081" >> .env
docker-compose -f docker-compose.cloud.yml up -d
```

### Service Won't Start | 服務無法啟動
```bash
# Check logs | 查看日誌
docker-compose -f docker-compose.cloud.yml logs

# Clean up and restart | 清理並重啟
docker-compose -f docker-compose.cloud.yml down
docker system prune -f
docker-compose -f docker-compose.cloud.yml up -d --build
```

### Out of Memory | 記憶體不足
```bash
# Reduce workers | 減少工作進程
echo "WORKERS=1" >> .env
docker-compose -f docker-compose.cloud.yml restart
```

## 🔒 Security Setup (Recommended) | 安全設置（建議）

### Enable Firewall | 啟用防火牆
```bash
sudo ufw allow ssh
sudo ufw allow 8080/tcp
sudo ufw enable
```

### Setup SSL with Nginx (Production) | 使用 Nginx 設置 SSL（生產環境）
```bash
# Install Nginx
sudo apt install -y nginx certbot python3-certbot-nginx

# Create Nginx config
sudo nano /etc/nginx/sites-available/aifx

# Add this configuration:
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

# Enable site
sudo ln -s /etc/nginx/sites-available/aifx /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com
```

## 📊 Performance Optimization | 性能優化

### For High-Performance Servers | 高性能伺服器
```bash
echo "WORKERS=4" >> .env
echo "MAX_CONNECTIONS=200" >> .env
docker-compose -f docker-compose.cloud.yml restart
```

### For Low-Resource Servers | 低資源伺服器
```bash
echo "WORKERS=1" >> .env
echo "MAX_CONNECTIONS=50" >> .env
docker-compose -f docker-compose.cloud.yml restart
```

## 📈 Monitoring | 監控

### System Resources | 系統資源
```bash
# Monitor system
htop

# Monitor Docker containers
docker stats

# Check disk usage
df -h
```

### Application Monitoring | 應用程式監控
```bash
# Health check
curl http://localhost:8080/api/health

# Get system status
curl http://localhost:8080/api/status

# View recent logs
docker-compose -f docker-compose.cloud.yml logs --tail=50
```

## 🔄 Backup and Recovery | 備份與恢復

### Create Backup | 創建備份
```bash
# Backup application data
docker run --rm -v aifx_data:/data -v $(pwd):/backup alpine tar czf /backup/aifx_backup_$(date +%Y%m%d).tar.gz -C /data .

# Backup configuration
tar czf aifx_config_$(date +%Y%m%d).tar.gz AIFX/cloud-deployment/.env
```

### Restore Backup | 恢復備份
```bash
# Stop services
docker-compose -f docker-compose.cloud.yml down

# Restore data
docker run --rm -v aifx_data:/data -v $(pwd):/backup alpine tar xzf /backup/aifx_backup_YYYYMMDD.tar.gz -C /data

# Start services
docker-compose -f docker-compose.cloud.yml up -d
```

## 🆘 Support | 支援

If you encounter issues:
如果遇到問題：

1. Check the logs first | 首先檢查日誌
2. Review the troubleshooting section | 檢查故障排除部分
3. Ensure all prerequisites are met | 確保滿足所有前置需求
4. Try the manual installation steps | 嘗試手動安裝步驟

---

**🎯 Your AIFX trading system should now be running successfully on Ubuntu!**
**🎯 您的 AIFX 交易系統現在應該在 Ubuntu 上成功運行了！**