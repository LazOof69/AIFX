# AIFX Cloud Deployment Package | AIFX 雲端部署包

> **🚀 Ready-to-deploy AIFX Trading System for Cloud Virtual Machines**
> **🚀 針對雲端虛擬機器的即部署 AIFX 交易系統**

## 📋 Contents | 內容

This cloud deployment package contains everything needed to deploy AIFX on cloud platforms:
此雲端部署包含在雲端平台部署 AIFX 所需的一切：

- `docker-compose.cloud.yml` - Optimized Docker configuration for cloud deployment | 針對雲端部署優化的 Docker 配置
- `.env.cloud` - Environment configuration template | 環境配置模板
- `deploy-cloud.sh` - Linux/macOS deployment script | Linux/macOS 部署腳本
- `deploy-cloud.bat` - Windows deployment script | Windows 部署腳本
- `README.md` - This documentation | 本文檔

## 🎯 Quick Start | 快速開始

### Prerequisites | 前置需求
- Docker & Docker Compose installed | 已安裝 Docker 和 Docker Compose
- Minimum 2GB RAM, 10GB disk space | 最少 2GB 記憶體，10GB 磁碟空間
- Port 8080 available | 端口 8080 可用

### 🐧 Linux/macOS Deployment | Linux/macOS 部署

```bash
# 1. Copy this cloud-deployment folder to your cloud VM
# 1. 將此 cloud-deployment 資料夾複製到您的雲端虛擬機器

# 2. Navigate to the deployment directory
# 2. 導航到部署目錄
cd cloud-deployment

# 3. Run the deployment script
# 3. 運行部署腳本
./deploy-cloud.sh
```

### 🪟 Windows Deployment | Windows 部署

```cmd
REM 1. Copy this cloud-deployment folder to your cloud VM
REM 1. 將此 cloud-deployment 資料夾複製到您的雲端虛擬機器

REM 2. Navigate to the deployment directory
REM 2. 導航到部署目錄
cd cloud-deployment

REM 3. Run the deployment script
REM 3. 運行部署腳本
deploy-cloud.bat
```

## 🔧 Configuration | 配置

### Environment Variables | 環境變數

The `.env.cloud` file contains all configuration options:
`.env.cloud` 文件包含所有配置選項：

```bash
# Application Settings | 應用程式設定
AIFX_WEB_PORT=8080          # Web interface port | 網頁介面端口
AIFX_LOG_LEVEL=INFO         # Logging level | 日誌級別
AIFX_DEBUG=false            # Debug mode | 調試模式

# Trading Configuration | 交易配置
TRADING_ENABLED=false       # Enable live trading | 啟用實時交易
TRADING_MODE=demo           # Trading mode (demo/live) | 交易模式
DEFAULT_CURRENCY_PAIRS=EUR/USD,USD/JPY  # Default pairs | 預設貨幣對

# Performance Settings | 性能設定
WORKERS=2                   # Number of workers | 工作進程數
MAX_CONNECTIONS=100         # Max connections | 最大連接數
CACHE_TTL=300              # Cache TTL in seconds | 緩存過期時間（秒）
```

### Security Configuration | 安全配置

For production deployment, update these settings:
生產部署時，請更新這些設定：

```bash
# Security Settings | 安全設定
ALLOWED_ORIGINS=https://yourdomain.com  # Your domain | 您的域名
API_KEY_REQUIRED=true                   # Require API key | 需要 API 金鑰
API_KEY=your-secure-api-key-here        # Your API key | 您的 API 金鑰
JWT_SECRET=your-jwt-secret-here         # JWT secret | JWT 密鑰
```

## 🌐 Cloud Platform Specific Setup | 雲端平台特定設置

### AWS EC2 | AWS EC2

```bash
# 1. Launch EC2 instance (t3.medium or larger recommended)
# 1. 啟動 EC2 實例（建議 t3.medium 或更大）

# 2. Install Docker
sudo yum update -y
sudo yum install docker -y
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker ec2-user

# 3. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 4. Deploy AIFX
./deploy-cloud.sh
```

### Google Cloud Platform (GCP) | Google Cloud Platform (GCP)

```bash
# 1. Create VM instance (e2-standard-2 or larger recommended)
# 1. 創建 VM 實例（建議 e2-standard-2 或更大）

# 2. Install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker $USER

# 3. Deploy AIFX
./deploy-cloud.sh
```

### Microsoft Azure | Microsoft Azure

```bash
# 1. Create VM (Standard_B2s or larger recommended)
# 1. 創建 VM（建議 Standard_B2s 或更大）

# 2. Install Docker
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -a -G docker $USER

# 3. Deploy AIFX
./deploy-cloud.sh
```

## 📱 Access URLs | 訪問網址

After successful deployment:
部署成功後：

- **Web Interface | 網頁介面**: `http://your-server-ip:8080`
- **Health Check | 健康檢查**: `http://your-server-ip:8080/api/health`
- **API Documentation | API 文件**: `http://your-server-ip:8080/docs`
- **Trading Signals | 交易信號**: `http://your-server-ip:8080/api/signals`

## 🛠️ Management Commands | 管理命令

### View Logs | 查看日誌
```bash
docker-compose -f docker-compose.cloud.yml logs -f
```

### Stop Service | 停止服務
```bash
docker-compose -f docker-compose.cloud.yml down
```

### Restart Service | 重新啟動服務
```bash
docker-compose -f docker-compose.cloud.yml restart
```

### Update Service | 更新服務
```bash
docker-compose -f docker-compose.cloud.yml pull
docker-compose -f docker-compose.cloud.yml up -d
```

### Scale Service | 擴展服務
```bash
docker-compose -f docker-compose.cloud.yml up -d --scale aifx-trading-system=2
```

## 🔍 Monitoring & Troubleshooting | 監控與故障排除

### Health Check | 健康檢查
```bash
curl -f http://localhost:8080/api/health
```

### Container Status | 容器狀態
```bash
docker-compose -f docker-compose.cloud.yml ps
```

### Resource Usage | 資源使用
```bash
docker stats
```

### Common Issues | 常見問題

#### Service Not Starting | 服務無法啟動
```bash
# Check logs
docker-compose -f docker-compose.cloud.yml logs

# Check container status
docker ps -a

# Restart with clean slate
docker-compose -f docker-compose.cloud.yml down
docker system prune -f
./deploy-cloud.sh
```

#### Port Already in Use | 端口被佔用
```bash
# Check what's using port 8080
netstat -tulpn | grep 8080

# Change port in .env file
echo "AIFX_WEB_PORT=8081" >> .env

# Restart service
docker-compose -f docker-compose.cloud.yml up -d
```

#### Out of Memory | 記憶體不足
```bash
# Check memory usage
free -h

# Reduce workers in .env
echo "WORKERS=1" >> .env

# Update resource limits in docker-compose.cloud.yml
```

## 🔐 Production Security Checklist | 生產安全檢查清單

- [ ] Update default passwords and API keys | 更新預設密碼和 API 金鑰
- [ ] Configure proper CORS origins | 配置適當的 CORS 來源
- [ ] Enable HTTPS with reverse proxy | 使用反向代理啟用 HTTPS
- [ ] Set up firewall rules | 設置防火牆規則
- [ ] Configure log rotation | 配置日誌輪轉
- [ ] Set up monitoring and alerting | 設置監控和警報
- [ ] Regular security updates | 定期安全更新

## 🚀 Performance Optimization | 性能優化

### For High Traffic | 針對高流量
```yaml
# In docker-compose.cloud.yml
environment:
  - WORKERS=4
  - MAX_CONNECTIONS=200
  - CACHE_TTL=600

deploy:
  resources:
    limits:
      cpus: '4.0'
      memory: 4G
```

### For Low Resource | 針對低資源
```yaml
# In docker-compose.cloud.yml
environment:
  - WORKERS=1
  - MAX_CONNECTIONS=50
  - CACHE_TTL=300

deploy:
  resources:
    limits:
      cpus: '1.0'
      memory: 1G
```

## 📊 Supported Features | 支援功能

This cloud deployment includes:
此雲端部署包括：

- ✅ **Real-time Trading Signals** | 即時交易信號
- ✅ **AI-Enhanced Analysis** | AI 增強分析
- ✅ **Multi-currency Support** | 多貨幣支援
- ✅ **Risk Management** | 風險管理
- ✅ **Web Dashboard** | 網頁儀表板
- ✅ **REST API** | REST API
- ✅ **Health Monitoring** | 健康監控
- ✅ **Auto-scaling Ready** | 自動擴展就緒

## 📞 Support | 支援

For issues and support:
如遇問題和支援：

- Check logs first: `docker-compose -f docker-compose.cloud.yml logs`
- Review troubleshooting section above
- Ensure all prerequisites are met

## 📄 License | 授權

AIFX Trading System - Professional quantitative trading platform
AIFX 交易系統 - 專業量化交易平台

---

**🎯 Ready for production deployment! | 生產部署就緒！**