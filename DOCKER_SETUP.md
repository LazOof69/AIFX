# AIFX Docker Setup Guide | AIFX Docker 設置指南

## 🐳 Quick Start | 快速開始

### Prerequisites | 前置條件
- **Docker Desktop** installed and running | Docker Desktop 已安裝並運行
- **8GB+ RAM** available for full stack | 8GB+ 記憶體用於完整堆棧
- **10GB+ disk space** for all images | 10GB+ 磁盤空間用於所有映像

### 1. Start Docker Desktop | 啟動 Docker Desktop
1. Open Docker Desktop from Windows Start Menu
2. Wait for Docker to start (green whale icon in system tray)
3. Verify Docker is running: `docker --version`

### 2. Choose Deployment Option | 選擇部署選項

#### Option A: Simple Deployment (Recommended for beginners) | 選項A：簡單部署（推薦初學者）
```bash
# Run the automated script | 運行自動化腳本
./run-docker.sh

# Or manually | 或手動運行
docker-compose -f docker-compose-free.yml up --build -d
```

**Services included | 包含的服務:**
- AIFX Application (Port 8000)
- PostgreSQL Database (Port 5432)
- Redis Cache (Port 6379)
- Grafana Monitoring (Port 3000)

#### Option B: Full Development Stack | 選項B：完整開發堆棧
```bash
# Run the full stack | 運行完整堆棧
docker-compose up --build -d
```

**Services included | 包含的服務:**
- All services from Option A | 選項A的所有服務
- MongoDB (Port 27017)
- Elasticsearch (Port 9200)
- Kibana (Port 5601)
- Prometheus (Port 9090)

#### Option C: Testing Only | 選項C：僅測試
```bash
# Build and run tests | 構建並運行測試
docker build --target testing -t aifx-testing .
docker run --rm aifx-testing
```

## 📊 Service URLs | 服務網址

| Service | URL | Credentials |
|---------|-----|-------------|
| AIFX App | http://localhost:8000 | - |
| Grafana | http://localhost:3000 | admin/admin123 |
| Kibana | http://localhost:5601 | - |
| Prometheus | http://localhost:9090 | - |
| PostgreSQL | localhost:5432 | aifx/password |
| Redis | localhost:6379 | - |
| MongoDB | localhost:27017 | aifx_admin/aifx_mongo_password |

## 🔧 Management Commands | 管理命令

### View Logs | 查看日誌
```bash
# Simple deployment | 簡單部署
docker-compose -f docker-compose-free.yml logs -f aifx-web

# Full stack | 完整堆棧
docker-compose logs -f aifx-app

# Specific service | 特定服務
docker-compose logs -f postgres
```

### Stop Services | 停止服務
```bash
# Simple deployment | 簡單部署
docker-compose -f docker-compose-free.yml down

# Full stack | 完整堆棧
docker-compose down

# Stop and remove volumes | 停止並刪除卷
docker-compose down -v
```

### Restart Services | 重啟服務
```bash
# Simple deployment | 簡單部署
docker-compose -f docker-compose-free.yml restart aifx-web

# Full stack | 完整堆棧
docker-compose restart aifx-app
```

### Scale Services | 擴展服務
```bash
# Scale to 3 app instances | 擴展到3個應用實例
docker-compose up --scale aifx-app=3 -d
```

## 🧪 Testing | 測試

### Run All Tests in Container | 在容器中運行所有測試
```bash
# Build testing image | 構建測試映像
docker build --target testing -t aifx-testing .

# Run tests | 運行測試
docker run --rm -v $(pwd):/workspace aifx-testing
```

### Run Specific Test Files | 運行特定測試文件
```bash
# Run Phase 1 tests | 運行階段1測試
docker run --rm -v $(pwd):/workspace aifx-testing python -m pytest test_phase1_complete.py -v

# Run Phase 2 tests | 運行階段2測試
docker run --rm -v $(pwd):/workspace aifx-testing python -m pytest test_phase2_complete.py -v
```

## 🔍 Troubleshooting | 故障排除

### Common Issues | 常見問題

#### 1. Docker Permission Denied | Docker 權限被拒絕
```bash
# Make sure Docker Desktop is running
# 確保 Docker Desktop 正在運行

# Check Docker status | 檢查 Docker 狀態
docker info
```

#### 2. Port Already in Use | 端口已被使用
```bash
# Find what's using the port | 查找使用端口的進程
netstat -ano | findstr :8000

# Kill the process (Windows) | 終止進程 (Windows)
taskkill /PID <PID> /F

# Or change port in docker-compose.yml | 或在 docker-compose.yml 中更改端口
# ports:
#   - "8001:8000"  # Use port 8001 instead
```

#### 3. Out of Memory | 內存不足
```bash
# Increase Docker Desktop memory limit to 8GB+
# 將 Docker Desktop 內存限制增加到 8GB+

# Or use simple deployment | 或使用簡單部署
docker-compose -f docker-compose-free.yml up -d
```

#### 4. Build Failures | 構建失敗
```bash
# Clean build with no cache | 無緩存清潔構建
docker-compose build --no-cache

# Remove old images | 刪除舊映像
docker system prune -a
```

## 📈 Performance Tuning | 性能調優

### Docker Desktop Settings | Docker Desktop 設置
- **Memory**: 8GB+ for full stack, 4GB+ for simple
- **CPU**: 4+ cores recommended
- **Disk**: Enable file sharing for project directory

### Production Optimizations | 生產優化
```bash
# Use production compose file | 使用生產組合文件
docker-compose -f docker-compose.prod.yml up -d

# Enable resource limits | 啟用資源限制
# (Already configured in compose files)
```

## 🔐 Security Notes | 安全說明

### Development Environment | 開發環境
- Default passwords are used for convenience
- All services exposed on localhost only
- Not suitable for production deployment

### Production Deployment | 生產部署
- Change all default passwords
- Use environment variables for secrets
- Enable TLS/SSL certificates
- Configure firewalls and security groups

## 📝 Development Workflow | 開發工作流程

### 1. Start Development Environment | 啟動開發環境
```bash
# Start with file synchronization | 使用文件同步啟動
docker-compose up -d
```

### 2. Make Code Changes | 進行代碼更改
- Edit files in `src/` directory
- Changes are automatically synced to container
- Application auto-reloads in development mode

### 3. Run Tests | 運行測試
```bash
# Run tests in container | 在容器中運行測試
docker-compose exec aifx-app python -m pytest src/test/ -v
```

### 4. View Logs and Monitoring | 查看日誌和監控
- Application logs: `docker-compose logs -f aifx-app`
- Grafana dashboards: http://localhost:3000
- System metrics: http://localhost:9090

---

## 🆘 Need Help? | 需要幫助？

1. **Check logs first** | 首先檢查日誌: `docker-compose logs`
2. **Verify all services are healthy** | 驗證所有服務健康: `docker-compose ps`
3. **Restart problematic services** | 重啟有問題的服務: `docker-compose restart <service>`
4. **Full reset** | 完全重置: `docker-compose down -v && docker-compose up --build -d`

For more help, check the main README.md or create an issue in the repository.
如需更多幫助，請查看主要的 README.md 或在存儲庫中創建問題。