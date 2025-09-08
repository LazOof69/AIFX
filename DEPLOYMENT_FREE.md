# 免費部署指南 | Free Deployment Guide

## 🎯 推薦方案：Railway + PostgreSQL

### 步驟 1：本地測試
```bash
# 使用免費 Docker Compose 配置
docker-compose -f docker-compose-free.yml up -d

# 測試所有功能
python test_phase1_complete.py
python test_integration_phase1_phase2.py
```

### 步驟 2：Railway 部署
1. 註冊 Railway 帳號 (免費 $5/月額度)
2. 連接 GitHub 倉庫
3. 添加 PostgreSQL 服務
4. 設置環境變數：
   ```
   AIFX_ENV=production
   DATABASE_URL=(自動生成)
   ```

### 步驟 3：域名設置 (可選)
- Railway 提供免費子域名
- 或使用 Cloudflare 免費 DNS

## 💰 費用估算

### 免費額度 (每月)
- **Railway**: $5 credit (足夠小型應用)
- **PostgreSQL**: 免費 (在 Railway 額度內)
- **域名**: 免費子域名
- **總計**: $0/月

### 如需擴展 (每月)
- **Railway Pro**: $20/月 (更多資源)
- **自定義域名**: $10-15/年
- **監控服務**: $0 (使用開源方案)
- **總計**: $20-25/月

## 🔄 升級路徑

### 階段 1：免費開發 ($0/月)
- 本地開發 + GitHub Actions
- Railway 免費額度測試

### 階段 2：小規模生產 ($5-20/月)
- Railway 部署
- 基礎監控

### 階段 3：商業化 ($50-100/月)
- 專用 VPS 或雲端服務
- 專業監控和警報
- 備份和災難恢復

## ⚡ 立即開始

```bash
# 克隆倉庫
git clone https://github.com/LazOof69/AIFX.git
cd AIFX

# 本地測試
docker-compose -f docker-compose-free.yml up -d

# 檢查服務
curl http://localhost:8000/health
```

## 📞 支援

如果遇到問題：
1. 檢查 Docker 日誌：`docker-compose logs`
2. 查看 Railway 控制台日誌
3. 使用 GitHub Issues 報告問題