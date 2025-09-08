# 🐳 WSL2 上安裝 Docker Desktop 指南

## 🎯 Windows + WSL2 Docker 安裝步驟

### **1. 下載並安裝 Docker Desktop**
1. 前往 https://www.docker.com/products/docker-desktop
2. 下載 "Docker Desktop for Windows"
3. 運行安裝程式 (.exe 檔案)
4. ✅ 確保勾選 "Use WSL 2 instead of Hyper-V"

### **2. 配置 Docker Desktop**
安裝完成後：
1. 啟動 Docker Desktop
2. 進入 Settings → General
3. ✅ 確認 "Use the WSL 2 based engine" 已啟用
4. 進入 Settings → Resources → WSL Integration
5. ✅ 啟用 "Enable integration with my default WSL distro"
6. ✅ 啟用你的 WSL2 分佈 (應該是 Ubuntu)

### **3. 驗證安裝**
在 WSL2 終端執行：
```bash
docker --version
docker-compose --version
docker run hello-world
```

### **4. 記憶體配置 (重要)**
1. Docker Desktop → Settings → Resources → Advanced
2. 設置記憶體至少 4GB (推薦 6GB)
3. CPU 至少 2 個核心
4. 點擊 "Apply & Restart"

### **5. 測試完整 SQL Server 環境**
安裝完成後回到 WSL2 終端：
```bash
cd /mnt/c/Users/butte/OneDrive/桌面/AIFX_CLAUDE
docker-compose -f docker-compose-sqlserver.yml up -d
python test_sqlserver_integration.py
```

---

## ⏱️ 安裝時間估計
- 下載：5-10 分鐘 (根據網速)
- 安裝：5-10 分鐘
- 配置：2-3 分鐘
- **總計：15-25 分鐘**

## 💾 空間需求
- Docker Desktop：約 500MB
- SQL Server 映像：約 1.5GB
- **總計：約 2GB**

## 🔧 故障排除
如果遇到問題：
1. 確保 Windows 版本支援 WSL2
2. 確保 WSL2 已正確安裝和啟用
3. 重啟電腦後再嘗試
4. 檢查防毒軟體是否阻擋

---

# 🚀 安裝完成後的優勢
- ✅ 完整測試 SQL Server Express (免費企業級資料庫)
- ✅ 容器化部署 (生產級環境)
- ✅ 完整的監控和管理工具
- ✅ 真實的雲端部署模擬