# 🔷 AIFX SQL Server 部署指南 | SQL Server Deployment Guide

## 💰 **SQL Server 費用說明**

### **免費選項 (推薦)**
- **SQL Server Express**: 完全免費
  - 資料庫大小限制：10 GB (足夠大部分應用)
  - 記憶體限制：1410 MB
  - CPU 限制：1 個處理器，4 核心
  - ✅ **適合 AIFX：完全滿足需求**

### **雲端選項**
- **Azure SQL Database**: 每月 $5 起 (Basic 層)
- **AWS RDS SQL Server Express**: 每月 $13 起
- **Google Cloud SQL Server**: 每月 $10 起

---

## 🚀 **快速開始**

### **步驟 1：啟動 SQL Server 環境**
```bash
# 啟動完整的 SQL Server 環境
docker-compose -f docker-compose-sqlserver.yml up -d

# 檢查服務狀態
docker-compose -f docker-compose-sqlserver.yml ps
```

### **步驟 2：驗證安裝**
```bash
# 測試資料庫連接和功能
python test_sqlserver_integration.py
```

### **步驟 3：訪問管理介面**
- **Adminer (資料庫管理)**: http://localhost:8080
  - 系統：SQL Server
  - 伺服器：sqlserver
  - 用戶名：sa
  - 密碼：YourStrongPassword123!
  - 資料庫：aifx

- **Grafana (監控儀表板)**: http://localhost:3000
  - 用戶名：admin
  - 密碼：admin123

---

## ⚙️ **環境變數配置**

### **SQL Server 設定**
```bash
# SQL Server 連接設定
export SQLSERVER_HOST=localhost
export SQLSERVER_DATABASE=aifx
export SQLSERVER_USERNAME=sa
export SQLSERVER_PASSWORD=YourStrongPassword123!
export SQLSERVER_DRIVER="ODBC Driver 17 for SQL Server"

# 應用程式設定
export AIFX_ENV=production
export REDIS_HOST=redis
export REDIS_PORT=6379
```

### **Docker Compose 環境文件 (.env)**
```bash
# 創建 .env 文件
cat > .env << 'EOF'
SQLSERVER_SA_PASSWORD=YourStrongPassword123!
AIFX_ENV=production
GRAFANA_ADMIN_PASSWORD=admin123
EOF
```

---

## 📊 **資料庫架構**

### **主要資料表**

#### **1. trading_data_eurusd / trading_data_usdjpy**
```sql
-- 交易數據表
CREATE TABLE trading_data_eurusd (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    datetime DATETIME2 NOT NULL,
    open_price DECIMAL(18,6) NOT NULL,
    high_price DECIMAL(18,6) NOT NULL,
    low_price DECIMAL(18,6) NOT NULL,
    close_price DECIMAL(18,6) NOT NULL,
    volume BIGINT NOT NULL DEFAULT 0,
    created_at DATETIME2 DEFAULT GETDATE(),
    updated_at DATETIME2 DEFAULT GETDATE()
);
```

#### **2. trading_signals**
```sql
-- 交易信號表
CREATE TABLE trading_signals (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    symbol NVARCHAR(20) NOT NULL,
    datetime DATETIME2 NOT NULL,
    signal_type NVARCHAR(10) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    strength DECIMAL(5,4) NOT NULL,
    source NVARCHAR(50) NOT NULL,
    metadata NVARCHAR(MAX)
);
```

#### **3. model_performance**
```sql
-- 模型性能表
CREATE TABLE model_performance (
    id BIGINT IDENTITY(1,1) PRIMARY KEY,
    model_name NVARCHAR(100) NOT NULL,
    symbol NVARCHAR(20) NOT NULL,
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    training_date DATETIME2 NOT NULL,
    evaluation_date DATETIME2 NOT NULL
);
```

---

## 🔧 **使用範例**

### **Python 代碼示例**
```python
from src.main.python.utils.database import DatabaseManager, save_trading_data

# 初始化資料庫管理器
db_manager = DatabaseManager()

# 測試連接
if db_manager.test_connection():
    print("SQL Server 連接成功！")

# 保存交易數據
import pandas as pd
df = pd.DataFrame({
    'datetime': pd.date_range('2024-01-01', periods=100, freq='H'),
    'open_price': [1.1000] * 100,
    'high_price': [1.1010] * 100,
    'low_price': [1.0990] * 100,
    'close_price': [1.1005] * 100,
    'volume': [1000] * 100
})

save_trading_data(df, "EURUSD")
print("數據保存成功！")
```

### **SQL 查詢示例**
```sql
-- 查看最新的 EURUSD 數據
SELECT TOP 10 *
FROM trading_data_eurusd
ORDER BY datetime DESC;

-- 查看最新交易信號
SELECT *
FROM v_latest_signals
WHERE symbol = 'EURUSD';

-- 查看模型性能
SELECT *
FROM model_performance
WHERE model_name = 'XGBoost'
ORDER BY evaluation_date DESC;
```

---

## 🚨 **故障排除**

### **常見問題**

#### **1. 連接失敗**
```bash
# 檢查 SQL Server 容器狀態
docker logs aifx_sqlserver_1

# 檢查端口是否開放
netstat -an | grep 1433

# 測試連接
docker exec -it aifx_sqlserver_1 /opt/mssql-tools/bin/sqlcmd -S localhost -U sa -P YourStrongPassword123!
```

#### **2. ODBC 驅動問題**
```bash
# 檢查 ODBC 驅動安裝
odbcinst -q -d

# 重新構建 Docker 映像
docker-compose -f docker-compose-sqlserver.yml build --no-cache aifx-web
```

#### **3. 權限問題**
```sql
-- 授予資料庫權限
USE aifx;
GO
GRANT ALL ON SCHEMA::dbo TO sa;
GO
```

---

## 📈 **性能優化**

### **索引建議**
```sql
-- 為查詢優化創建索引
CREATE INDEX IX_trading_data_eurusd_datetime 
ON trading_data_eurusd(datetime);

CREATE INDEX IX_trading_signals_symbol_datetime 
ON trading_signals(symbol, datetime);
```

### **記憶體調整**
```yaml
# docker-compose-sqlserver.yml 中調整
deploy:
  resources:
    limits:
      memory: 2G  # SQL Server Express 最大限制
    reservations:
      memory: 1G
```

---

## 🔄 **備份和恢復**

### **自動備份腳本**
```sql
-- 創建資料庫備份
BACKUP DATABASE aifx 
TO DISK = '/var/opt/mssql/backup/aifx.bak'
WITH FORMAT, INIT;
```

### **恢復資料庫**
```sql
-- 恢復資料庫
RESTORE DATABASE aifx 
FROM DISK = '/var/opt/mssql/backup/aifx.bak'
WITH REPLACE;
```

---

## 🎯 **總結**

✅ **優點**：
- **完全免費** (SQL Server Express)
- **企業級功能** (事務、索引、存儲過程)
- **Microsoft 生態整合** (Azure、Power BI)
- **強大查詢性能**
- **完整的 ACID 支援**

⚠️ **限制**：
- 資料庫大小限制 10 GB
- 記憶體限制 1410 MB
- CPU 限制 4 核心

**🎉 對於 AIFX 來說，SQL Server Express 完全滿足需求，是一個優秀的免費資料庫選擇！**