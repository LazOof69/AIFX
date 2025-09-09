# 🔧 AIFX 依賴安裝指南 | AIFX Dependency Setup Guide

## 📋 **當前狀態 | Current Status**

### ✅ **已測試可用 | Currently Working**
- **性能組件**: 負載生成器和指標計算 | Performance components: Load generator and metrics
- **故障轉移**: 電路熔斷器和健康監控 | Failover: Circuit breaker and health monitoring  
- **配置處理**: YAML配置載入和驗證 | Configuration: YAML loading and validation
- **數據結構**: ForexTick 和相關數據類型 | Data structures: ForexTick and related types
- **基礎架構**: 4,193行代碼，75%品質評級 | Infrastructure: 4,193 lines, 75% quality rating

### ⚠️ **需要依賴 | Requires Dependencies**
- **資料庫整合**: PostgreSQL 和 Redis 連接 | Database integration: PostgreSQL and Redis
- **WebSocket**: 即時數據串流 | Real-time data streaming  
- **監控**: Prometheus 指標收集 | Monitoring: Prometheus metrics

## 🚀 **完整安裝步驟 | Complete Installation Steps**

### **步驟 1: 安裝系統套件 | Step 1: Install System Packages**

```bash
# 安裝 Python 虛擬環境支援
sudo apt update
sudo apt install python3.12-venv python3-dev build-essential

# 驗證安裝
python3 --version  # 應顯示 Python 3.12.3
```

### **步驟 2: 創建虛擬環境 | Step 2: Create Virtual Environment**

```bash
# 創建虛擬環境
python3 -m venv aifx-venv

# 啟動虛擬環境  
source aifx-venv/bin/activate

# 驗證虛擬環境
which python3  # 應顯示 aifx-venv 路徑
```

### **步驟 3: 安裝 Python 依賴 | Step 3: Install Python Dependencies**

```bash
# 必要依賴 (Required dependencies)
pip install websocket-client psycopg2-binary redis pyyaml

# 監控和分析 (Monitoring and analytics)
pip install prometheus_client 

# 可選依賴 (已系統安裝) (Optional - already system-installed)
# numpy pandas matplotlib scipy scikit-learn

# 驗證安裝
python3 -c "
import websocket, psycopg2, redis, yaml, prometheus_client
print('✅ 所有依賴安裝成功!')
"
```

### **步驟 4: 啟動資料庫服務 | Step 4: Start Database Services**

```bash
# PostgreSQL (已運行)
docker ps | grep postgres  # 檢查狀態

# Redis (已運行)  
docker ps | grep redis     # 檢查狀態

# 如果需要重新啟動 (If restart needed):
# docker start aifx-postgres aifx-redis
```

### **步驟 5: 運行完整測試 | Step 5: Run Complete Tests**

```bash
# 啟動虛擬環境
source aifx-venv/bin/activate

# 檢查所有依賴
python3 check_dependencies.py

# 運行完整組件測試
python3 test_components.py

# 運行整合測試
python3 src/test/integration/test_phase4_pipeline_integration.py

# 測試管道協調器
python3 src/main/python/data/pipeline_orchestrator.py
```

## 🧪 **測試場景 | Testing Scenarios**

### **場景 1: 基本功能測試 (無依賴) | Basic Tests (No Dependencies)**
```bash
python3 test_simple.py           # ✅ 4/4 通過
python3 test_phase4_structure.py # ✅ 結構驗證
```

### **場景 2: 組件測試 (部分依賴) | Component Tests (Partial Dependencies)**  
```bash
python3 test_components.py       # 混合結果 - 部分需要依賴
```

### **場景 3: 完整整合測試 (全部依賴) | Full Integration (All Dependencies)**
```bash
# 需要虛擬環境 + 所有依賴
source aifx-venv/bin/activate
python3 src/test/integration/test_phase4_pipeline_integration.py
```

## 🔍 **故障排除 | Troubleshooting**

### **問題 1: 虛擬環境創建失敗**
```
錯誤: ensurepip is not available
解決: sudo apt install python3.12-venv
```

### **問題 2: psycopg2 編譯錯誤**
```
錯誤: pg_config executable not found
解決: sudo apt install libpq-dev python3-dev
```

### **問題 3: 資料庫連接失敗**
```
檢查: docker ps
啟動: docker start aifx-postgres aifx-redis
```

### **問題 4: 權限錯誤**
```
替代方案: 使用 --user 標誌
pip install --user package_name
```

## 📊 **預期結果 | Expected Results**

### **完全安裝後的測試結果:**
- ✅ **check_dependencies.py**: 所有依賴 100% 可用
- ✅ **test_simple.py**: 4/4 基本測試通過  
- ✅ **test_components.py**: 所有組件測試通過
- ✅ **pipeline_orchestrator.py**: 完整管道運行
- ✅ **Prometheus 指標**: http://localhost:8002 可訪問

### **性能指標驗證:**
- ⚡ **延遲**: <50ms P95 延遲時間
- 🔄 **吞吐量**: >100 operations/sec
- 📈 **成功率**: >95% 操作成功率
- 🛡️ **故障轉移**: <30秒自動切換

## 🎯 **下一步 | Next Steps**

1. **手動執行**: 按照步驟 1-5 安裝依賴
2. **測試驗證**: 運行所有測試場景
3. **生產部署**: 配置實際的外匯數據源
4. **監控設置**: 配置 Prometheus + Grafana 儀表板

---

**💡 提示**: 如果遇到權限問題，可以考慮使用 Docker 容器化整個 Python 環境，或者聯繫系統管理員安裝必要的系統套件。