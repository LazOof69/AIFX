# AIFX Phase 1 Testing Guide | AIFX第一階段測試指南

This guide provides step-by-step instructions for testing the AIFX Phase 1 infrastructure.  
本指南提供測試AIFX第一階段基礎設施的詳細說明。

## 🚀 Quick Start | 快速開始

### Step 1: Environment Check | 步驟1：環境檢查

First, verify your environment is ready:  
首先，驗證您的環境是否準備就緒：

```bash
# Basic environment check | 基礎環境檢查
python check_environment.py

# With verbose output | 詳細輸出
python check_environment.py --verbose

# Auto-install missing packages | 自動安裝缺失的包
python check_environment.py --install
```

**Expected Results | 預期結果:**
- ✅ Python 3.8+ detected | 檢測到Python 3.8+
- ✅ All required packages installed | 所有必需包已安裝
- ✅ Project structure complete | 項目結構完整
- ✅ System capabilities sufficient | 系統能力充足

### Step 2: Comprehensive Testing | 步驟2：綜合測試

Run the complete Phase 1 test suite:  
運行完整的第一階段測試套件：

```bash
# Run all Phase 1 tests | 運行所有第一階段測試
python test_phase1_complete.py
```

**Test Coverage | 測試覆蓋範圍:**
1. **Environment Setup** | **環境設置** - Python version, project structure
2. **Dependencies** | **依賴項** - Required and optional packages
3. **Configuration System** | **配置系統** - Config loading and validation
4. **Logging System** | **日誌系統** - Structured logging functionality
5. **Data Pipeline** | **數據管道** - Data loading and validation
6. **Data Preprocessing** | **數據預處理** - Feature engineering pipeline
7. **Technical Indicators** | **技術指標** - All 50+ indicators
8. **Unit Testing** | **單元測試** - Test framework functionality
9. **Integration** | **集成** - End-to-end pipeline testing

### Step 3: Unit Tests (Optional) | 步驟3：單元測試（可選）

Run specific unit tests for detailed validation:  
運行特定單元測試以進行詳細驗證：

```bash
# Run all unit tests | 運行所有單元測試
python -m pytest src/test/ -v

# Run specific test file | 運行特定測試文件
python -m pytest src/test/unit/test_data_loader.py -v

# Run with coverage | 運行並顯示覆蓋率
python -m pytest src/test/ --cov=src/main/python --cov-report=html
```

## 📊 Understanding Test Results | 理解測試結果

### Test Status Indicators | 測試狀態指標

- ✅ **PASS** - Component working correctly | 組件工作正常
- ❌ **FAIL** - Component has issues | 組件有問題
- ⚠️ **WARNING** - Minor issues or optional features | 輕微問題或可選功能

### Pass Rate Interpretation | 通過率解釋

- **90-100%** - 🟢 **EXCELLENT** - Ready for Phase 2 | 準備進入第二階段
- **80-89%** - 🟡 **GOOD** - Minor issues, mostly ready | 輕微問題，基本準備就緒
- **60-79%** - 🟡 **ACCEPTABLE** - Some issues to address | 有一些問題需要解決
- **<60%** - 🔴 **NEEDS WORK** - Significant issues | 重大問題

## 🔧 Common Issues and Solutions | 常見問題及解決方案

### Issue 1: Missing Dependencies | 問題1：缺少依賴項

**Problem | 問題:** Package not found errors  
**Solution | 解決方案:**

```bash
# Install from requirements.txt | 從requirements.txt安裝
pip install -r requirements.txt

# Or use the auto-installer | 或使用自動安裝器
python check_environment.py --install

# Install specific package | 安裝特定包
pip install package_name
```

### Issue 2: Network Connection Issues | 問題2：網絡連接問題

**Problem | 問題:** Data download tests fail  
**Solution | 解決方案:**

```bash
# Skip network tests | 跳過網絡測試
export SKIP_NETWORK_TESTS=true
python test_phase1_complete.py

# Or run offline-only tests | 或只運行離線測試
python -m pytest src/test/ -m "not network"
```

### Issue 3: Memory Issues | 問題3：內存問題

**Problem | 問題:** Tests fail due to insufficient memory  
**Solution | 解決方案:**

- Close other applications | 關閉其他應用程序
- Use smaller data samples | 使用更小的數據樣本
- Run tests individually | 單獨運行測試

### Issue 4: Permission Issues | 問題4：權限問題

**Problem | 問題:** Cannot create directories or files  
**Solution | 解決方案:**

```bash
# Check and fix permissions | 檢查並修復權限
chmod -R 755 .
mkdir -p data logs models output

# Or run as administrator | 或以管理員身份運行
sudo python test_phase1_complete.py  # Linux/Mac
# Run PowerShell as Administrator on Windows
```

## 🎯 Specific Component Testing | 特定組件測試

### Testing Configuration System | 測試配置系統

```python
# Test configuration loading | 測試配置載入
from src.main.python.utils.config import Config, get_config

# Test default config | 測試默認配置
config = Config()
print(f"Trading symbols: {config.trading.symbols}")

# Test environment-specific config | 測試特定環境配置
dev_config = get_config('development')
prod_config = get_config('production')
```

### Testing Data Pipeline | 測試數據管道

```python
# Test data downloading | 測試數據下載
from src.main.python.utils.data_loader import DataLoader

loader = DataLoader()
# This will use cached data or download from Yahoo Finance
# 這將使用緩存數據或從Yahoo財經下載
data = loader.download_data(['EURUSD'], period='5d', interval='1h')
print(f"Downloaded {len(data['EURUSD'])} records")
```

### Testing Technical Indicators | 測試技術指標

```python
# Test technical indicators | 測試技術指標
from src.main.python.utils.technical_indicators import TechnicalIndicators
import pandas as pd
import numpy as np

# Create sample data | 創建樣本數據
dates = pd.date_range('2024-01-01', periods=100, freq='H')
sample_data = pd.DataFrame({
    'Open': np.random.uniform(1.1000, 1.1100, 100),
    'High': np.random.uniform(1.1050, 1.1150, 100),
    'Low': np.random.uniform(1.0950, 1.1050, 100),
    'Close': np.random.uniform(1.1000, 1.1100, 100),
    'Volume': np.random.uniform(1000, 5000, 100)
}, index=dates)

# Calculate indicators | 計算指標
ti = TechnicalIndicators()
df_with_indicators = ti.add_all_indicators(sample_data)
print(f"Added {len(df_with_indicators.columns) - len(sample_data.columns)} indicators")
```

### Testing Logging System | 測試日誌系統

```python
# Test logging functionality | 測試日誌功能
from src.main.python.utils.logger import get_logger

logger = get_logger("TEST")
logger.info("This is a test message | 這是一條測試消息")
logger.log_data_event("download", "EURUSD", "1h", 1000)
logger.log_trade_signal("BUY", "EURUSD", 1.0850, 0.75)
```

## 📈 Performance Benchmarking | 性能基準測試

### Data Processing Speed | 數據處理速度

```python
import time
from src.main.python.utils.data_preprocessor import DataPreprocessor

# Benchmark preprocessing | 基準預處理
start_time = time.time()
preprocessor = DataPreprocessor()
processed_data = preprocessor.preprocess_data(sample_data)
processing_time = time.time() - start_time

print(f"Processing time: {processing_time:.2f}s for {len(sample_data)} records")
print(f"Records per second: {len(sample_data)/processing_time:.0f}")
```

### Technical Indicators Speed | 技術指標速度

```python
import time
from src.main.python.utils.technical_indicators import TechnicalIndicators

# Benchmark indicators calculation | 基準指標計算
start_time = time.time()
ti = TechnicalIndicators()
df_with_indicators = ti.add_all_indicators(sample_data)
calculation_time = time.time() - start_time

print(f"Indicators calculation time: {calculation_time:.2f}s")
print(f"Indicators per second: {(len(df_with_indicators.columns) - 5)/calculation_time:.0f}")
```

## ✅ Success Criteria | 成功標準

Your Phase 1 infrastructure is ready for Phase 2 if:  
如果滿足以下條件，您的第一階段基礎設施已準備好進入第二階段：

### Minimum Requirements | 最低要求
- ✅ **80%+ test pass rate** | **80%+測試通過率**
- ✅ **All critical dependencies installed** | **所有關鍵依賴項已安裝**
- ✅ **Data pipeline functional** | **數據管道正常運行**
- ✅ **Technical indicators working** | **技術指標工作正常**
- ✅ **Configuration system operational** | **配置系統運行正常**

### Optimal Requirements | 最佳要求
- ✅ **90%+ test pass rate** | **90%+測試通過率**
- ✅ **All optional packages available** | **所有可選包可用**
- ✅ **Network data download working** | **網絡數據下載正常**
- ✅ **No critical warnings** | **無關鍵警告**
- ✅ **Good system performance** | **良好的系統性能**

## 🔄 Next Steps After Testing | 測試後的下一步

### If Tests Pass (80%+) | 如果測試通過（80%+）
1. **Proceed to Phase 2** | **進入第二階段**
2. **Start AI model development** | **開始AI模型開發**
3. **Implement trading strategies** | **實施交易策略**

### If Tests Need Work (<80%) | 如果測試需要改進（<80%）
1. **Review failed tests** | **檢查失敗的測試**
2. **Fix critical issues** | **修復關鍵問題**
3. **Install missing dependencies** | **安裝缺失的依賴項**
4. **Re-run tests** | **重新運行測試**

## 🆘 Getting Help | 獲取幫助

If you encounter issues during testing:  
如果在測試過程中遇到問題：

1. **Check the error messages** | **檢查錯誤消息**
2. **Review this guide** | **查看此指南**
3. **Run individual tests** | **運行單個測試**
4. **Check system resources** | **檢查系統資源**
5. **Verify network connectivity** | **驗證網絡連接**

Remember: The goal is to have a solid foundation before proceeding to AI model development.  
記住：目標是在進行AI模型開發之前擁有堅實的基礎。