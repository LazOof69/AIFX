# 🎯 HOW TO RUN AIFX TRADING SYSTEM | 如何運行AIFX交易系統

> **Complete Guide to Running the AIFX Quantitative Trading System**  
> **運行AIFX量化交易系統的完整指南**

This guide explains exactly how to run the AIFX trading system in production, addressing the question: **"How to actually do the trading, what is the process when we run the project?"**

本指南詳細說明如何在生產環境中運行AIFX交易系統，回答問題：**"如何實際進行交易，運行項目時的流程是什麼？"**

---

## 🚀 QUICK START | 快速開始

### 1. **System Validation** | 系統驗證
```bash
# First, validate that all components are working
# 首先，驗證所有組件都正常工作
python test_integration_complete.py
```

### 2. **Demo Mode** | 演示模式
```bash
# Run paper trading demonstration  
# 運行紙上交易演示
python run_trading_demo.py --mode demo
```

### 3. **Live Trading** | 實盤交易
```bash
# Run live trading (requires API credentials)
# 運行實盤交易（需要API憑證）
python run_trading_demo.py --mode live
```

---

## 📋 COMPLETE TRADING WORKFLOW | 完整交易工作流程

### 🔄 **What Happens When You Run the Trading System?** | 運行交易系統時會發生什麼？

Here's the complete step-by-step process that occurs when you start the AIFX trading system:

以下是啟動AIFX交易系統時發生的完整逐步流程：

#### **Phase 1: System Initialization** | 第一階段：系統初始化
```
🚀 Starting AIFX Trading System...
├── 📊 Initialize Position Manager
├── 🤖 Initialize Live Trader  
├── ⚙️ Initialize Execution Engine
├── 📈 Initialize Trading Dashboard
├── 🔍 Validate API Connections (live mode only)
└── ✅ System Ready for Trading
```

#### **Phase 2: Market Data Ingestion** | 第二階段：市場數據接入
```
📊 Market Data Pipeline:
├── 🔄 Connect to IG Markets API
├── 📈 Fetch EUR/USD 1-hour data
├── 📈 Fetch USD/JPY 1-hour data  
├── 🧮 Calculate Technical Indicators
│   ├── Moving Averages (SMA, EMA)
│   ├── Momentum (RSI, MACD)
│   ├── Volatility (Bollinger Bands, ATR)
│   └── Volume Indicators
└── ✅ Market Data Updated
```

#### **Phase 3: AI Model Predictions** | 第三階段：AI模型預測
```
🤖 AI Analysis Pipeline:
├── 🌲 XGBoost Model Prediction
├── 🌳 Random Forest Model Prediction
├── 🧠 LSTM Neural Network Prediction
├── 📊 Ensemble Model Combination
├── 🎯 Confidence Score Calculation
└── ✅ AI Predictions Generated
```

#### **Phase 4: Signal Generation** | 第四階段：信號生成
```
📈 Trading Signal Generation:
├── 🔄 Combine AI Predictions + Technical Signals
├── 📊 Apply Confidence Filtering (>60% threshold)
├── 🎯 Generate Trading Decisions
│   ├── Symbol: EUR/USD or USD/JPY
│   ├── Action: BUY or SELL  
│   ├── Size: Risk-adjusted position size
│   ├── Stop Loss: ATR-based levels
│   └── Take Profit: Risk-reward optimized
└── ✅ Trading Signals Ready
```

#### **Phase 5: Risk Management** | 第五階段：風險管理
```
🛡️ Risk Validation:
├── 💰 Check Account Balance
├── 📊 Validate Position Limits
├── 🎯 Confirm Risk-per-Trade (2% max)
├── ⚖️ Check Correlation Exposure
├── 🔒 Apply Circuit Breakers
└── ✅ Risk Checks Passed
```

#### **Phase 6: Trade Execution** | 第六階段：交易執行
```
⚙️ Trade Execution Workflow:
├── 📋 Submit Order to IG Markets API
├── 🔍 Monitor Order Status
├── 📊 Track Position Opening
├── 💾 Record Position in Database
├── 📈 Start Real-time P&L Tracking
└── ✅ Trade Successfully Executed
```

#### **Phase 7: Position Monitoring** | 第七階段：倉位監控
```
📊 Continuous Monitoring:
├── 🔄 Real-time Price Updates
├── 💰 P&L Calculation
├── 🎯 Exit Condition Monitoring
│   ├── Stop Loss Triggers
│   ├── Take Profit Targets
│   └── Trailing Stop Updates
├── 📊 Performance Metrics Update
└── 🔔 Alert System Active
```

---

## 🎯 TRADING MODES EXPLAINED | 交易模式說明

### 🧪 **Test Mode** | 測試模式
```bash
python run_trading_demo.py --mode test
```

**Purpose**: Validate system components without any trading
**用途**: 驗證系統組件但不進行任何交易

**What it does**:
- Tests all system components
- Validates data connections
- Checks AI model loading
- Verifies risk management
- No money involved

### 📝 **Demo Mode** | 演示模式
```bash
python run_trading_demo.py --mode demo
```

**Purpose**: Paper trading with simulated orders
**用途**: 使用模擬訂單進行紙上交易

**What it does**:
- Full trading logic execution
- Simulated order placement
- Real market data analysis
- AI model predictions
- Portfolio tracking (virtual money)
- Performance measurement

**Perfect for**:
- Learning how the system works
- Strategy validation
- System testing
- Training purposes

### ⚠️ **Live Mode** | 實盤模式
```bash
python run_trading_demo.py --mode live
```

**Purpose**: Real money trading with actual orders
**用途**: 使用實際訂單進行真錢交易

**What it does**:
- Real order execution
- Actual money at risk
- Live position management
- Real-time monitoring
- Profit/Loss tracking

**Requirements**:
- Valid IG Markets API credentials
- Sufficient account balance
- Risk management understanding
- Continuous monitoring capability

---

## ⚙️ SYSTEM CONFIGURATION | 系統配置

### 📁 **Configuration Files** | 配置文件

```
src/main/resources/config/
├── demo_config.json      # Demo mode settings
├── production_config.json # Live trading settings  
└── test_config.json      # Test mode settings
```

### 🔑 **API Credentials Setup** | API憑證設置

For live trading, you need to configure IG Markets API credentials:
對於實盤交易，您需要配置IG Markets API憑證：

```json
{
  "ig_api": {
    "api_key": "YOUR_API_KEY",
    "username": "YOUR_USERNAME", 
    "password": "YOUR_PASSWORD",
    "account_id": "YOUR_ACCOUNT_ID",
    "environment": "DEMO" // or "LIVE"
  }
}
```

**🚨 SECURITY WARNING**: Never commit API credentials to version control!
**🚨 安全警告**: 永遠不要將API憑證提交到版本控制！

---

## 📊 MONITORING & CONTROL | 監控與控制

### 📈 **Real-time Dashboard** | 即時儀表板

When running, the system displays a live dashboard:
運行時，系統會顯示實時儀表板：

```
╔══════════════════════════════════════════════════════════════════════╗
║                    AIFX TRADING SYSTEM DASHBOARD                     ║
╠══════════════════════════════════════════════════════════════════════╣
║ System Status: 🟢 ACTIVE          │ Uptime: 02:15:32                ║
║ Mode: LIVE TRADING                 │ Last Update: 2025-01-14 15:30:45║
╠══════════════════════════════════════════════════════════════════════╣
║                           TRADING METRICS                            ║
║ Total Positions: 3                │ Open P&L: +$127.50              ║
║ Success Rate: 67%                  │ Daily P&L: +$89.25              ║
║ Risk Exposure: 4.2%               │ Max Drawdown: -1.8%             ║
╠══════════════════════════════════════════════════════════════════════╣
║                           MARKET DATA                                ║
║ EUR/USD: 1.0547 (↑+0.0012)       │ Last Update: 15:30:45           ║
║ USD/JPY: 150.23 (↓-0.15)         │ Data Source: IG Markets         ║
╠══════════════════════════════════════════════════════════════════════╣
║                              ALERTS                                  ║
║ 🔔 New signal: BUY EUR/USD (Confidence: 72%)                        ║
║ ℹ️ Position EUR/USD approaching take profit                         ║
╚══════════════════════════════════════════════════════════════════════╝
```

### ⌨️ **Control Commands** | 控制命令

While the system is running, you can use these controls:
系統運行時，您可以使用這些控制：

- **Ctrl+C**: Graceful shutdown | 優雅關閉
- **Space**: Pause/Resume trading | 暫停/恢復交易
- **R**: Refresh dashboard | 刷新儀表板
- **Q**: Quick shutdown | 快速關閉

---

## 🛡️ RISK MANAGEMENT | 風險管理

### 📊 **Built-in Risk Controls** | 內建風險控制

1. **Position Sizing**: Maximum 2% risk per trade
   **倉位大小**: 每筆交易最大2%風險

2. **Daily Loss Limit**: Stop trading if daily loss > 5%
   **日損失限制**: 日損失>5%時停止交易

3. **Maximum Positions**: Limit concurrent positions
   **最大倉位**: 限制並發倉位

4. **Correlation Check**: Avoid highly correlated positions
   **相關性檢查**: 避免高度相關的倉位

5. **Circuit Breakers**: Automatic shutdown on anomalies
   **斷路器**: 異常時自動關閉

### ⚠️ **Risk Warnings** | 風險警告

**For Live Trading**:
- Start with small position sizes
- Monitor continuously
- Have stop-loss plans
- Understand the risks
- Never risk more than you can afford to lose

**對於實盤交易**:
- 從小倉位開始
- 持續監控
- 制定止損計劃
- 了解風險
- 永遠不要冒險超出您承受能力的資金

---

## 🔧 TROUBLESHOOTING | 故障排除

### ❌ **Common Issues** | 常見問題

#### **"Import Error"**
```bash
# Solution: Ensure you're in the project root directory
cd /path/to/AIFX_CLAUDE
python run_trading_demo.py --mode demo
```

#### **"API Connection Failed"**
```bash
# Check your internet connection and API credentials
# Verify IG Markets API status
# Ensure credentials are correctly configured
```

#### **"Insufficient Balance"**
```bash
# Check your trading account balance
# Reduce position sizes
# Review risk management settings
```

#### **"High CPU Usage"**
```bash
# Normal during AI model inference
# Consider running on more powerful hardware
# Adjust update frequencies in config
```

### 📞 **Getting Help** | 獲取幫助

If you encounter issues:
如果遇到問題：

1. Check the logs in `logs/` directory
   檢查`logs/`目錄中的日誌

2. Run the integration tests
   運行集成測試

3. Review configuration files
   檢查配置文件

4. Verify API credentials
   驗證API憑證

---

## 📈 **PERFORMANCE OPTIMIZATION** | 性能優化

### 🚀 **System Requirements** | 系統要求

**Minimum**:
- CPU: 4 cores, 2.5GHz
- RAM: 8GB
- Storage: 50GB SSD
- Network: Stable internet connection

**Recommended**:
- CPU: 8 cores, 3.0GHz+
- RAM: 16GB+
- Storage: 100GB+ NVMe SSD
- Network: Low-latency connection

### ⚡ **Optimization Tips** | 優化提示

1. **Use SSD Storage**: Faster data access
   **使用SSD存儲**: 更快的數據訪問

2. **Stable Network**: Reduce API timeouts
   **穩定網路**: 減少API超時

3. **Monitor Resources**: Watch CPU/RAM usage
   **監控資源**: 監視CPU/RAM使用

4. **Regular Updates**: Keep dependencies current
   **定期更新**: 保持依賴項最新

---

## 🎯 **SUCCESS METRICS** | 成功指標

### 📊 **Key Performance Indicators** | 關鍵績效指標

Monitor these metrics to evaluate trading performance:
監控這些指標來評估交易性能：

1. **Win Rate**: % of profitable trades
   **勝率**: 獲利交易的百分比

2. **Profit Factor**: Gross profit / Gross loss  
   **獲利因子**: 總獲利 / 總虧損

3. **Sharpe Ratio**: Risk-adjusted returns
   **夏普比率**: 風險調整回報

4. **Maximum Drawdown**: Largest peak-to-trough decline
   **最大回撤**: 最大的峰谷下降

5. **Average Trade**: Mean profit/loss per trade
   **平均交易**: 每筆交易的平均盈虧

---

## 🎉 **READY TO TRADE!** | 準備交易！

You now have a complete understanding of how to run the AIFX trading system. The process is:
您現在已經完全了解如何運行AIFX交易系統。流程是：

1. **Validate** → Run integration tests
2. **Practice** → Use demo mode  
3. **Go Live** → Start live trading
4. **Monitor** → Watch the dashboard
5. **Optimize** → Improve based on results

**Happy Trading!** 🚀📈💰  
**祝您交易愉快！** 🚀📈💰