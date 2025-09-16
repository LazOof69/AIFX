# 🚀 AIFX 24/7 USD/JPY Trading System | AIFX 24/7 美元/日圓交易系統

## 🎯 COMPLETE 24/7 PRODUCTION-READY TRADING SYSTEM | 完整的24/7生產就緒交易系統

Your AIFX system is now configured for **continuous 24/7 USD/JPY trading** with professional-grade infrastructure, monitoring, and safety features.

您的AIFX系統現已配置為**連續24/7美元/日圓交易**，具備專業級基礎設施、監控和安全功能。

---

## ⚡ QUICK START | 快速開始

### **🎯 One-Command Deployment | 一鍵部署**

```bash
# Start 24/7 trading system | 啟動24/7交易系統
./start_24x7_trading.sh
```

**That's it! Your system will be running 24/7.**
**就這樣！您的系統將24/7運行。**

---

## 🔐 STEP 1: Setup Credentials | 步驟1：設置憑證

### **Required: Your IG Markets Live Trading Credentials | 必需：您的IG Markets實盤交易憑證**

```bash
# Run credential setup (interactive) | 運行憑證設置（互動式）
./setup_credentials.sh
```

**You'll be prompted for:**
- IG Markets Live API Key
- IG Markets Username & Password
- IG Markets Live Account ID
- Database passwords
- Monitoring passwords

**⚠️ IMPORTANT: These must be REAL live trading credentials for 24/7 operation.**

---

## 🏗️ SYSTEM ARCHITECTURE | 系統架構

### **🔧 Production Infrastructure | 生產基礎設施**

| Component | Purpose | Port | Status |
|-----------|---------|------|---------|
| **AIFX Trading** | Main trading engine | 8088 | 24/7 Active |
| **PostgreSQL** | Trading database | 5432 | Persistent |
| **Redis Cache** | Market data cache | 6379 | High-speed |
| **Grafana** | Monitoring dashboard | 3000 | Real-time |
| **Backup Service** | Automated backups | - | Every 6hrs |

### **💾 Data Persistence | 數據持久化**
- **All trading data persists across restarts**
- **Automatic database backups every 6 hours**
- **30-day backup retention**
- **Complete transaction history**

---

## 📊 MONITORING & ACCESS | 監控與訪問

### **🌐 Web Dashboards | 網頁儀表板**

| Dashboard | URL | Purpose |
|-----------|-----|---------|
| **Trading Dashboard** | http://localhost:8088 | Live trading status |
| **Grafana Monitor** | http://localhost:3000 | System metrics |
| **Health Check** | http://localhost:8089 | System health |

**Default Grafana Login:**
- Username: `admin`
- Password: Your configured password

---

## 🎯 TRADING CONFIGURATION | 交易配置

### **🎨 USD/JPY Only Trading | 僅美元/日圓交易**
✅ **Symbol**: USD/JPY only (as requested)
✅ **Mode**: Live trading with real money
✅ **Timeframe**: 1H (optimized for analysis)
✅ **AI Models**: XGBoost + Random Forest + LSTM
✅ **Risk Management**: 2% per trade maximum
✅ **Auto-restart**: System recovers from failures

### **🔒 Safety Features | 安全功能**
- **Stop-loss protection**: ATR-based levels
- **Daily loss limits**: 5% maximum drawdown
- **Position limits**: Maximum 5 concurrent positions
- **Circuit breaker**: Automatic halt on excessive losses

---

## 🛠️ MANAGEMENT COMMANDS | 管理命令

### **📋 System Control | 系統控制**

```bash
# View live trading logs | 查看實時交易日誌
docker-compose -f docker-compose-24x7-usdjpy.yml logs -f aifx-trading

# Check system status | 檢查系統狀態
docker-compose -f docker-compose-24x7-usdjpy.yml ps

# Restart trading system | 重啟交易系統
docker-compose -f docker-compose-24x7-usdjpy.yml restart aifx-trading

# Stop all services | 停止所有服務
docker-compose -f docker-compose-24x7-usdjpy.yml down

# View database | 查看資料庫
docker-compose -f docker-compose-24x7-usdjpy.yml exec postgres-db psql -U aifx -d aifx_trading_24x7
```

### **📊 Monitoring Commands | 監控命令**

```bash
# View system health | 查看系統健康狀況
curl http://localhost:8088/health

# Check trading performance | 檢查交易績效
curl http://localhost:8088/api/performance

# Monitor active positions | 監控活躍倉位
curl http://localhost:8088/api/positions
```

---

## 🚨 SAFETY & RISK MANAGEMENT | 安全與風險管理

### **⚠️ CRITICAL RISK WARNINGS | 重要風險警告**

🔴 **THIS IS LIVE TRADING WITH REAL MONEY**
- All trades use your actual IG Markets account
- Losses are real and permanent
- Monitor the system regularly
- Understand the risks before starting

🔴 **24/7 OPERATION MEANS CONTINUOUS TRADING**
- System trades while you sleep
- Positions can open/close automatically
- Market volatility can cause rapid changes
- Weekend gaps can affect positions

### **🛡️ Built-in Protections | 內置保護**
✅ **Automatic stop-losses** on all positions
✅ **Daily loss limits** prevent excessive drawdown
✅ **Position size limits** control risk exposure
✅ **Circuit breaker** halts trading on anomalies
✅ **Health monitoring** with automatic alerts
✅ **Complete audit trail** of all activities

---

## 📈 PERFORMANCE MONITORING | 績效監控

### **📊 Key Metrics Tracked | 追蹤的關鍵指標**

- **Real-time P&L**: Live profit/loss tracking
- **Win Rate**: Percentage of profitable trades
- **Drawdown**: Maximum loss from peak
- **Sharpe Ratio**: Risk-adjusted returns
- **Position Count**: Active trading positions
- **System Uptime**: 24/7 availability status

### **🔔 Automatic Alerts | 自動警報**
- Email notifications for significant events
- Webhook alerts for system issues
- Dashboard warnings for risk thresholds
- SMS alerts (if configured)

---

## 🔄 BACKUP & RECOVERY | 備份與恢復

### **💾 Automated Backups | 自動備份**
- **Database**: Every 6 hours
- **Trading Data**: Daily full backup
- **System Logs**: Continuous archival
- **Retention**: 30 days of history

### **🔧 Recovery Procedures | 恢復程序**
```bash
# Quick restart if system stops | 系統停止時快速重啟
./start_24x7_trading.sh

# Manual database recovery | 手動資料庫恢復
docker-compose -f docker-compose-24x7-usdjpy.yml exec postgres-db pg_restore -U aifx -d aifx_trading_24x7 /backups/latest.dump
```

---

## 🎉 YOU'RE ALL SET! | 一切就緒！

### **🚀 Your 24/7 USD/JPY Trading System Includes | 您的24/7美元/日圓交易系統包括：**

✅ **Complete Docker infrastructure** for 24/7 operation
✅ **Live USD/JPY trading** with AI-powered signals
✅ **Professional monitoring** with Grafana dashboards
✅ **Automated backups** and disaster recovery
✅ **Real-time performance** tracking and analytics
✅ **Secure credential** management system
✅ **Production-grade** scalability and reliability

### **🎯 Ready to Trade 24/7 | 準備24/7交易**

**Your system will:**
- Trade USD/JPY continuously
- Use AI models for signal generation
- Monitor performance in real-time
- Backup data automatically
- Restart on failures
- Provide complete audit trails

**Just run: `./start_24x7_trading.sh` and you're live!**

---

## 📞 SUPPORT | 支援

For questions about:
- **IG Markets API**: Contact IG Markets support
- **System Configuration**: Check logs and documentation
- **Trading Performance**: Monitor Grafana dashboards
- **Technical Issues**: Review Docker logs

**🎯 You now have a complete, professional, 24/7 USD/JPY trading system!**
**🎯 您現在擁有一個完整、專業的24/7美元/日圓交易系統！**

---

**⚠️ Trade Responsibly | 負責任交易**
**💡 Monitor Regularly | 定期監控**
**🚀 Enjoy Professional Trading | 享受專業交易**