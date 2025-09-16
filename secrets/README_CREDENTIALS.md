# AIFX 24/7 Trading System - Secure Credentials Setup
# AIFX 24/7 交易系統 - 安全憑證設置

## 🔐 REQUIRED CREDENTIALS FOR 24/7 OPERATION | 24/7運行所需憑證

**⚠️ IMPORTANT: You must provide your real IG Markets live trading credentials**
**⚠️ 重要：您必須提供您的真實IG Markets實盤交易憑證**

### 📋 STEP-BY-STEP SETUP | 逐步設置

1. **Create the following secret files in this directory:**
   **在此目錄中創建以下秘密文件：**

```bash
# Database password
echo "your_secure_database_password_here" > db_password.txt

# IG Markets LIVE API Credentials (YOU PROVIDE THESE)
echo "YOUR_LIVE_IG_API_KEY" > ig_api_key.txt
echo "YOUR_LIVE_IG_USERNAME" > ig_username.txt
echo "YOUR_LIVE_IG_PASSWORD" > ig_password.txt
echo "YOUR_LIVE_IG_ACCOUNT_ID" > ig_account_id.txt

# Monitoring passwords
echo "grafana_admin_password_123" > grafana_password.txt

# Alert configuration (optional)
echo "your-email@example.com" > alert_email.txt
echo "https://hooks.slack.com/your-webhook" > alert_webhook.txt
```

2. **Set secure permissions:**
   **設置安全權限：**

```bash
chmod 600 secrets/*.txt
```

### 🔑 IG MARKETS API CREDENTIALS | IG MARKETS API憑證

**To get your IG Markets live trading credentials:**
**獲取您的IG Markets實盤交易憑證：**

1. **Login to IG Markets** | 登入IG Markets
2. **Go to API Settings** | 前往API設置
3. **Generate Live API Key** | 生成實盤API密鑰
4. **Note your account details** | 記錄您的帳戶詳情

**Required Information:**
- **API Key**: Your live trading API key
- **Username**: Your IG Markets username
- **Password**: Your IG Markets password
- **Account ID**: Your live trading account ID

### 🛡️ SECURITY BEST PRACTICES | 安全最佳實踐

1. **Never commit secrets to version control**
   **絕不將秘密提交到版本控制**

2. **Use strong, unique passwords**
   **使用強大、唯一的密碼**

3. **Regular credential rotation**
   **定期更換憑證**

4. **Monitor access logs**
   **監控訪問日誌**

### ⚠️ DEMO vs LIVE MODE | 演示與實盤模式

- **Demo Mode**: Uses paper trading (no real money)
- **Live Mode**: Uses real money trading (RISK INVOLVED)

**The system is configured for LIVE TRADING MODE!**
**系統已配置為實盤交易模式！**

### 🚨 RISK WARNING | 風險警告

**LIVE TRADING INVOLVES REAL FINANCIAL RISK**
**實盤交易涉及真實的金融風險**

- Only trade with money you can afford to lose
- Monitor the system regularly
- Have stop-loss mechanisms in place
- Understand the risks involved

### 📞 SUPPORT | 支援

If you need help with credentials setup:
1. Check IG Markets documentation
2. Contact IG Markets support
3. Review AIFX system logs

---

**🎯 Once you've created all secret files, you can start the 24/7 system with:**
**🎯 創建所有秘密文件後，您可以使用以下命令啟動24/7系統：**

```bash
docker-compose -f docker-compose-24x7-usdjpy.yml up -d
```