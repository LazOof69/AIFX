#!/bin/bash
# AIFX 24/7 Trading System - Credential Setup Script
# AIFX 24/7 交易系統 - 憑證設置腳本

set -e

echo "🔐 AIFX 24/7 Trading System - Credential Setup"
echo "🔐 AIFX 24/7 交易系統 - 憑證設置"
echo "=============================================="

# Create secrets directory
mkdir -p secrets

# Function to prompt for secure input
prompt_secure() {
    local prompt="$1"
    local filename="$2"

    echo ""
    echo "📋 $prompt"
    read -s -p "Enter value: " value
    echo ""

    if [ -z "$value" ]; then
        echo "⚠️ Warning: Empty value provided for $filename"
        echo "default_value_please_change" > "secrets/$filename"
    else
        echo "$value" > "secrets/$filename"
    fi

    chmod 600 "secrets/$filename"
    echo "✅ Saved: secrets/$filename"
}

echo ""
echo "🚨 IMPORTANT: This will set up LIVE TRADING credentials"
echo "🚨 重要：這將設置實盤交易憑證"
echo ""
echo "⚠️ RISK WARNING: Live trading involves real money!"
echo "⚠️ 風險警告：實盤交易涉及真實資金！"
echo ""
read -p "Do you want to continue with LIVE TRADING setup? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "❌ Setup cancelled. Use demo mode instead."
    exit 1
fi

# Database credentials
echo ""
echo "🗄️ DATABASE CREDENTIALS | 資料庫憑證"
prompt_secure "Database password (used internally)" "db_password.txt"

# IG Markets Live API credentials
echo ""
echo "🏦 IG MARKETS LIVE API CREDENTIALS | IG MARKETS 實盤API憑證"
echo "⚠️ These must be your real IG Markets live trading credentials"
echo ""

prompt_secure "IG Markets Live API Key" "ig_api_key.txt"
prompt_secure "IG Markets Username" "ig_username.txt"
prompt_secure "IG Markets Password" "ig_password.txt"
prompt_secure "IG Markets Live Account ID" "ig_account_id.txt"

# Monitoring credentials
echo ""
echo "📊 MONITORING CREDENTIALS | 監控憑證"
prompt_secure "Grafana Admin Password" "grafana_password.txt"

# Optional alert configuration
echo ""
echo "📧 ALERT CONFIGURATION (Optional) | 警報配置（可選）"
echo "Press Enter to skip if you don't want email/webhook alerts"
echo ""

read -p "Alert Email (optional): " alert_email
if [ -n "$alert_email" ]; then
    echo "$alert_email" > secrets/alert_email.txt
    chmod 600 secrets/alert_email.txt
    echo "✅ Saved: secrets/alert_email.txt"
else
    echo "no-email@localhost" > secrets/alert_email.txt
    chmod 600 secrets/alert_email.txt
fi

read -p "Alert Webhook URL (optional): " alert_webhook
if [ -n "$alert_webhook" ]; then
    echo "$alert_webhook" > secrets/alert_webhook.txt
    chmod 600 secrets/alert_webhook.txt
    echo "✅ Saved: secrets/alert_webhook.txt"
else
    echo "http://localhost/no-webhook" > secrets/alert_webhook.txt
    chmod 600 secrets/alert_webhook.txt
fi

# Set secure permissions on secrets directory
chmod 700 secrets
chmod 600 secrets/*.txt

echo ""
echo "✅ CREDENTIAL SETUP COMPLETED | 憑證設置完成"
echo "=============================================="
echo ""
echo "📁 Created files in secrets/ directory:"
ls -la secrets/

echo ""
echo "🚀 NEXT STEPS | 後續步驟:"
echo "1. Verify your IG Markets credentials are correct"
echo "2. Start the 24/7 trading system:"
echo "   docker-compose -f docker-compose-24x7-usdjpy.yml up -d"
echo ""
echo "📊 Monitor your trading system at:"
echo "   • Trading Dashboard: http://localhost:8088"
echo "   • Grafana Monitor: http://localhost:3000"
echo ""
echo "⚠️ REMEMBER: This is LIVE TRADING with real money!"
echo "⚠️ 記住：這是使用真實資金的實盤交易！"