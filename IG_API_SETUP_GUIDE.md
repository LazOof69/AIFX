# IG Markets API Setup Guide for AIFX
# IG Markets API 設置指南 - AIFX

## 🚨 CRITICAL: You Need REST API Key (Not Web API)

### Issue Identified:
- You currently have a **Web API key**
- AIFX requires a **REST API key**
- These are completely different authentication types

## 📋 Step-by-Step Setup:

### 1. Login to IG Platform
- Go to: https://www.ig.com
- Login with your account (lazoof)

### 2. Navigate to API Settings
- Click: **My IG** → **Settings** → **API Access**

### 3. Create REST API Key (Not Web API)
- Look for **"REST API"** section (not Web API)
- Click **"Create API Key"** under REST API
- Choose **"Demo"** for testing
- Save the new REST API key

### 4. Important Settings:
- **API Type**: REST API (not Web API)
- **Environment**: Demo
- **Status**: Active

### 5. Update AIFX Configuration
Replace the API key in `config/trading-config.yaml` with your new REST API key.

## 🔧 Alternative Solution (If REST API Not Available):

If IG doesn't offer REST API keys anymore, we can modify AIFX to use Web API with OAuth flow.

## 💡 Quick Test:
After getting REST API key, run:
```bash
python test_ig_api.py
```

## 📞 Support:
If you can't find REST API option:
1. Contact IG support
2. Ask specifically for "REST API access for trading applications"
3. Mention you need it for automated trading software

---
**Note**: This is a common issue - IG has different API types for different use cases.