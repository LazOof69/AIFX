# IG Markets Token Refresh System Guide | IG Markets 令牌刷新系統指南

## 🎯 Overview | 概述

Your AIFX trading system now includes a comprehensive OAuth token management system that handles the constantly changing access and refresh tokens for IG Markets API integration. This system ensures uninterrupted trading operations by automatically refreshing tokens before they expire.

您的 AIFX 交易系統現在包含一個全面的 OAuth 令牌管理系統，處理 IG Markets API 整合中不斷變化的訪問和刷新令牌。此系統通過在令牌到期前自動刷新令牌，確保不間斷的交易操作。

## 🔄 Key Features | 主要功能

### ✅ Automatic Token Refresh | 自動令牌刷新
- **Background Monitoring**: Continuously monitors token expiry status
- **Proactive Refresh**: Refreshes tokens 10 seconds before expiry
- **Seamless Operation**: Trading continues without interruption
- **後台監控**: 持續監控令牌到期狀態
- **主動刷新**: 在到期前10秒刷新令牌
- **無縫操作**: 交易持續進行而不中斷

### 💾 Persistent Token Storage | 持久令牌存儲
- **File-Based Persistence**: Tokens saved to `ig_demo_credentials.json`
- **Session Recovery**: Resumes with existing tokens after restart
- **Automatic Updates**: Credentials file updated with new tokens
- **基於文件的持久性**: 令牌保存到 `ig_demo_credentials.json`
- **會話恢復**: 重啟後使用現有令牌恢復
- **自動更新**: 憑據文件使用新令牌更新

### 🔒 Thread-Safe Operations | 線程安全操作
- **Concurrent Access**: Safe for multi-threaded applications
- **Lock Protection**: Thread locks prevent race conditions
- **Production Ready**: Suitable for high-frequency trading
- **併發訪問**: 對多線程應用程序安全
- **鎖保護**: 線程鎖防止競爭條件
- **生產就緒**: 適用於高頻交易

### 🛡️ Robust Error Handling | 強大的錯誤處理
- **Retry Logic**: Automatic retry on refresh failures
- **Graceful Degradation**: Continues operation with existing valid tokens
- **Comprehensive Logging**: Detailed logs for debugging
- **重試邏輯**: 刷新失敗時自動重試
- **優雅降級**: 使用現有有效令牌繼續操作
- **全面日誌**: 詳細日誌用於調試

## 📋 System Components | 系統組件

### 1. IGTokenManager Class | IGTokenManager 類
**File**: `src/main/python/brokers/token_manager.py`

Key methods:
- `initialize_tokens()`: Initialize with OAuth response data
- `get_valid_access_token()`: Get current valid access token
- `start_auto_refresh()`: Start background monitoring
- `stop_auto_refresh()`: Stop background monitoring
- `get_auth_headers()`: Generate authentication headers

主要方法：
- `initialize_tokens()`: 使用 OAuth 響應數據初始化
- `get_valid_access_token()`: 獲取當前有效訪問令牌
- `start_auto_refresh()`: 啟動後台監控
- `stop_auto_refresh()`: 停止後台監控
- `get_auth_headers()`: 生成認證標頭

### 2. Enhanced IG Markets Connector | 增強的 IG Markets 連接器
**File**: `src/main/python/brokers/ig_markets.py`

Integration features:
- Automatic token manager initialization
- Seamless REST API authentication
- Connection status reporting
- Proper cleanup on disconnect

整合功能：
- 自動令牌管理器初始化
- 無縫 REST API 認證
- 連接狀態報告
- 斷開連接時適當清理

## 🚀 Usage Examples | 使用示例

### Basic Usage | 基本使用

```python
from brokers.ig_markets import IGMarketsConnector

# Initialize connector
connector = IGMarketsConnector()

# Connect with automatic token management
success = await connector.connect(demo=True, force_oauth=True)

if success:
    # Make API calls - tokens refreshed automatically
    accounts = await connector.get_accounts()
    positions = await connector.get_positions()

    # Get connection status including token info
    status = connector.get_connection_status()
    print(f"Token expires in: {status['token_status']['expires_in']} seconds")

# Cleanup when done
await connector.disconnect()
```

### Advanced Token Management | 高級令牌管理

```python
from brokers.token_manager import IGTokenManager

# Direct token manager usage
token_manager = IGTokenManager("ig_demo_credentials.json", demo=True)

# Initialize with OAuth response
oauth_response = {
    'access_token': 'your_access_token',
    'refresh_token': 'your_refresh_token',
    'expires_in': 60,
    'token_type': 'Bearer',
    'scope': 'profile'
}

token_manager.initialize_tokens(oauth_response)

# Start automatic refresh
await token_manager.start_auto_refresh()

# Get valid token for API calls
access_token = token_manager.get_valid_access_token()

# Get authentication headers
headers = token_manager.get_auth_headers()
```

## 📊 Token Lifecycle | 令牌生命週期

```
1. Token Initialization | 令牌初始化
   ├── Load from config file or OAuth response
   └── Calculate expiry time

2. Active Monitoring | 主動監控
   ├── Check expiry every 5 seconds
   ├── Refresh when ≤10 seconds remaining
   └── Update config file with new tokens

3. API Request Flow | API 請求流程
   ├── Get valid access token
   ├── Auto-refresh if expired
   ├── Generate auth headers
   └── Make authenticated request

4. Error Handling | 錯誤處理
   ├── Retry refresh on failure
   ├── Log detailed error information
   └── Graceful degradation
```

## 🔧 Configuration | 配置

### Credentials File Structure | 憑據文件結構

```json
{
  "ig_markets": {
    "demo": {
      "enabled": true,
      "clientId": "104475397",
      "accountId": "Z63C06",
      "oauthToken": {
        "access_token": "56d54df0-6011-4f05-8ef5-f5c6286aeaa0",
        "refresh_token": "00d583a4-96c4-42ee-843f-3d6263916a25",
        "token_type": "Bearer",
        "expires_in": "30",
        "scope": "profile"
      }
    }
  }
}
```

### Token Refresh Settings | 令牌刷新設置

- **Refresh Threshold**: 10 seconds before expiry | 刷新閾值：到期前10秒
- **Monitor Interval**: Check every 5 seconds | 監控間隔：每5秒檢查一次
- **Retry Delay**: 5 seconds on failure | 重試延遲：失敗時5秒
- **Request Timeout**: 30 seconds | 請求超時：30秒

## 🎯 Production Deployment | 生產部署

### Requirements | 要求
1. **Valid IG Markets API Key** | 有效的 IG Markets API 密鑰
2. **OAuth Credentials** | OAuth 憑據
3. **Persistent Storage Access** | 持久存儲訪問
4. **Network Connectivity** | 網絡連接

### Best Practices | 最佳實踐
- **Monitor Token Status**: Regular status checks | 監控令牌狀態：定期狀態檢查
- **Handle Network Failures**: Implement retry logic | 處理網絡故障：實現重試邏輯
- **Log Token Events**: Track refresh activity | 記錄令牌事件：跟踪刷新活動
- **Secure Storage**: Protect credentials file | 安全存儲：保護憑據文件

### Performance Considerations | 性能考慮
- **Minimal Overhead**: Background monitoring uses minimal resources | 最小開銷：後台監控使用最少資源
- **Thread Safety**: Safe for concurrent trading operations | 線程安全：對併發交易操作安全
- **Memory Efficient**: Small memory footprint | 內存高效：佔用內存小
- **Fast Token Access**: Cached token validation | 快速令牌訪問：緩存令牌驗證

## 🧪 Testing | 測試

### Run Token Refresh Tests | 運行令牌刷新測試
```bash
python test_token_refresh.py
```

### Test Components | 測試組件
- **Token Manager**: Standalone functionality testing | 令牌管理器：獨立功能測試
- **Connector Integration**: Integration testing | 連接器整合：整合測試
- **Persistence**: Token storage testing | 持久性：令牌存儲測試
- **Auto-refresh**: Background monitoring testing | 自動刷新：後台監控測試

## 📋 Status Monitoring | 狀態監控

### Token Status Information | 令牌狀態信息
```python
status = connector.get_connection_status()
token_status = status['token_status']

print(f"Status: {token_status['status']}")           # valid/expired
print(f"Expires in: {token_status['expires_in']}s")  # seconds remaining
print(f"Expires at: {token_status['expires_at']}")   # ISO timestamp
print(f"Token type: {token_status['token_type']}")   # Bearer
print(f"Scope: {token_status['scope']}")              # profile
```

### Connection Health Check | 連接健康檢查
```python
status = connector.get_connection_status()
print(f"Connection: {status['status']}")         # AUTHENTICATED/DISCONNECTED
print(f"Auth method: {status['auth_method']}")   # oauth/rest
print(f"Account: {status['account_info']}")      # account details
```

## 🛠️ Troubleshooting | 故障排除

### Common Issues | 常見問題

#### Token Refresh Failures | 令牌刷新失敗
- **Cause**: Invalid refresh token or network issues | 原因：無效刷新令牌或網絡問題
- **Solution**: Check credentials and network connectivity | 解決方案：檢查憑據和網絡連接

#### API Key Errors | API 密鑰錯誤
- **Cause**: Missing or invalid API key | 原因：缺少或無效的 API 密鑰
- **Solution**: Obtain valid IG Markets API key | 解決方案：獲取有效的 IG Markets API 密鑰

#### File Permission Errors | 文件權限錯誤
- **Cause**: Cannot write to credentials file | 原因：無法寫入憑據文件
- **Solution**: Check file permissions | 解決方案：檢查文件權限

### Debug Logging | 調試日誌
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Detailed logs for token refresh activities
# 令牌刷新活動的詳細日誌
```

## 🎉 Success! Your System is Ready | 成功！您的系統已準備就緒

Your AIFX trading system now features:
- ✅ **Automatic OAuth Token Management** | 自動 OAuth 令牌管理
- ✅ **Seamless Token Refresh** | 無縫令牌刷新
- ✅ **Production-Ready Integration** | 生產就緒整合
- ✅ **Comprehensive Error Handling** | 全面錯誤處理
- ✅ **Thread-Safe Operations** | 線程安全操作

The token refresh system handles all the complexity of IG Markets' changing tokens automatically, ensuring your trading operations continue uninterrupted with proper authentication.

令牌刷新系統自動處理 IG Markets 變化令牌的所有複雜性，確保您的交易操作在適當認證下不間斷地繼續進行。