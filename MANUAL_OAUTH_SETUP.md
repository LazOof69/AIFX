# 🔐 Manual OAuth Setup for IG API | 手動OAuth設定

## 🎯 **IMMEDIATE SOLUTION - Manual OAuth Process**

Since you're in WSL2 environment without browser, follow these steps:

### **Step 1: Open OAuth URL in Windows Browser**
Copy and paste this URL in your **Windows browser** (Chrome/Edge):

```
https://demo-api.ig.com/gateway/oauth/authorize?response_type=code&client_id=3a0f12d07fe51ab5f4f1835ae037e1f5e876726e&redirect_uri=http%3A%2F%2Flocalhost%3A8080%2Fcallback&scope=read+write
```

### **Step 2: IG Login Process**
1. IG login page will appear
2. Enter your credentials:
   - **Username**: `lazoof`
   - **Password**: `Lazy666chen`
3. Authorize the application
4. You'll be redirected to: `http://localhost:8080/callback?code=AUTHORIZATION_CODE`

### **Step 3: Extract Authorization Code**
From the redirect URL, copy the **authorization code** (the part after `code=`)

Example: If URL is `http://localhost:8080/callback?code=abc123def456`
Your authorization code is: `abc123def456`

### **Step 4: Run Manual Token Exchange**
Run this Python script with your authorization code:

```python
import requests

def exchange_code_for_tokens(auth_code):
    url = "https://demo-api.ig.com/gateway/oauth/token"
    
    data = {
        'grant_type': 'authorization_code',
        'code': auth_code,
        'client_id': '3a0f12d07fe51ab5f4f1835ae037e1f5e876726e',
        'redirect_uri': 'http://localhost:8080/callback'
    }
    
    response = requests.post(url, data=data)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    
    return response.json()

# Replace 'YOUR_AUTH_CODE_HERE' with the code from Step 3
tokens = exchange_code_for_tokens('YOUR_AUTH_CODE_HERE')
```

### **Step 5: Save Tokens**
If successful, you'll get:
```json
{
    "access_token": "your-access-token",
    "refresh_token": "your-refresh-token",
    "expires_in": 3600,
    "token_type": "Bearer"
}
```

Save these tokens and use them for API authentication.

## ⚡ **QUICK TEST AFTER OAUTH**

Once you have tokens, test with:

```python
import requests

headers = {
    'Authorization': 'Bearer YOUR_ACCESS_TOKEN',
    'X-IG-API-KEY': '3a0f12d07fe51ab5f4f1835ae037e1f5e876726e',
    'Version': '1'
}

response = requests.get('https://demo-api.ig.com/gateway/deal/accounts', headers=headers)
print(f"Account Info: {response.json()}")
```

## 🎯 **RECOMMENDATION**

**For Production Use**: Contact IG Support for REST API key - much simpler than OAuth for automated trading.

**For Immediate Testing**: Use manual OAuth process above to get system working today.