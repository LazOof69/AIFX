#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test IG Markets API using POST /session v3 (OAuth compatible)
根據文檔測試 IG Markets API 使用 POST /session v3 (OAuth 兼容)
"""

import requests
import json
import yaml

def test_session_v3_auth():
    """Test IG API using POST /session v3 endpoint"""
    print("🚀 Testing IG API with POST /session v3 (OAuth compatible)")
    
    # Load configuration
    with open('config/trading-config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    demo_config = config['ig_markets']['demo']
    
    # IG Demo API endpoint for session v3
    url = "https://demo-api.ig.com/gateway/deal/session"
    
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Accept': 'application/json; charset=UTF-8',
        'X-IG-API-KEY': demo_config['api_key'],
        'Version': '3'  # Key difference - using v3 for OAuth tokens
    }
    
    payload = {
        'identifier': demo_config['username'],
        'password': demo_config['password']
    }
    
    try:
        print(f"📡 Sending POST /session v3 request...")
        print(f"🔑 API Key: {demo_config['api_key'][:10]}...")
        print(f"👤 Username: {demo_config['username']}")
        print(f"📚 Version: 3 (OAuth compatible)")
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Session v3 Authentication Successful!")
            print(f"🎯 Account ID: {data.get('currentAccountId', 'N/A')}")
            print(f"💰 Currency: {data.get('currencyIsoCode', 'N/A')}")
            
            # Check for OAuth tokens
            oauth_token = data.get('oauthToken')
            if oauth_token:
                print("🎉 OAuth Tokens Received:")
                print(f"   Access Token: {oauth_token.get('access_token', 'N/A')[:20]}...")
                print(f"   Refresh Token: {oauth_token.get('refresh_token', 'N/A')[:20]}...")
                print(f"   Expires In: {oauth_token.get('expires_in', 'N/A')} seconds")
                print(f"   Token Type: {oauth_token.get('token_type', 'N/A')}")
                
                # Test API call with OAuth token
                test_api_with_oauth_token(oauth_token, data.get('currentAccountId'))
                
            else:
                # Check for traditional tokens
                cst = response.headers.get('CST')
                security_token = response.headers.get('X-SECURITY-TOKEN')
                if cst and security_token:
                    print("🎉 Traditional Tokens Received:")
                    print(f"   CST: {cst[:20]}...")
                    print(f"   Security Token: {security_token[:20]}...")
            
            return True
            
        else:
            print(f"❌ Session v3 Authentication Failed:")
            try:
                error_data = response.json()
                print(f"   Error: {error_data}")
            except:
                print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

def test_api_with_oauth_token(oauth_token, account_id):
    """Test API call using OAuth token"""
    print("\n🧪 Testing API call with OAuth token...")
    
    headers = {
        'Authorization': f"Bearer {oauth_token['access_token']}",
        'IG-ACCOUNT-ID': account_id,
        'Version': '1'
    }
    
    # Test endpoint: Get accounts
    url = "https://demo-api.ig.com/gateway/deal/accounts"
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"📊 Account API Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ OAuth API Call Successful!")
            print(f"📄 Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"❌ OAuth API Call Failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ OAuth API Error: {e}")
        return False

def main():
    """Run session v3 authentication test"""
    print("🧪 IG Markets API Session v3 Test")
    print("=" * 60)
    
    success = test_session_v3_auth()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS: Session v3 authentication works!")
        print("💡 Your Web API key IS compatible with REST API via session v3")
        print("🚀 AIFX can now integrate with IG Markets API")
    else:
        print("❌ FAILED: Session v3 authentication failed")
        print("💡 Next steps:")
        print("   1. Verify IG demo account is active")
        print("   2. Check username/password are correct")
        print("   3. Contact IG support for API access verification")

if __name__ == "__main__":
    main()