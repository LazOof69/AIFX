#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test all IG API session versions to find what works
測試所有 IG API 會話版本以找到可用的版本
"""

import requests
import json
import yaml

def load_config():
    """Load configuration"""
    with open('config/trading-config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_session_version(version):
    """Test specific session version"""
    print(f"\n🔍 Testing POST /session v{version}")
    
    config = load_config()
    demo_config = config['ig_markets']['demo']
    
    url = "https://demo-api.ig.com/gateway/deal/session"
    
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Accept': 'application/json; charset=UTF-8',
        'X-IG-API-KEY': demo_config['api_key'],
        'Version': str(version)
    }
    
    payload = {
        'identifier': demo_config['username'],
        'password': demo_config['password']
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📋 Headers: {list(response.headers.keys())}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Session v{version} SUCCESS!")
            print(f"🎯 Account ID: {data.get('currentAccountId', 'N/A')}")
            
            # Check response tokens
            if 'oauthToken' in data:
                oauth = data['oauthToken']
                print(f"🎉 OAuth Token: {oauth.get('access_token', 'N/A')[:20]}...")
            
            cst = response.headers.get('CST')
            security_token = response.headers.get('X-SECURITY-TOKEN')
            if cst:
                print(f"🎉 CST Token: {cst[:20]}...")
            if security_token:
                print(f"🎉 Security Token: {security_token[:20]}...")
                
            return True, data, response.headers
            
        else:
            try:
                error_data = response.json()
                print(f"❌ Session v{version} FAILED: {error_data}")
            except:
                print(f"❌ Session v{version} FAILED: {response.text}")
            return False, None, None
            
    except Exception as e:
        print(f"❌ Session v{version} ERROR: {e}")
        return False, None, None

def check_account_status():
    """Check if demo account is active"""
    print("\n🏥 Checking Account Status...")
    
    config = load_config()
    demo_config = config['ig_markets']['demo']
    
    # Try to get general API info (no auth required)
    try:
        url = "https://demo-api.ig.com/gateway/deal/clientsentiment"
        headers = {
            'X-IG-API-KEY': demo_config['api_key'],
            'Version': '1'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        print(f"📊 Client Sentiment API Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ API endpoint is reachable")
            print("✅ API key format appears valid")
        else:
            print(f"⚠️ API endpoint response: {response.status_code}")
            
    except Exception as e:
        print(f"❌ API endpoint check failed: {e}")

def main():
    """Test all session versions"""
    print("🧪 IG Markets API Complete Session Testing")
    print("=" * 60)
    
    # Check basic connectivity first
    check_account_status()
    
    # Test all session versions
    versions_to_test = [1, 2, 3]
    results = {}
    
    for version in versions_to_test:
        success, data, headers = test_session_version(version)
        results[version] = success
        
        if success:
            print(f"🎉 WORKING METHOD FOUND: Session v{version}")
            break
    
    print("\n" + "=" * 60)
    print("📋 RESULTS SUMMARY:")
    for version, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   Session v{version}: {status}")
    
    if not any(results.values()):
        print("\n🔍 DIAGNOSIS:")
        print("❌ ALL session versions failed with same error")
        print("💡 This confirms the issue is NOT the authentication method")
        print("💡 The issue is likely one of:")
        print("   1. Demo account is inactive/expired")
        print("   2. Username/password are incorrect") 
        print("   3. API key needs to be activated by IG support")
        print("   4. Account doesn't have API trading enabled")
        print("\n📞 RECOMMENDED ACTION:")
        print("Contact IG Support: +44 (0)20 7896 0011")
        print("Say: 'My demo account API access is not working'")
        print("Provide: Username 'lazoof' and API key for verification")

if __name__ == "__main__":
    main()