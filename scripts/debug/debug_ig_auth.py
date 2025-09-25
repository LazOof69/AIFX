#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IG Authentication Debug Script
IG 認證調試腳本

Test different authentication methods with IG API
測試 IG API 的不同認證方法
"""

import requests
import json
import yaml

def load_config():
    """Load configuration"""
    with open('config/trading-config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_direct_session_auth():
    """Test direct session authentication"""
    print("🔍 Testing Direct IG REST API Authentication...")
    
    config = load_config()
    demo_config = config['ig_markets']['demo']
    
    # IG Demo API endpoint
    url = "https://demo-api.ig.com/gateway/deal/session"
    
    headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Accept': 'application/json; charset=UTF-8',
        'X-IG-API-KEY': demo_config['api_key'],
        'Version': '2'
    }
    
    payload = {
        'identifier': demo_config['username'],
        'password': demo_config['password']
    }
    
    try:
        print(f"📡 Sending request to: {url}")
        print(f"🔑 API Key: {demo_config['api_key'][:10]}...")
        print(f"👤 Username: {demo_config['username']}")
        
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        
        print(f"📊 Status Code: {response.status_code}")
        print(f"📋 Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Authentication Successful!")
            print(f"🎯 Account ID: {data.get('currentAccountId', 'N/A')}")
            print(f"💰 Currency: {data.get('currencyIsoCode', 'N/A')}")
            print(f"🔐 CST Token: {response.headers.get('CST', 'N/A')[:20]}...")
            print(f"🔐 Security Token: {response.headers.get('X-SECURITY-TOKEN', 'N/A')[:20]}...")
            return True
        else:
            print(f"❌ Authentication Failed:")
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

def test_trading_ig_library():
    """Test with trading-ig library verbose output"""
    print("\n🔍 Testing trading-ig Library Authentication...")
    
    try:
        from trading_ig import IGService
        import logging
        
        # Enable debug logging
        logging.basicConfig(level=logging.DEBUG)
        
        config = load_config()
        demo_config = config['ig_markets']['demo']
        
        # Create IG service with debug
        ig_service = IGService(
            username=demo_config['username'],
            password=demo_config['password'],
            api_key=demo_config['api_key'],
            acc_type='DEMO'
        )
        
        print("🔄 Attempting to create session...")
        response = ig_service.create_session()
        
        if response and response.get('accountId'):
            print("✅ trading-ig Authentication Successful!")
            print(f"🎯 Account ID: {response.get('accountId')}")
            print(f"💰 Currency: {response.get('currencyIsoCode')}")
            return True
        else:
            print("❌ trading-ig Authentication Failed")
            print(f"Response: {response}")
            return False
            
    except Exception as e:
        print(f"❌ trading-ig Error: {e}")
        return False

def main():
    """Run authentication debug tests"""
    print("🧪 IG Markets Authentication Debug Suite")
    print("=" * 60)
    
    # Test 1: Direct REST API
    result1 = test_direct_session_auth()
    
    # Test 2: trading-ig library  
    result2 = test_trading_ig_library()
    
    print("\n" + "=" * 60)
    print("📋 Results Summary:")
    print(f"   Direct REST API: {'✅ PASS' if result1 else '❌ FAIL'}")
    print(f"   trading-ig Library: {'✅ PASS' if result2 else '❌ FAIL'}")
    
    if result1 or result2:
        print("\n🎉 At least one method works!")
        print("💡 Your API key type is compatible with REST API")
    else:
        print("\n⚠️ Both methods failed")
        print("💡 Possible issues:")
        print("   - API key is Web API type (incompatible)")
        print("   - Credentials are incorrect")
        print("   - Account needs API access enabled")
        print("   - Demo account expired/inactive")

if __name__ == "__main__":
    main()