#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Demo Account Specific IG API Issues
測試模擬帳戶特定的 IG API 問題
"""

import requests
import yaml
import json
from trading_ig import IGService

def load_config():
    """Load configuration"""
    with open('config/trading-config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_demo_account_specifics():
    """Test demo account specific API behavior"""
    print("🧪 Demo Account API Analysis")
    print("🧪 模擬帳戶 API 分析")
    print("=" * 60)
    
    config = load_config()
    demo_config = config['ig_markets']['demo']
    
    print(f"📋 Demo Account Configuration:")
    print(f"   Username: {demo_config['username']}")
    print(f"   API Key: {demo_config['api_key'][:15]}...")
    print(f"   Environment: DEMO")
    print("")
    
    # Test 1: Demo-specific headers
    print("🔍 Test 1: Demo Account Headers")
    print("-" * 30)
    
    demo_headers = {
        'Content-Type': 'application/json; charset=UTF-8',
        'Accept': 'application/json; charset=UTF-8',
        'X-IG-API-KEY': demo_config['api_key'],
        'Version': '2',
        'IG-ACCOUNT-TYPE': 'DEMO'  # Try demo-specific header
    }
    
    payload = {
        'identifier': demo_config['username'],
        'password': demo_config['password']
    }
    
    try:
        response = requests.post(
            "https://demo-api.ig.com/gateway/deal/session",
            headers=demo_headers,
            json=payload,
            timeout=10
        )
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("✅ SUCCESS with demo headers!")
            data = response.json()
            print(f"Account ID: {data.get('accountId', 'N/A')}")
            return True
        else:
            try:
                error_data = response.json()
                print(f"❌ Error: {error_data}")
            except:
                print(f"❌ Error: {response.text}")
    except Exception as e:
        print(f"❌ Request failed: {e}")
    
    # Test 2: Check if demo account is active
    print("\n🔍 Test 2: Demo Account Status Check")
    print("-" * 30)
    
    try:
        # Try to get account info without authentication
        status_url = "https://demo-api.ig.com/gateway/deal/ping"
        ping_response = requests.get(status_url, timeout=10)
        print(f"Demo API Ping Status: {ping_response.status_code}")
        
        if ping_response.status_code == 200:
            print("✅ Demo API is accessible")
        else:
            print("❌ Demo API may be down")
    except Exception as e:
        print(f"❌ Ping failed: {e}")
    
    # Test 3: Check demo account activation
    print("\n🔍 Test 3: Demo Account Activation Status")
    print("-" * 30)
    
    # Common demo account issues
    demo_issues = [
        "Demo account might be expired (usually 30-90 days)",
        "Demo account needs reactivation", 
        "API access not enabled for demo account",
        "Wrong demo API endpoint",
        "Demo account credentials different from live account"
    ]
    
    print("💡 Common Demo Account Issues:")
    for i, issue in enumerate(demo_issues, 1):
        print(f"   {i}. {issue}")
    
    return False

def test_demo_vs_live_endpoints():
    """Test different demo endpoints"""
    print("\n🔍 Test 4: Demo vs Live Endpoints")
    print("-" * 30)
    
    config = load_config()
    demo_config = config['ig_markets']['demo']
    
    # Different possible demo endpoints
    demo_endpoints = [
        "https://demo-api.ig.com/gateway/deal",
        "https://demo-apd.ig.com/gateway/deal",  # Alternative demo
        "https://api.ig.com/gateway/deal",       # Live endpoint (might work for demo)
    ]
    
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
    
    for endpoint in demo_endpoints:
        print(f"\n📡 Testing: {endpoint}")
        try:
            response = requests.post(
                f"{endpoint}/session",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print("   ✅ SUCCESS!")
                data = response.json()
                print(f"   Account ID: {data.get('accountId', 'N/A')}")
                return True
            else:
                try:
                    error_data = response.json()
                    error_code = error_data.get('errorCode', 'unknown')
                    print(f"   ❌ Error: {error_code}")
                except:
                    print(f"   ❌ HTTP {response.status_code}")
                    
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Connection failed - endpoint may not exist")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return False

def test_demo_account_with_trading_ig():
    """Test using trading-ig library with explicit demo settings"""
    print("\n🔍 Test 5: trading-ig Library Demo Settings")
    print("-" * 30)
    
    config = load_config()
    demo_config = config['ig_markets']['demo']
    
    try:
        # Try with explicit demo settings
        print("🔄 Testing with trading-ig explicit demo mode...")
        
        ig_service = IGService(
            username=demo_config['username'],
            password=demo_config['password'],
            api_key=demo_config['api_key'],
            acc_type='DEMO'
        )
        
        # Try to create session with verbose error handling
        try:
            session_response = ig_service.create_session()
            
            if session_response:
                print("✅ SUCCESS with trading-ig demo mode!")
                print(f"   Account ID: {session_response.get('accountId', 'N/A')}")
                print(f"   Currency: {session_response.get('currencyIsoCode', 'N/A')}")
                return True
            else:
                print("❌ Session creation returned None")
                
        except Exception as session_error:
            print(f"❌ Session error: {session_error}")
            
            # Check if it's a specific demo account error
            error_str = str(session_error).lower()
            if 'invalid-details' in error_str:
                print("🔍 This is the same 'invalid-details' error")
                print("💡 Issue is with API key type, not demo vs live")
            elif 'demo' in error_str:
                print("🔍 Demo-specific error detected")
            elif 'expired' in error_str:
                print("🔍 Account may be expired")
                
    except Exception as e:
        print(f"❌ Library setup failed: {e}")
    
    return False

def show_demo_account_solutions():
    """Show solutions specific to demo accounts"""
    print("\n" + "=" * 60)
    print("🎯 DEMO ACCOUNT SPECIFIC SOLUTIONS")
    print("🎯 模擬帳戶特定解決方案")
    print("=" * 60)
    
    solutions = [
        {
            "title": "Solution 1: Reactivate Demo Account",
            "steps": [
                "1. Login to IG web platform",
                "2. Go to Demo Account section", 
                "3. Check if demo account is active",
                "4. Reactivate if expired (usually free)",
                "5. Get new API key for reactivated demo"
            ]
        },
        {
            "title": "Solution 2: Create New Demo Account",
            "steps": [
                "1. Go to IG Markets website",
                "2. Create new demo account",
                "3. Apply for API access on new demo",
                "4. Request REST API key (not Web API)",
                "5. Update AIFX configuration"
            ]
        },
        {
            "title": "Solution 3: Use Live Account (Small Balance)",
            "steps": [
                "1. Open IG live account with minimum deposit",
                "2. Apply for REST API access",
                "3. Use small position sizes for testing",
                "4. Enable paper trading mode in AIFX",
                "5. Real API with simulated trades"
            ]
        },
        {
            "title": "Solution 4: Use OAuth with Demo (Current Solution)",
            "steps": [
                "1. Run: python ig_oauth_complete.py", 
                "2. Complete OAuth flow with demo credentials",
                "3. Verify demo account access works",
                "4. Continue with AIFX OAuth integration",
                "5. Demo trading through OAuth"
            ]
        }
    ]
    
    for solution in solutions:
        print(f"\n📋 {solution['title']}")
        print("-" * len(solution['title']))
        for step in solution['steps']:
            print(f"   {step}")
    
    print(f"\n💡 RECOMMENDED FOR DEMO ACCOUNT:")
    print(f"   ✅ Try Solution 4 first (OAuth with your current demo)")
    print(f"   ✅ If that fails, try Solution 1 (reactivate demo)")  
    print(f"   ✅ Solution 2 as backup (new demo account)")

def main():
    """Main test function"""
    print("🚀 IG Markets Demo Account Specific Testing")
    print("🚀 IG Markets 模擬帳戶特定測試")
    print("")
    
    # Run tests
    test1_success = test_demo_account_specifics()
    test2_success = test_demo_vs_live_endpoints()
    test3_success = test_demo_account_with_trading_ig()
    
    # Show solutions regardless of test results
    show_demo_account_solutions()
    
    print("\n" + "=" * 60)
    print("📊 DEMO ACCOUNT DIAGNOSIS")
    print("=" * 60)
    
    if any([test1_success, test2_success, test3_success]):
        print("✅ Demo account API is working!")
        print("🎉 Proceed with standard AIFX integration")
    else:
        print("❌ Demo account API access issues confirmed")
        print("🔍 Most likely causes:")
        print("   1. Demo account expired/inactive")
        print("   2. Web API key instead of REST API key") 
        print("   3. API access not enabled for demo")
        print("")
        print("🎯 RECOMMENDATION: Use OAuth solution or reactivate demo account")

if __name__ == "__main__":
    main()