#!/usr/bin/env python3
"""
Create Session Tokens from OAuth | 從OAuth創建會話令牌
================================================

Convert OAuth tokens to session tokens (CST/X-SECURITY-TOKEN) for trading.
將OAuth令牌轉換為交易用的會話令牌。
"""

import requests
import json
from datetime import datetime

def create_session_from_oauth():
    """Create trading session from OAuth tokens"""
    print("🔑 Creating IG Trading Session...")

    # OAuth tokens from your response
    oauth_tokens = {
        "access_token": "c85f5067-d20e-4e5a-b05f-075b4d65b09d",
        "refresh_token": "030c855d-05b1-4f17-8e18-4593de43dc37",
        "client_id": "104475397",
        "account_id": "Z63C06"
    }

    # Try different session creation methods

    # Method 1: Direct OAuth API call
    print("\n1️⃣ Trying OAuth API access...")
    headers_oauth = {
        'Accept': 'application/json; charset=UTF-8',
        'Content-Type': 'application/json; charset=UTF-8',
        'Authorization': f'Bearer {oauth_tokens["access_token"]}',
        'X-IG-API-KEY': '3a0f12d07fe51ab5f4f1835ae037e1f5e876726e',
        'Version': '1'
    }

    try:
        response = requests.get('https://demo-api.ig.com/gateway/deal/accounts',
                              headers=headers_oauth, timeout=10)
        print(f"OAuth API Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ OAuth tokens work directly!")
            return oauth_tokens
        else:
            print(f"OAuth Error: {response.text}")
    except Exception as e:
        print(f"OAuth Exception: {e}")

    # Method 2: Create session via REST API
    print("\n2️⃣ Trying session creation...")
    session_url = "https://demo-api.ig.com/gateway/deal/session"

    # Your IG credentials
    session_payload = {
        "identifier": "lazoof",
        "password": "Lazy666chen"
    }

    session_headers = {
        'Accept': 'application/json; charset=UTF-8',
        'Content-Type': 'application/json; charset=UTF-8',
        'X-IG-API-KEY': '3a0f12d07fe51ab5f4f1835ae037e1f5e876726e',
        'Version': '2'
    }

    try:
        response = requests.post(session_url,
                               json=session_payload,
                               headers=session_headers,
                               timeout=10)

        print(f"Session Creation Status: {response.status_code}")

        if response.status_code == 200:
            # Extract session tokens from headers
            cst_token = response.headers.get('CST')
            security_token = response.headers.get('X-SECURITY-TOKEN')

            if cst_token and security_token:
                session_tokens = {
                    'cst': cst_token,
                    'x_security_token': security_token,
                    'api_key': '3a0f12d07fe51ab5f4f1835ae037e1f5e876726e'
                }

                print("✅ Session created successfully!")
                print(f"CST: {cst_token[:50]}...")
                print(f"Security Token: {security_token[:50]}...")

                # Test the session tokens
                return test_session_tokens(session_tokens)
            else:
                print("❌ No session tokens in response headers")
                return None
        else:
            print(f"Session Creation Error: {response.text}")
            return None

    except Exception as e:
        print(f"Session Creation Exception: {e}")
        return None

def test_session_tokens(tokens):
    """Test session tokens"""
    print("\n3️⃣ Testing session tokens...")

    headers = {
        'Accept': 'application/json; charset=UTF-8',
        'X-IG-API-KEY': tokens['api_key'],
        'CST': tokens['cst'],
        'X-SECURITY-TOKEN': tokens['x_security_token'],
        'Version': '1'
    }

    try:
        response = requests.get('https://demo-api.ig.com/gateway/deal/accounts',
                              headers=headers, timeout=10)

        print(f"Session Test Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            accounts = data.get('accounts', [])

            print(f"✅ Session tokens work! Found {len(accounts)} accounts")

            for account in accounts:
                print(f"📊 Account: {account.get('accountId')} - {account.get('accountName')}")
                balance = account.get('balance', {})
                print(f"💰 Balance: {balance.get('balance', 'N/A')} {account.get('currency')}")

            return tokens
        else:
            print(f"Session Test Error: {response.text}")
            return None

    except Exception as e:
        print(f"Session Test Exception: {e}")
        return None

def execute_test_trade(tokens):
    """Execute a small test trade"""
    print("\n4️⃣ Executing test trade...")

    if not tokens:
        print("❌ No valid tokens available")
        return False

    # Use session tokens for trading
    headers = {
        'Accept': 'application/json; charset=UTF-8',
        'Content-Type': 'application/json; charset=UTF-8',
        'X-IG-API-KEY': tokens['api_key'],
        'CST': tokens['cst'],
        'X-SECURITY-TOKEN': tokens['x_security_token'],
        'Version': '2'
    }

    # Get USD/JPY market info first
    try:
        market_response = requests.get('https://demo-api.ig.com/gateway/deal/markets/CS.D.USDJPY.MINI.IP',
                                     headers={**headers, 'Version': '3'}, timeout=10)

        if market_response.status_code == 200:
            market_data = market_response.json()
            snapshot = market_data.get('snapshot', {})
            print(f"📈 USD/JPY: Bid={snapshot.get('bid')} Ask={snapshot.get('offer')}")
        else:
            print(f"❌ Market data error: {market_response.status_code}")
            return False

    except Exception as e:
        print(f"❌ Market data exception: {e}")
        return False

    # Create a small position
    trade_payload = {
        "epic": "CS.D.USDJPY.MINI.IP",
        "expiry": "DFB",
        "direction": "BUY",
        "size": 0.1,  # Very small size for test
        "orderType": "MARKET",
        "timeInForce": "EXECUTE_AND_ELIMINATE",
        "guaranteedStop": False,
        "forceOpen": True,
        "stopDistance": 20,
        "limitDistance": 30
    }

    try:
        trade_response = requests.post('https://demo-api.ig.com/gateway/deal/positions/otc',
                                     json=trade_payload,
                                     headers=headers,
                                     timeout=10)

        print(f"Trade Status: {trade_response.status_code}")

        if trade_response.status_code in [200, 201]:
            trade_data = trade_response.json()
            deal_reference = trade_data.get('dealReference')
            print(f"✅ Test trade created! Deal Reference: {deal_reference}")
            return True
        else:
            print(f"❌ Trade Error: {trade_response.text}")
            return False

    except Exception as e:
        print(f"❌ Trade Exception: {e}")
        return False

def main():
    """Main function"""
    print("🚀 IG MARKETS SESSION TOKEN CREATOR")
    print("🚀 IG Markets會話令牌創建器")
    print("=" * 50)

    # Create session tokens
    tokens = create_session_from_oauth()

    if tokens:
        print(f"\n🎉 SUCCESS: Valid tokens obtained!")

        # Save tokens for later use
        with open('ig_session_tokens.json', 'w') as f:
            json.dump(tokens, f, indent=2)
        print("💾 Tokens saved to ig_session_tokens.json")

        # Execute test trade
        trade_success = execute_test_trade(tokens)

        if trade_success:
            print("\n🎊 COMPLETE SUCCESS: IG Markets trading is working!")
            print("🎊 完全成功：IG Markets交易正常工作！")
        else:
            print("\n⚠️ Tokens work but trading failed")

        return tokens
    else:
        print("\n❌ FAILED: Could not create valid session")
        return None

if __name__ == "__main__":
    main()