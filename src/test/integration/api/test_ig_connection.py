#!/usr/bin/env python3
"""
Test IG Markets API Connection | 測試IG Markets API連接
=======================================================

Check if session tokens are valid and investigate position issues.
檢查會話令牌是否有效並調查倉位問題。
"""

import json
import requests
from datetime import datetime

def test_ig_connection():
    """Test IG API connection and troubleshoot position issues"""
    print("🔍 DIAGNOSING IG MARKETS API CONNECTION")
    print("🔍 診斷IG Markets API連接")
    print("=" * 50)

    # Step 1: Load and verify session tokens
    print("\n1️⃣ Loading Session Tokens...")
    try:
        with open('ig_session_tokens.json', 'r') as f:
            tokens = json.load(f)
        print("✅ Session tokens loaded successfully")
        print(f"CST: {tokens['cst'][:50]}...")
        print(f"X-SECURITY-TOKEN: {tokens['x_security_token'][:50]}...")
        print(f"API-KEY: {tokens['api_key']}")
    except Exception as e:
        print(f"❌ Token loading error: {e}")
        return False

    # Step 2: Test API connection
    print("\n2️⃣ Testing API Connection...")
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
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            accounts = data.get('accounts', [])
            print(f"✅ API Connection successful - Found {len(accounts)} accounts")

            for account in accounts:
                print(f"📊 Account ID: {account.get('accountId')}")
                print(f"💰 Balance: ${account.get('balance', {}).get('balance', 'N/A')}")
                print(f"💵 Available: ${account.get('balance', {}).get('available', 'N/A')}")
                print(f"💱 Currency: {account.get('currency')}")
        else:
            print(f"❌ API Connection failed: {response.text}")
            if response.status_code == 401:
                print("🔑 Session tokens may have expired - need to recreate")
            return False

    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False

    # Step 3: Check current positions
    print("\n3️⃣ Checking Current Positions...")
    try:
        headers['Version'] = '2'
        response = requests.get('https://demo-api.ig.com/gateway/deal/positions',
                              headers=headers, timeout=10)

        if response.status_code == 200:
            positions = response.json().get('positions', [])
            print(f"📊 Current Open Positions: {len(positions)}")

            if positions:
                for i, pos in enumerate(positions, 1):
                    market = pos.get('market', {})
                    position = pos.get('position', {})

                    print(f"\n{i}. Position Details:")
                    print(f"   Instrument: {market.get('instrumentName')}")
                    print(f"   Direction: {position.get('direction')}")
                    print(f"   Size: {position.get('size')}")
                    print(f"   Entry Level: {position.get('openLevel')}")
                    print(f"   P&L: {position.get('unrealisedPL')}")
            else:
                print("❌ No open positions found")
        else:
            print(f"❌ Position query failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Position check error: {e}")

    # Step 4: Check position history/activity
    print("\n4️⃣ Checking Position History...")
    try:
        headers['Version'] = '2'
        response = requests.get('https://demo-api.ig.com/gateway/deal/history/activity',
                              headers=headers, timeout=10)

        if response.status_code == 200:
            activities = response.json().get('activities', [])
            print(f"📋 Recent Activity Count: {len(activities)}")

            # Show recent trading activities
            recent_trades = [a for a in activities if a.get('type') in ['POSITION', 'DEAL']][:5]
            if recent_trades:
                print("📈 Recent Trading Activity:")
                for activity in recent_trades:
                    print(f"   {activity.get('date')}: {activity.get('description')}")
            else:
                print("❌ No recent trading activity found")
        else:
            print(f"❌ History query failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ History check error: {e}")

    # Step 5: Test market data access
    print("\n5️⃣ Testing Market Data Access...")
    try:
        headers['Version'] = '3'
        response = requests.get('https://demo-api.ig.com/gateway/deal/markets/CS.D.USDJPY.MINI.IP',
                              headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            instrument = data.get('instrument', {})
            snapshot = data.get('snapshot', {})

            print("✅ Market data access successful")
            print(f"📈 Instrument: {instrument.get('name')}")
            print(f"💹 Bid: {snapshot.get('bid')} | Ask: {snapshot.get('offer')}")
            print(f"📊 Status: {instrument.get('marketStatus')}")
        else:
            print(f"❌ Market data failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"❌ Market data error: {e}")

    return True

if __name__ == "__main__":
    test_ig_connection()