#!/usr/bin/env python3
"""
Simple IG Demo Credentials Test | 簡單 IG 演示憑據測試
==================================================

Tests the provided IG demo account credentials structure and
creates a basic trading simulation using the account information.
測試提供的 IG 演示帳戶憑據結構並創建基本交易模擬。
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def test_credentials():
    """
    Test the IG demo credentials file
    測試 IG 演示憑據文件
    """

    print("="*80)
    print("🔍 IG DEMO CREDENTIALS VALIDATION | IG 演示憑據驗證")
    print("="*80)

    try:
        # Load credentials file
        creds_path = Path("ig_demo_credentials.json")

        if not creds_path.exists():
            print("❌ Credentials file not found: ig_demo_credentials.json")
            return False

        with open(creds_path, 'r') as f:
            config = json.load(f)

        print("✅ Credentials file loaded successfully")

        # Validate structure
        if 'ig_markets' not in config:
            print("❌ Missing 'ig_markets' section")
            return False

        if 'demo' not in config['ig_markets']:
            print("❌ Missing 'demo' account section")
            return False

        demo_config = config['ig_markets']['demo']

        # Check required fields
        required_fields = ['clientId', 'accountId', 'oauthToken']
        for field in required_fields:
            if field not in demo_config:
                print(f"❌ Missing required field: {field}")
                return False

        print("✅ All required fields present")

        # Display account info
        print(f"📊 Client ID: {demo_config['clientId']}")
        print(f"🏦 Account ID: {demo_config['accountId']}")
        print(f"🌐 Endpoint: {demo_config.get('lightstreamerEndpoint', 'N/A')}")

        # Check OAuth token
        oauth_token = demo_config['oauthToken']
        if 'access_token' in oauth_token and oauth_token['access_token']:
            print("✅ OAuth access token found")
            print(f"🔑 Token type: {oauth_token.get('token_type', 'N/A')}")
            print(f"⏰ Expires in: {oauth_token.get('expires_in', 'N/A')} minutes")
        else:
            print("❌ OAuth access token missing or empty")
            return False

        return True

    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_demo_trading_session():
    """
    Create a demo trading session using the provided credentials
    使用提供的憑據創建演示交易會話
    """

    print("\n" + "="*80)
    print("🎯 DEMO TRADING SESSION SIMULATION | 演示交易會話模擬")
    print("="*80)

    try:
        with open('ig_demo_credentials.json', 'r') as f:
            config = json.load(f)

        demo_config = config['ig_markets']['demo']

        # Create simulated trading session
        trading_session = {
            'session_id': f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'account_id': demo_config['accountId'],
            'client_id': demo_config['clientId'],
            'account_type': 'DEMO',
            'base_currency': 'USD',
            'initial_balance': 10000.00,
            'available_balance': 10000.00,
            'trading_pairs': demo_config.get('trading_config', {}).get('currency_pairs', ['USD/JPY', 'EUR/USD']),
            'status': 'ACTIVE',
            'created_at': datetime.now().isoformat()
        }

        print("✅ Demo trading session created:")
        print(f"🆔 Session ID: {trading_session['session_id']}")
        print(f"🏦 Account: {trading_session['account_id']}")
        print(f"💰 Balance: ${trading_session['initial_balance']:,.2f}")
        print(f"📈 Trading Pairs: {', '.join(trading_session['trading_pairs'])}")
        print(f"🔄 Status: {trading_session['status']}")

        # Save session info
        session_file = Path("demo_trading_session.json")
        with open(session_file, 'w') as f:
            json.dump(trading_session, f, indent=2)

        print(f"💾 Session saved to: {session_file}")

        return True

    except Exception as e:
        print(f"❌ Session creation error: {e}")
        return False

def main():
    """Main test function"""

    print("🚀 Starting IG Demo Credentials Test...")

    # Test 1: Validate credentials
    creds_valid = test_credentials()

    if not creds_valid:
        print("\n🚨 Credentials validation failed!")
        sys.exit(1)

    # Test 2: Create demo session
    session_created = create_demo_trading_session()

    if not session_created:
        print("\n🚨 Demo session creation failed!")
        sys.exit(1)

    print("\n" + "="*80)
    print("🎉 ALL TESTS PASSED! | 所有測試通過！")
    print("="*80)
    print("✅ Your IG demo credentials are properly structured")
    print("✅ Demo trading session ready for use")
    print("✅ You can now run trading simulations")

    print("\n💡 Next steps:")
    print("   1. Run: python run_trading_demo.py --mode demo")
    print("   2. The system will use your credentials automatically")
    print("   3. Monitor trading via the dashboard")

    sys.exit(0)

if __name__ == "__main__":
    main()