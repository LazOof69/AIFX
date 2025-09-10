#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test IG API using official documentation format
使用官方文檔格式測試 IG API
"""

import yaml
from trading_ig import IGService

def load_config():
    """Load configuration from yaml file"""
    with open('config/trading-config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_official_format():
    """Test using exact format from official docs"""
    print("🧪 Testing IG API using Official Documentation Format")
    print("🧪 使用官方文檔格式測試 IG API")
    print("=" * 60)
    
    # Load config
    config = load_config()
    demo_config = config['ig_markets']['demo']
    
    print(f"📋 Configuration:")
    print(f"   Username: {demo_config['username']}")
    print(f"   API Key: {demo_config['api_key'][:10]}...")
    print(f"   Account Type: DEMO")
    print("")
    
    try:
        # Create IG service using official format
        print("🔄 Creating IG service...")
        ig_service = IGService(
            username=demo_config['username'],
            password=demo_config['password'], 
            api_key=demo_config['api_key'],
            acc_type='DEMO'
        )
        
        print("✅ IG service created successfully")
        
        # Create session
        print("🔄 Creating session...")
        session_response = ig_service.create_session()
        
        if session_response:
            print("✅ Session created successfully!")
            print(f"   Account ID: {session_response.get('accountId', 'N/A')}")
            print(f"   Currency: {session_response.get('currencyIsoCode', 'N/A')}")
            print(f"   Lightstreamer Endpoint: {session_response.get('lightstreamerEndpoint', 'N/A')}")
            
            # Test account info (as per official docs)
            print("\n🔄 Fetching account info...")
            try:
                account_info = ig_service.fetch_accounts()
                print("✅ Account info retrieved:")
                if account_info and 'accounts' in account_info:
                    for acc in account_info['accounts']:
                        print(f"   Account ID: {acc['accountId']}")
                        print(f"   Account Name: {acc['accountName']}")
                        print(f"   Status: {acc['status']}")
                        if 'balance' in acc:
                            balance = acc['balance']
                            print(f"   Balance: {balance.get('balance', 'N/A')} {acc['currency']}")
                            print(f"   Available: {balance.get('available', 'N/A')} {acc['currency']}")
            except Exception as e:
                print(f"⚠️ Account info error: {e}")
            
            # Test open positions (as per official docs)  
            print("\n🔄 Fetching open positions...")
            try:
                open_positions = ig_service.fetch_open_positions()
                print("✅ Open positions retrieved:")
                print(f"   Positions: {len(open_positions.get('positions', []))} found")
                if open_positions.get('positions'):
                    for pos in open_positions['positions'][:3]:  # Show first 3
                        print(f"   - {pos['market']['instrumentName']}: {pos['position']['size']} @ {pos['position']['level']}")
                else:
                    print("   - No open positions")
            except Exception as e:
                print(f"⚠️ Open positions error: {e}")
            
            # Test historical prices (as per official docs)
            print("\n🔄 Fetching historical prices...")
            try:
                epic = 'CS.D.EURUSD.MINI.IP'
                resolution = 'D'
                num_points = 5
                
                response = ig_service.fetch_historical_prices_by_epic_and_num_points(
                    epic, resolution, num_points
                )
                
                if response and 'prices' in response:
                    print("✅ Historical prices retrieved:")
                    print(f"   Epic: {epic}")
                    print(f"   Resolution: {resolution}")
                    print(f"   Points: {num_points}")
                    
                    if 'ask' in response['prices']:
                        ask_prices = response['prices']['ask']
                        print(f"   Ask prices shape: {ask_prices.shape if hasattr(ask_prices, 'shape') else 'N/A'}")
                        if hasattr(ask_prices, 'head'):
                            print("   Sample ask prices:")
                            print(ask_prices.head().to_string(max_cols=4))
                    
            except Exception as e:
                print(f"⚠️ Historical prices error: {e}")
            
            print("\n🎉 SUCCESS: Official format test completed!")
            print("🎉 成功：官方格式測試完成！")
            print("\n💡 Your API key IS working with REST API!")
            print("💡 您的 API 密鑰確實適用於 REST API！")
            return True
            
        else:
            print("❌ Session creation failed")
            print("❌ 會話創建失敗")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"❌ 錯誤：{e}")
        
        # Check if it's the invalid details error
        if "invalid-details" in str(e).lower():
            print("\n🔍 Analysis:")
            print("   This could be:")
            print("   1. Wrong API key type (Web API vs REST API)")
            print("   2. Incorrect credentials")
            print("   3. Account needs API access activation")
            print("   4. Demo account expired")
        
        return False

if __name__ == "__main__":
    test_official_format()