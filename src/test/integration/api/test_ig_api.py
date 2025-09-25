#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IG Markets API Test Script
IG Markets API 測試腳本

This script tests the IG Markets API connection and basic functionality.
此腳本測試 IG Markets API 連接和基本功能。
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src path for imports | 添加 src 路徑用於導入
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test if all required imports work | 測試所有必需的導入是否工作"""
    print("🔍 Testing imports...")
    
    try:
        import trading_ig
        print("✅ trading-ig imported successfully")
        
        from trading_ig import IGService
        print("✅ IGService imported successfully") 
        
        from src.main.python.brokers.ig_markets import IGMarketsConnector, create_ig_connector
        print("✅ IG Markets connector imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_configuration():
    """Test configuration loading | 測試配置載入"""
    print("\n📋 Testing configuration...")
    
    try:
        import yaml
        
        config_path = "config/trading-config.yaml"
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            print("💡 Please ensure you have configured your IG credentials in the file")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Check required sections | 檢查必需的部分
        if 'ig_markets' not in config:
            print("❌ ig_markets section missing from configuration")
            return False
            
        if 'demo' not in config['ig_markets']:
            print("❌ demo account configuration missing")
            return False
            
        demo_config = config['ig_markets']['demo']
        
        # Check required fields | 檢查必需字段
        required_fields = ['api_key', 'username', 'password']
        for field in required_fields:
            if field not in demo_config or demo_config[field] in ['YOUR_USERNAME', 'YOUR_PASSWORD', '']:
                print(f"❌ {field} not configured properly")
                print(f"💡 Please update {config_path} with your actual IG credentials")
                return False
        
        print("✅ Configuration file loaded and validated successfully")
        print(f"   API Key: {demo_config['api_key'][:10]}...")
        print(f"   Username: {demo_config['username']}")
        print(f"   Demo account: {'enabled' if demo_config.get('enabled', False) else 'disabled'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

async def test_connection():
    """Test IG API connection | 測試 IG API 連接"""
    print("\n🔗 Testing IG API connection...")
    
    try:
        from src.main.python.brokers.ig_markets import create_ig_connector
        
        # Create connector | 創建連接器
        connector = create_ig_connector("config/trading-config.yaml")
        print("✅ IG connector created successfully")
        
        # Test connection | 測試連接
        print("🔄 Attempting to connect to IG demo account...")
        success = await connector.connect(demo=True)
        
        if success:
            print("✅ Successfully connected to IG Markets demo account!")
            
            # Get status | 獲取狀態
            status = connector.get_status()
            print(f"✅ Connection status: {status['status']}")
            
            if status['account_info']:
                account = status['account_info']
                print(f"✅ Account ID: {account['account_id']}")
                print(f"✅ Account Name: {account['account_name']}")
                print(f"✅ Balance: {account['balance']} {account['currency']}")
                print(f"✅ Available: {account['available']} {account['currency']}")
            
            return True
        else:
            print("❌ Failed to connect to IG Markets")
            return False
            
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        print("💡 Common issues:")
        print("   - Check your IG credentials in config/trading-config.yaml")
        print("   - Ensure your IG account has API access enabled")
        print("   - Verify you're using the correct demo account credentials")
        return False
    finally:
        if 'connector' in locals():
            await connector.disconnect()

async def test_market_data():
    """Test market data retrieval | 測試市場數據獲取"""
    print("\n📊 Testing market data retrieval...")
    
    try:
        from src.main.python.brokers.ig_markets import create_ig_connector
        
        connector = create_ig_connector("config/trading-config.yaml")
        
        # Connect | 連接
        success = await connector.connect(demo=True)
        if not success:
            print("❌ Cannot test market data - connection failed")
            return False
        
        # Test EUR/USD market data | 測試 EUR/USD 市場數據
        print("🔄 Fetching EUR/USD market data...")
        market_data = await connector.get_market_data("CS.D.EURUSD.MINI.IP")
        
        if market_data:
            print("✅ Market data retrieved successfully:")
            print(f"   Epic: {market_data['epic']}")
            print(f"   Name: {market_data['instrument_name']}")
            print(f"   Bid: {market_data['bid']:.5f}")
            print(f"   Ask: {market_data['ask']:.5f}")
            print(f"   Mid: {market_data['mid']:.5f}")
            print(f"   Status: {market_data['market_status']}")
            return True
        else:
            print("❌ No market data received")
            return False
            
    except Exception as e:
        print(f"❌ Market data test failed: {e}")
        return False
    finally:
        if 'connector' in locals():
            await connector.disconnect()

async def run_full_test():
    """Run complete IG API test suite | 運行完整的 IG API 測試套件"""
    print("🧪 IG Markets API Test Suite")
    print("=" * 50)
    
    # Test results | 測試結果
    results = {
        'imports': False,
        'configuration': False,
        'connection': False,
        'market_data': False
    }
    
    # Run tests | 運行測試
    results['imports'] = test_imports()
    
    if results['imports']:
        results['configuration'] = test_configuration()
        
        if results['configuration']:
            results['connection'] = await test_connection()
            
            if results['connection']:
                results['market_data'] = await test_market_data()
    
    # Summary | 摘要
    print("\n" + "=" * 50)
    print("📋 Test Results Summary:")
    print("-" * 25)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\n🎯 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! IG API integration is working correctly.")
        print("\n📋 Next steps:")
        print("   - You can now use IG API in your trading strategies")
        print("   - Market data is available for analysis")
        print("   - Demo trading orders can be placed")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
        print("\n💡 Common solutions:")
        print("   - Ensure all Python packages are installed")
        print("   - Verify your IG credentials are correct") 
        print("   - Check your internet connection")
        print("   - Ensure your IG account has API access enabled")

def main():
    """Main function | 主函數"""
    try:
        asyncio.run(run_full_test())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error during testing: {e}")

if __name__ == "__main__":
    main()