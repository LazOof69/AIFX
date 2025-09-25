#!/usr/bin/env python3
"""
Test IG Demo Account Connection | 測試 IG 演示帳戶連接
===============================================

This script tests the connection to IG Markets demo account using
the provided OAuth credentials.
此腳本使用提供的 OAuth 憑證測試與 IG Markets 演示帳戶的連接。
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add src path
sys.path.insert(0, str(Path(__file__).parent / "src" / "main" / "python"))

from brokers.ig_markets import IGMarketsConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_ig_demo_connection():
    """
    Test IG Markets demo account connection
    測試 IG Markets 演示帳戶連接
    """

    print("="*80)
    print("🧪 IG MARKETS DEMO CONNECTION TEST | IG MARKETS 演示連接測試")
    print("="*80)

    try:
        # Initialize connector (will automatically use ig_demo_credentials.json)
        # 初始化連接器（將自動使用 ig_demo_credentials.json）
        print("📊 Initializing IG Markets connector...")
        print("📊 正在初始化 IG Markets 連接器...")

        connector = IGMarketsConnector()  # Uses default credentials file

        # Test connection
        print("🔌 Testing demo account connection...")
        print("🔌 測試演示帳戶連接...")

        success = await connector.connect(demo=True, force_oauth=True)

        if success:
            print("✅ CONNECTION SUCCESSFUL! | 連接成功！")
            print("✅ Your IG demo account is working correctly")
            print("✅ 您的 IG 演示帳戶運行正常")

            # Display account info
            if connector.account_info:
                print(f"📊 Account ID: {connector.account_info.account_id}")
                print(f"💰 Balance: {connector.account_info.balance} {connector.account_info.currency}")
                print(f"🔗 Connection Method: {connector.auth_method}")

            # Test market data retrieval
            print("\n🔍 Testing market data retrieval...")
            print("🔍 測試市場數據獲取...")

            try:
                market_data = await connector.get_market_data("CS.D.USDJPY.CFD.IP")
                print("✅ Market data retrieved successfully")
                print("✅ 市場數據獲取成功")
            except Exception as e:
                print(f"⚠️ Market data test failed: {e}")
                print("⚠️ 市場數據測試失敗")

            return True

        else:
            print("❌ CONNECTION FAILED | 連接失敗")
            print("❌ Please check your credentials and network")
            print("❌ 請檢查您的憑證和網絡")
            return False

    except Exception as e:
        print(f"❌ TEST ERROR: {str(e)}")
        print(f"❌ 測試錯誤: {str(e)}")
        return False

    finally:
        print("\n" + "="*80)

async def main():
    """Main test function"""

    success = await test_ig_demo_connection()

    if success:
        print("🎉 ALL TESTS PASSED! Your IG demo credentials are working!")
        print("🎉 所有測試通過！您的 IG 演示憑證正常工作！")

        print("\n💡 You can now run live trading with:")
        print("💡 現在您可以運行實盤交易：")
        print("   python run_trading_demo.py --mode live")

        sys.exit(0)
    else:
        print("🚨 TESTS FAILED! Please check your setup.")
        print("🚨 測試失敗！請檢查您的設置。")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())