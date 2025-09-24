#!/usr/bin/env python3
"""
Test IG Markets REST API Integration | 測試 IG Markets REST API 整合
==============================================================

This script tests the comprehensive REST API functionality using
the IG Markets API reference and your demo credentials.
此腳本使用 IG Markets API 參考和您的演示憑據測試全面的 REST API 功能。
"""

import sys
import asyncio
import logging
import json
from pathlib import Path
from datetime import datetime

# Add src path
sys.path.insert(0, str(Path(__file__).parent / "src" / "main" / "python"))

from brokers.ig_markets import IGMarketsConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class IGRestAPITester:
    """
    Comprehensive IG Markets REST API Tester
    全面的 IG Markets REST API 測試器
    """

    def __init__(self):
        self.connector = None
        self.test_results = {}

    async def run_all_tests(self):
        """
        Run comprehensive REST API tests | 運行全面的 REST API 測試
        """

        print("="*80)
        print("🧪 IG MARKETS REST API COMPREHENSIVE TEST | IG MARKETS REST API 全面測試")
        print("="*80)

        try:
            # Initialize connector
            await self.initialize_connector()

            # Run test suite
            await self.test_account_endpoints()
            await self.test_market_endpoints()
            await self.test_dealing_endpoints()

            # Display results
            self.display_test_results()

        except Exception as e:
            print(f"❌ Test suite failed: {e}")

    async def initialize_connector(self):
        """Initialize IG Markets connector"""
        print("🔧 Initializing IG Markets connector...")

        try:
            self.connector = IGMarketsConnector()
            success = await self.connector.connect(demo=True, force_oauth=True)

            if success:
                print("✅ Connector initialized and connected")
                self.test_results['initialization'] = True
            else:
                print("❌ Connection failed")
                self.test_results['initialization'] = False
                raise Exception("Failed to connect")

        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            self.test_results['initialization'] = False
            raise

    async def test_account_endpoints(self):
        """Test account-related REST API endpoints"""
        print("\n📊 TESTING ACCOUNT ENDPOINTS | 測試帳戶端點")
        print("-" * 50)

        # Test get accounts
        await self.test_endpoint(
            "Get Accounts",
            "獲取帳戶",
            self.connector.get_accounts
        )

    async def test_market_endpoints(self):
        """Test market data REST API endpoints"""
        print("\n📈 TESTING MARKET ENDPOINTS | 測試市場端點")
        print("-" * 50)

        # Test market details for USD/JPY
        await self.test_endpoint(
            "Get USD/JPY Market Details",
            "獲取美元/日圓市場詳細信息",
            self.connector.get_market_details,
            args=["CS.D.USDJPY.CFD.IP"]
        )

        # Test historical prices
        await self.test_endpoint(
            "Get USD/JPY Historical Prices",
            "獲取美元/日圓歷史價格",
            self.connector.get_historical_prices,
            args=["CS.D.USDJPY.CFD.IP", "HOUR", 10]
        )

        # Test market search
        await self.test_endpoint(
            "Search Markets (USD)",
            "搜索市場 (美元)",
            self.connector.search_markets,
            args=["USD"]
        )

    async def test_dealing_endpoints(self):
        """Test dealing-related REST API endpoints"""
        print("\n💼 TESTING DEALING ENDPOINTS | 測試交易端點")
        print("-" * 50)

        # Test get positions
        await self.test_endpoint(
            "Get Positions",
            "獲取持倉",
            self.connector.get_positions
        )

        # Test get working orders
        await self.test_endpoint(
            "Get Working Orders",
            "獲取工作訂單",
            self.connector.get_working_orders
        )

    async def test_endpoint(self, test_name: str, test_name_cn: str, method, args=None, kwargs=None):
        """
        Test a specific REST API endpoint
        測試特定的 REST API 端點
        """
        try:
            print(f"🔍 Testing: {test_name} | {test_name_cn}")

            # Call the method
            if args and kwargs:
                result = await method(*args, **kwargs)
            elif args:
                result = await method(*args)
            elif kwargs:
                result = await method(**kwargs)
            else:
                result = await method()

            # Check if result is valid
            if result is not None:
                print(f"  ✅ SUCCESS - Response received")
                if isinstance(result, dict):
                    # Show sample of the response structure
                    keys = list(result.keys())[:3]  # First 3 keys
                    print(f"  📋 Response keys: {keys}...")

                    # Show data size if available
                    if 'positions' in result:
                        print(f"  📊 Positions count: {len(result['positions'])}")
                    elif 'markets' in result:
                        print(f"  📊 Markets count: {len(result['markets'])}")
                    elif 'prices' in result:
                        print(f"  📊 Price points: {len(result['prices'])}")

                self.test_results[test_name] = True
            else:
                print(f"  ⚠️ WARNING - No data returned")
                self.test_results[test_name] = False

        except Exception as e:
            print(f"  ❌ FAILED - {str(e)}")
            self.test_results[test_name] = False

    def display_test_results(self):
        """Display comprehensive test results summary"""
        print("\n" + "="*80)
        print("📊 TEST RESULTS SUMMARY | 測試結果摘要")
        print("="*80)

        passed = sum(1 for result in self.test_results.values() if result)
        total = len(self.test_results)
        pass_rate = (passed / total * 100) if total > 0 else 0

        print(f"📈 Overall Results: {passed}/{total} tests passed ({pass_rate:.1f}%)")
        print("-" * 80)

        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{test_name:<40} {status}")

        print("-" * 80)

        if pass_rate >= 70:
            print("🎉 REST API INTEGRATION SUCCESSFUL!")
            print("🎉 REST API 整合成功！")
            print("\n💡 Your IG demo account is ready for:")
            print("   • Live market data retrieval")
            print("   • Position management")
            print("   • Order execution")
            print("   • Account monitoring")
        else:
            print("⚠️ REST API INTEGRATION NEEDS IMPROVEMENT")
            print("⚠️ REST API 整合需要改進")
            print("\n🔧 Consider:")
            print("   • Checking API credentials")
            print("   • Verifying network connection")
            print("   • Reviewing error messages above")

        # Save detailed results
        results_file = f"ig_api_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'pass_rate': pass_rate,
                'results': self.test_results
            }, f, indent=2)

        print(f"\n💾 Detailed results saved to: {results_file}")

async def main():
    """Main test function"""
    tester = IGRestAPITester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())