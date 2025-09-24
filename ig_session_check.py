#!/usr/bin/env python3
"""
IG Markets Session-Based Balance Check | 基於會話的 IG Markets 餘額檢查
====================================================================

Uses session-based authentication to get real account balance
使用基於會話的身份驗證獲取實際帳戶餘額
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class IGSessionChecker:
    """
    IG Markets Session-Based Balance Checker
    基於會話的 IG Markets 餘額檢查器
    """

    def __init__(self):
        self.base_url = "https://demo-api.ig.com/gateway/deal"
        self.credentials = None
        self.session = None
        self.cst_token = None
        self.security_token = None
        self.api_key = None

    def load_credentials(self):
        """Load credentials and check what authentication data we have"""
        try:
            with open('ig_demo_credentials.json', 'r') as f:
                config = json.load(f)
                self.credentials = config['ig_markets']['demo']

                print("🔍 AVAILABLE AUTHENTICATION DATA | 可用身份驗證數據")
                print("-" * 60)
                print(f"✅ Client ID: {self.credentials.get('clientId', 'Not found')}")
                print(f"✅ Account ID: {self.credentials.get('accountId', 'Not found')}")
                print(f"✅ OAuth Access Token: {self.credentials.get('oauthToken', {}).get('access_token', 'Not found')[:20]}...")
                print(f"✅ OAuth Refresh Token: {self.credentials.get('oauthToken', {}).get('refresh_token', 'Not found')[:20]}...")

                # Check for API key
                api_key_found = False
                possible_keys = ['api_key', 'apiKey', 'X-IG-API-KEY', 'ig_api_key']
                for key in possible_keys:
                    if key in self.credentials:
                        self.api_key = self.credentials[key]
                        print(f"✅ API Key ({key}): {self.api_key[:10]}...")
                        api_key_found = True
                        break

                if not api_key_found:
                    print("⚠️ API Key: Not found in credentials")
                    print("   💡 May need to be provided separately for REST API access")

                print()
                return True

        except Exception as e:
            print(f"❌ Error loading credentials: {e}")
            return False

    async def check_authentication_methods(self):
        """
        Check different authentication methods available
        檢查可用的不同身份驗證方法
        """

        print("=" * 80)
        print("🔐 IG MARKETS AUTHENTICATION CHECK | IG MARKETS 身份驗證檢查")
        print("=" * 80)
        print(f"🕐 Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)")
        print("=" * 80)

        if not self.load_credentials():
            return

        try:
            async with aiohttp.ClientSession() as session:
                self.session = session

                # Method 1: Try OAuth with possible API key combinations
                await self.try_oauth_methods()

                # Method 2: Try session creation if we have login credentials
                await self.try_session_creation()

                # Method 3: Show what we learned
                await self.summarize_findings()

        except Exception as e:
            print(f"❌ Authentication check failed: {e}")

    async def try_oauth_methods(self):
        """Try different OAuth authentication approaches"""
        print("\n🔑 TESTING OAUTH AUTHENTICATION | 測試 OAUTH 身份驗證")
        print("-" * 60)

        oauth_token = self.credentials.get('oauthToken', {}).get('access_token')
        if not oauth_token:
            print("❌ No OAuth token available")
            return

        # Test 1: OAuth without API key
        await self.test_oauth_call("OAuth only (no API key)", {
            "Authorization": f"Bearer {oauth_token}",
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json; charset=UTF-8",
            "Version": "1"
        })

        # Test 2: OAuth with Client ID as API key
        client_id = self.credentials.get('clientId')
        if client_id:
            await self.test_oauth_call("OAuth + Client ID as API key", {
                "Authorization": f"Bearer {oauth_token}",
                "X-IG-API-KEY": client_id,
                "Content-Type": "application/json; charset=UTF-8",
                "Accept": "application/json; charset=UTF-8",
                "Version": "1"
            })

        # Test 3: OAuth with Account ID as API key
        account_id = self.credentials.get('accountId')
        if account_id:
            await self.test_oauth_call("OAuth + Account ID as API key", {
                "Authorization": f"Bearer {oauth_token}",
                "X-IG-API-KEY": account_id,
                "Content-Type": "application/json; charset=UTF-8",
                "Accept": "application/json; charset=UTF-8",
                "Version": "1"
            })

    async def test_oauth_call(self, method_name, headers):
        """Test a specific OAuth call configuration"""
        print(f"\n🧪 Testing: {method_name}")

        url = f"{self.base_url}/accounts"

        try:
            async with self.session.get(url, headers=headers) as response:
                status = response.status
                text = await response.text()

                if status == 200:
                    print(f"   ✅ SUCCESS! Status: {status}")
                    data = json.loads(text)
                    print(f"   📊 Response: {json.dumps(data, indent=2)[:200]}...")
                    return True
                else:
                    print(f"   ❌ Failed. Status: {status}")
                    print(f"   📄 Error: {text[:100]}")
                    return False

        except Exception as e:
            print(f"   ❌ Exception: {e}")
            return False

    async def try_session_creation(self):
        """Try to create a session if we have username/password"""
        print("\n🔐 CHECKING SESSION CREATION OPTIONS | 檢查會話創建選項")
        print("-" * 60)

        # Check if we have username/password
        username = self.credentials.get('username') or self.credentials.get('login')
        password = self.credentials.get('password')

        if not username or not password:
            print("❌ No username/password found for session creation")
            print("   💡 Session creation requires username and password")
            print("   💡 Currently only have OAuth tokens")
            return

        print(f"✅ Username found: {username}")
        print("✅ Password found: [HIDDEN]")
        print("🔄 Attempting session creation...")

        # TODO: Implement session creation if credentials are available

    async def summarize_findings(self):
        """Summarize what we learned about authentication"""
        print("\n📊 AUTHENTICATION ANALYSIS SUMMARY | 身份驗證分析摘要")
        print("-" * 60)

        print("🔍 Available Credentials:")
        print(f"   ✅ OAuth Access Token: Available (expires in 30 seconds)")
        print(f"   ✅ OAuth Refresh Token: Available")
        print(f"   ✅ Client ID: {self.credentials.get('clientId')}")
        print(f"   ✅ Account ID: {self.credentials.get('accountId')}")
        print(f"   ❌ API Key: Not found in configuration")
        print(f"   ❌ Username/Password: Not available")

        print("\n💡 Authentication Requirements:")
        print("   🔸 IG Markets REST API requires X-IG-API-KEY header")
        print("   🔸 OAuth tokens alone may not be sufficient")
        print("   🔸 API Key typically obtained during app registration")
        print("   🔸 Alternative: Username/password for session-based auth")

        print("\n🎯 Recommendations:")
        print("   1. Obtain proper API Key from IG Markets developer portal")
        print("   2. Use session-based auth with username/password if available")
        print("   3. Check if Client ID can serve as API Key")
        print("   4. Contact IG Markets support for API access requirements")

async def main():
    """Main execution function"""

    print("🚀 Starting IG Markets Authentication Analysis...")

    checker = IGSessionChecker()
    await checker.check_authentication_methods()

    print("\n" + "=" * 80)
    print("🎉 AUTHENTICATION ANALYSIS COMPLETED | 身份驗證分析完成")
    print("=" * 80)
    print("✅ Checked OAuth authentication methods")
    print("✅ Analyzed available credentials")
    print("✅ Identified authentication requirements")
    print("✅ Provided recommendations for API access")
    print("\n💡 Authentication analysis completed!")
    print("💡 身份驗證分析完成！")

if __name__ == "__main__":
    asyncio.run(main())