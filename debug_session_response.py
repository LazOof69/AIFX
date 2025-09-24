#!/usr/bin/env python3
"""
Debug IG Markets Session Response | 調試 IG Markets 會話響應
========================================================

Debug the session creation response to understand token location
調試會話創建響應以了解令牌位置
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

class IGSessionDebugger:
    """Debug IG Markets session creation response"""

    def __init__(self):
        self.base_url = "https://demo-api.ig.com/gateway/deal"
        self.credentials = None

    def load_credentials(self):
        """Load complete credentials"""
        try:
            with open('ig_demo_credentials.json', 'r') as f:
                config = json.load(f)
                self.credentials = config['ig_markets']['demo']
                return True
        except Exception as e:
            print(f"❌ Error loading credentials: {e}")
            return False

    async def debug_session_response(self):
        """Debug the complete session response"""

        print("=" * 80)
        print("🔍 IG MARKETS SESSION RESPONSE DEBUG | IG MARKETS 會話響應調試")
        print("=" * 80)
        print(f"🕐 Debug Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)")
        print("=" * 80)

        if not self.load_credentials():
            return

        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/session"
            headers = {
                "X-IG-API-KEY": self.credentials['api_key'],
                "Content-Type": "application/json; charset=UTF-8",
                "Accept": "application/json; charset=UTF-8",
                "Version": "3"
            }

            session_data = {
                "identifier": self.credentials['username'],
                "password": self.credentials['password']
            }

            try:
                print("📡 MAKING SESSION REQUEST | 發出會話請求")
                print("-" * 60)
                print(f"URL: {url}")
                print(f"Method: POST")
                print(f"API Key: {self.credentials['api_key'][:20]}...")
                print(f"Username: {self.credentials['username']}")
                print(f"Password: {'*' * len(self.credentials['password'])}")
                print()

                async with session.post(url, headers=headers, json=session_data) as response:
                    print("📊 RESPONSE ANALYSIS | 響應分析")
                    print("-" * 60)
                    print(f"Status Code: {response.status}")
                    print(f"Status Reason: {response.reason}")

                    print("\n📋 Response Headers:")
                    for name, value in response.headers.items():
                        print(f"   {name}: {value[:50]}{'...' if len(str(value)) > 50 else ''}")

                    print("\n📄 Response Body:")
                    response_text = await response.text()
                    try:
                        response_json = json.loads(response_text)
                        print(json.dumps(response_json, indent=2))
                    except:
                        print(f"Raw text: {response_text}")

                    # Check for tokens in different locations
                    print("\n🔍 TOKEN SEARCH | 令牌搜索")
                    print("-" * 60)

                    # Check response headers
                    cst_token = response.headers.get('CST')
                    security_token = response.headers.get('X-SECURITY-TOKEN')

                    print(f"CST in headers: {'✅ Found' if cst_token else '❌ Not found'}")
                    print(f"X-SECURITY-TOKEN in headers: {'✅ Found' if security_token else '❌ Not found'}")

                    if cst_token:
                        print(f"   CST Token: {cst_token[:30]}...")
                    if security_token:
                        print(f"   Security Token: {security_token[:30]}...")

                    # Check response body for tokens
                    if response.status == 200:
                        try:
                            data = json.loads(response_text)
                            print("\n🔎 Checking response body for tokens...")

                            # Common token field names
                            token_fields = ['cst', 'CST', 'sessionToken', 'token', 'authToken',
                                          'securityToken', 'X-SECURITY-TOKEN', 'lightstreamerEndpoint']

                            for field in token_fields:
                                if field in data:
                                    print(f"   {field}: {str(data[field])[:50]}...")

                            # Check nested objects
                            for key, value in data.items():
                                if isinstance(value, dict):
                                    print(f"   Nested object '{key}': {list(value.keys())}")

                        except Exception as e:
                            print(f"   Error parsing response body: {e}")

                    print("\n🎯 AUTHENTICATION STATUS | 身份驗證狀態")
                    print("-" * 60)
                    if response.status == 200:
                        print("✅ Authentication successful!")
                        print("✅ API Key is valid")
                        print("✅ Username/password accepted")

                        if cst_token and security_token:
                            print("✅ Session tokens found in headers")
                            return True
                        else:
                            print("⚠️ Session tokens not in standard header locations")
                            print("💡 Need to investigate alternative token sources")
                            return False
                    else:
                        print("❌ Authentication failed")
                        return False

            except Exception as e:
                print(f"❌ Session debug error: {e}")
                return False

async def main():
    """Main execution function"""

    print("🚀 Starting IG Markets Session Response Debug...")

    debugger = IGSessionDebugger()
    success = await debugger.debug_session_response()

    print("\n" + "=" * 80)
    print("🎉 SESSION DEBUG COMPLETED | 會話調試完成")
    print("=" * 80)

    if success:
        print("✅ Session tokens located and extracted")
        print("🎯 Ready to proceed with account balance retrieval")
    else:
        print("⚠️ Session tokens need further investigation")
        print("💡 Response structure may differ from expected format")

if __name__ == "__main__":
    asyncio.run(main())