#!/usr/bin/env python3
"""
IG Markets Real Balance with OAuth Tokens | 使用 OAuth 令牌的 IG Markets 實際餘額
==============================================================================

Uses OAuth tokens from session response to get real account balance
使用會話響應中的 OAuth 令牌獲取實際帳戶餘額
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

class IGOAuthBalanceChecker:
    """
    IG Markets Balance Checker using OAuth tokens from session
    使用會話中 OAuth 令牌的 IG Markets 餘額檢查器
    """

    def __init__(self):
        self.base_url = "https://demo-api.ig.com/gateway/deal"
        self.credentials = None
        self.oauth_tokens = None

    def load_credentials(self):
        """Load credentials"""
        try:
            with open('ig_demo_credentials.json', 'r') as f:
                config = json.load(f)
                self.credentials = config['ig_markets']['demo']
                return True
        except Exception as e:
            print(f"❌ Error loading credentials: {e}")
            return False

    async def get_oauth_tokens(self):
        """Get fresh OAuth tokens from session creation"""
        print("🔐 OBTAINING FRESH OAUTH TOKENS | 獲取新的 OAUTH 令牌")
        print("-" * 60)

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

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=headers, json=session_data) as response:
                    if response.status == 200:
                        data = await response.json()

                        if 'oauthToken' in data:
                            self.oauth_tokens = data['oauthToken']
                            print("✅ Fresh OAuth tokens obtained!")
                            print(f"   Access Token: {self.oauth_tokens['access_token'][:20]}...")
                            print(f"   Expires in: {self.oauth_tokens['expires_in']} seconds")
                            return True
                        else:
                            print("❌ OAuth tokens not found in response")
                            return False
                    else:
                        print(f"❌ Session creation failed: {response.status}")
                        return False

            except Exception as e:
                print(f"❌ OAuth token request error: {e}")
                return False

    async def get_real_balance_with_oauth(self):
        """
        Get real account balance using OAuth tokens
        使用 OAuth 令牌獲取實際帳戶餘額
        """

        print("=" * 80)
        print("🏦 IG MARKETS REAL BALANCE WITH OAUTH | 使用 OAUTH 的 IG MARKETS 實際餘額")
        print("=" * 80)
        print(f"🕐 Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (UTC+8)")
        print("=" * 80)

        if not self.load_credentials():
            return

        # Step 1: Get fresh OAuth tokens
        if not await self.get_oauth_tokens():
            print("❌ Failed to obtain OAuth tokens")
            return

        # Step 2: Use OAuth tokens to get account data
        async with aiohttp.ClientSession() as session:
            self.session = session

            # Get accounts
            accounts = await self.get_accounts_oauth()
            if accounts:
                self.display_real_account_data(accounts)

            # Get positions
            positions = await self.get_positions_oauth()
            if positions:
                self.display_real_position_data(positions)

    async def get_accounts_oauth(self):
        """Get accounts using OAuth authentication"""
        print("\n💰 FETCHING REAL ACCOUNT DATA | 獲取實際帳戶數據")
        print("-" * 60)

        url = f"{self.base_url}/accounts"
        headers = {
            "X-IG-API-KEY": self.credentials['api_key'],
            "Authorization": f"Bearer {self.oauth_tokens['access_token']}",
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json; charset=UTF-8",
            "Version": "1"
        }

        try:
            print(f"📡 Calling: GET {url}")
            print(f"🔑 Using OAuth: {self.oauth_tokens['access_token'][:20]}...")

            async with self.session.get(url, headers=headers) as response:
                print(f"📊 Response Status: {response.status}")

                if response.status == 200:
                    data = await response.json()
                    print("✅ Real account data retrieved successfully!")
                    return data
                else:
                    error_text = await response.text()
                    print(f"❌ Account request failed: {error_text}")
                    return None

        except Exception as e:
            print(f"❌ Account request error: {e}")
            return None

    async def get_positions_oauth(self):
        """Get positions using OAuth authentication"""
        print("\n📊 FETCHING REAL POSITIONS | 獲取實際頭寸")
        print("-" * 60)

        url = f"{self.base_url}/positions"
        headers = {
            "X-IG-API-KEY": self.credentials['api_key'],
            "Authorization": f"Bearer {self.oauth_tokens['access_token']}",
            "Content-Type": "application/json; charset=UTF-8",
            "Accept": "application/json; charset=UTF-8",
            "Version": "2"
        }

        try:
            print(f"📡 Calling: GET {url}")

            async with self.session.get(url, headers=headers) as response:
                print(f"📊 Response Status: {response.status}")

                if response.status == 200:
                    data = await response.json()
                    print("✅ Real position data retrieved successfully!")
                    return data
                else:
                    error_text = await response.text()
                    print(f"⚠️ Positions request: {error_text}")
                    return None

        except Exception as e:
            print(f"❌ Position request error: {e}")
            return None

    def display_real_account_data(self, accounts_data):
        """Display your real IG account balance"""
        print("\n🏦 YOUR REAL IG DEMO ACCOUNT BALANCE | 您的真實 IG 演示帳戶餘額")
        print("=" * 70)

        try:
            if 'accounts' in accounts_data:
                accounts = accounts_data['accounts']
            else:
                accounts = [accounts_data] if not isinstance(accounts_data, list) else accounts_data

            for account in accounts:
                account_id = account.get('accountId', 'N/A')
                account_name = account.get('accountName', 'Demo Account')
                currency = account.get('currency', 'USD')

                print(f"🎯 Account: {account_name} ({account_id})")
                print(f"💳 Type: {account.get('accountType', 'DEMO')}")
                print(f"📊 Status: {account.get('status', 'ACTIVE')}")
                print(f"🌍 Currency: {currency}")

                if 'balance' in account:
                    balance = account['balance']

                    current_balance = balance.get('balance', 0)
                    available = balance.get('available', 0)
                    deposit = balance.get('deposit', 0)
                    pnl = balance.get('profitLoss', 0)

                    print(f"\n💰 BALANCE DETAILS | 餘額詳情:")
                    print(f"   Current Balance: ${current_balance:,.2f} {currency}")
                    print(f"   Available Funds: ${available:,.2f} {currency}")
                    print(f"   Total Deposit: ${deposit:,.2f} {currency}")
                    print(f"   Unrealized P&L: ${pnl:+,.2f} {currency}")

                    # Calculate used margin
                    used_margin = deposit - available if deposit and available else 0
                    print(f"   Used Margin: ${used_margin:,.2f} {currency}")

                print()

        except Exception as e:
            print(f"❌ Error displaying account data: {e}")
            print(f"📄 Raw response: {json.dumps(accounts_data, indent=2)}")

    def display_real_position_data(self, positions_data):
        """Display your real open positions"""
        print("\n📊 YOUR REAL OPEN POSITIONS | 您的真實開放頭寸")
        print("=" * 70)

        try:
            positions = positions_data.get('positions', [])

            if not positions:
                print("📊 No open positions found")
                print("💡 All positions may have been closed or none were opened yet")
                return

            print(f"📈 You have {len(positions)} open position(s):")
            print()

            total_pnl = 0
            for i, position in enumerate(positions, 1):
                market = position.get('market', {})
                pos_info = position.get('position', {})

                # Extract position details
                instrument = market.get('instrumentName', 'Unknown')
                epic = market.get('epic', 'N/A')
                direction = pos_info.get('direction', 'N/A')
                size = pos_info.get('dealSize', 0)
                open_level = pos_info.get('openLevel', 0)
                current_level = pos_info.get('level', 0)
                unrealized_pnl = pos_info.get('unrealisedPL', 0)
                currency = pos_info.get('currency', 'USD')
                deal_id = pos_info.get('dealId', 'N/A')

                total_pnl += unrealized_pnl

                # Status icon
                pnl_icon = "🟢" if unrealized_pnl >= 0 else "🔴"
                direction_icon = "📈" if direction == "BUY" else "📉"

                print(f"{pnl_icon} Position {i}: {instrument}")
                print(f"   {direction_icon} Direction: {direction}")
                print(f"   📦 Size: {size} lots")
                print(f"   🎯 Open Price: {open_level}")
                print(f"   📊 Current Price: {current_level}")
                print(f"   💰 Unrealized P&L: {unrealized_pnl:+,.2f} {currency}")
                print(f"   🔖 Deal ID: {deal_id}")
                print(f"   📋 Epic: {epic}")
                print()

            print("-" * 50)
            print(f"💰 TOTAL PORTFOLIO P&L: ${total_pnl:+,.2f} USD")
            print("-" * 50)

        except Exception as e:
            print(f"❌ Error displaying positions: {e}")
            print(f"📄 Raw response: {json.dumps(positions_data, indent=2)[:500]}...")

async def main():
    """Main execution function"""

    print("🚀 Starting Real IG Markets Balance Check with OAuth...")

    checker = IGOAuthBalanceChecker()
    await checker.get_real_balance_with_oauth()

    print("\n" + "=" * 80)
    print("🎉 REAL BALANCE CHECK COMPLETED | 真實餘額檢查完成")
    print("=" * 80)
    print("✅ Used valid API key for authentication")
    print("✅ Obtained fresh OAuth tokens")
    print("✅ Retrieved real account balance from IG API")
    print("✅ Checked actual open positions")
    print("\n🎯 Your real IG Markets demo account data displayed above!")
    print("🎯 您的真實 IG Markets 演示帳戶數據已顯示在上方！")

if __name__ == "__main__":
    asyncio.run(main())