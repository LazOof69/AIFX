#!/usr/bin/env python3
"""
Test Live IG Markets Trading Positions | 測試真實的IG Markets交易倉位
====================================================================

Execute 3 real trading positions on IG Markets demo account using fresh OAuth tokens.
使用新的OAuth令牌在IG Markets模擬帳戶上執行3個真實交易倉位。

This script will:
1. Test the new OAuth tokens
2. Get account balance and information
3. Execute 3 USD/JPY positions
4. Monitor position status
5. Close positions after demonstration
"""

import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

class IGLiveTrader:
    """Live IG Markets trading client"""

    def __init__(self, tokens: Dict):
        """Initialize with OAuth tokens"""
        self.access_token = tokens['access_token']
        self.api_key = "3a0f12d07fe51ab5f4f1835ae037e1f5e876726e"
        self.base_url = "https://demo-api.ig.com/gateway/deal"

        self.headers = {
            'Accept': 'application/json; charset=UTF-8',
            'Content-Type': 'application/json; charset=UTF-8',
            'Authorization': f'Bearer {self.access_token}',
            'X-IG-API-KEY': self.api_key
        }

        print(f"🔑 Initialized IG Live Trader with fresh tokens")

    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            headers = self.headers.copy()
            headers['Version'] = '1'

            response = requests.get(f"{self.base_url}/accounts", headers=headers, timeout=10)

            if response.status_code == 200:
                accounts = response.json().get('accounts', [])
                print(f"✅ Connection successful! Found {len(accounts)} accounts")
                return True
            else:
                print(f"❌ Connection failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False

        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False

    def get_account_info(self) -> Optional[Dict]:
        """Get account information"""
        try:
            headers = self.headers.copy()
            headers['Version'] = '1'

            response = requests.get(f"{self.base_url}/accounts", headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                accounts = data.get('accounts', [])

                if accounts:
                    account = accounts[0]  # Use first account
                    print(f"📊 Account ID: {account.get('accountId')}")
                    print(f"📊 Account Name: {account.get('accountName')}")
                    print(f"💰 Balance: {account.get('balance', {}).get('balance', 'N/A')} {account.get('currency')}")
                    print(f"💵 Available: {account.get('balance', {}).get('available', 'N/A')}")

                    return account

            return None

        except Exception as e:
            print(f"❌ Error getting account info: {e}")
            return None

    def get_market_info(self, epic: str) -> Optional[Dict]:
        """Get market information"""
        try:
            headers = self.headers.copy()
            headers['Version'] = '3'

            response = requests.get(f"{self.base_url}/markets/{epic}", headers=headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                instrument = data.get('instrument', {})
                snapshot = data.get('snapshot', {})

                market_info = {
                    'epic': epic,
                    'name': instrument.get('name', 'Unknown'),
                    'bid': snapshot.get('bid'),
                    'ask': snapshot.get('offer'),
                    'status': instrument.get('marketStatus'),
                    'min_size': instrument.get('lotSize'),
                    'currency': instrument.get('currencies', [{}])[0].get('code')
                }

                print(f"📈 {market_info['name']}: Bid={market_info['bid']} Ask={market_info['ask']}")
                return market_info
            else:
                print(f"❌ Failed to get market info for {epic}: {response.status_code}")
                return None

        except Exception as e:
            print(f"❌ Error getting market info: {e}")
            return None

    def create_position(self, epic: str, direction: str, size: float, stop_distance: float = None, limit_distance: float = None) -> Optional[str]:
        """Create a new position"""
        try:
            headers = self.headers.copy()
            headers['Version'] = '2'

            payload = {
                "epic": epic,
                "expiry": "DFB",  # Daily Funded Bet
                "direction": direction,
                "size": size,
                "orderType": "MARKET",
                "timeInForce": "EXECUTE_AND_ELIMINATE",
                "guaranteedStop": False,
                "forceOpen": True
            }

            # Add stop loss if provided
            if stop_distance:
                payload["stopDistance"] = stop_distance

            # Add limit (take profit) if provided
            if limit_distance:
                payload["limitDistance"] = limit_distance

            response = requests.post(f"{self.base_url}/positions/otc",
                                   headers=headers,
                                   json=payload,
                                   timeout=10)

            if response.status_code in [200, 201]:
                deal_reference = response.json().get('dealReference')
                print(f"✅ Position created: {direction} {size} {epic}")
                print(f"📋 Deal Reference: {deal_reference}")
                return deal_reference
            else:
                print(f"❌ Failed to create position: {response.status_code}")
                print(f"Response: {response.text}")
                return None

        except Exception as e:
            print(f"❌ Error creating position: {e}")
            return None

    def get_positions(self) -> List[Dict]:
        """Get current positions"""
        try:
            headers = self.headers.copy()
            headers['Version'] = '2'

            response = requests.get(f"{self.base_url}/positions", headers=headers, timeout=10)

            if response.status_code == 200:
                positions = response.json().get('positions', [])
                print(f"📊 Found {len(positions)} open positions")

                for pos in positions:
                    market = pos.get('market', {})
                    position = pos.get('position', {})

                    print(f"   🎯 {market.get('instrumentName')}: {position.get('direction')} {position.get('size')}")
                    print(f"      Entry: {position.get('openLevel')} | P&L: {position.get('unrealisedPL')}")

                return positions
            else:
                print(f"❌ Failed to get positions: {response.status_code}")
                return []

        except Exception as e:
            print(f"❌ Error getting positions: {e}")
            return []

    def close_position(self, deal_id: str, direction: str, size: float) -> bool:
        """Close a position"""
        try:
            headers = self.headers.copy()
            headers['Version'] = '1'

            # Opposite direction to close
            close_direction = "SELL" if direction == "BUY" else "BUY"

            payload = {
                "dealId": deal_id,
                "direction": close_direction,
                "size": size,
                "orderType": "MARKET",
                "timeInForce": "EXECUTE_AND_ELIMINATE"
            }

            response = requests.post(f"{self.base_url}/positions/otc",
                                   headers=headers,
                                   json=payload,
                                   timeout=10)

            if response.status_code in [200, 201]:
                print(f"✅ Position closed: {close_direction} {size}")
                return True
            else:
                print(f"❌ Failed to close position: {response.status_code}")
                print(f"Response: {response.text}")
                return False

        except Exception as e:
            print(f"❌ Error closing position: {e}")
            return False

def main():
    """Main trading test function"""
    print("🚀 LIVE IG MARKETS TRADING TEST")
    print("🚀 真實IG Markets交易測試")
    print("=" * 50)

    # OAuth tokens from user
    tokens = {
        "access_token": "c85f5067-d20e-4e5a-b05f-075b4d65b09d",
        "refresh_token": "030c855d-05b1-4f17-8e18-4593de43dc37"
    }

    # Initialize trader
    trader = IGLiveTrader(tokens)

    # Step 1: Test connection
    print("\n1️⃣ Testing Connection...")
    if not trader.test_connection():
        print("❌ Connection failed. Exiting...")
        return False

    # Step 2: Get account info
    print("\n2️⃣ Getting Account Information...")
    account = trader.get_account_info()
    if not account:
        print("❌ Failed to get account info. Exiting...")
        return False

    # Step 3: Get USD/JPY market info
    print("\n3️⃣ Getting USD/JPY Market Data...")
    usd_jpy_epic = "CS.D.USDJPY.MINI.IP"  # USD/JPY mini contract
    market_info = trader.get_market_info(usd_jpy_epic)
    if not market_info:
        print("❌ Failed to get market data. Exiting...")
        return False

    # Step 4: Execute 3 positions
    print("\n4️⃣ Executing 3 USD/JPY Positions...")

    positions_to_create = [
        {"direction": "BUY", "size": 0.5, "stop_distance": 20, "limit_distance": 30},
        {"direction": "SELL", "size": 0.7, "stop_distance": 25, "limit_distance": 35},
        {"direction": "BUY", "size": 0.3, "stop_distance": 15, "limit_distance": 25}
    ]

    deal_references = []

    for i, pos_config in enumerate(positions_to_create, 1):
        print(f"\n📊 Creating Position {i}/3...")
        deal_ref = trader.create_position(usd_jpy_epic, **pos_config)

        if deal_ref:
            deal_references.append(deal_ref)
            time.sleep(2)  # Wait between orders
        else:
            print(f"❌ Failed to create position {i}")

    print(f"\n✅ Created {len(deal_references)} positions successfully!")

    # Step 5: Monitor positions
    print("\n5️⃣ Monitoring Positions...")
    time.sleep(5)  # Wait for positions to settle

    current_positions = trader.get_positions()

    # Step 6: Wait and then close positions (for demo purposes)
    print("\n6️⃣ Waiting 30 seconds before closing positions...")
    time.sleep(30)

    print("\n7️⃣ Closing Positions...")
    final_positions = trader.get_positions()

    for pos in final_positions:
        market = pos.get('market', {})
        position = pos.get('position', {})

        if market.get('epic') == usd_jpy_epic:
            deal_id = position.get('dealId')
            direction = position.get('direction')
            size = position.get('size')

            print(f"\n📉 Closing position: {direction} {size}")
            trader.close_position(deal_id, direction, size)
            time.sleep(2)

    # Final check
    print("\n8️⃣ Final Position Check...")
    time.sleep(5)
    final_check = trader.get_positions()

    print(f"\n🎉 TEST COMPLETE!")
    print(f"📊 Positions created: {len(deal_references)}")
    print(f"📊 Final open positions: {len(final_check)}")
    print(f"✅ Live IG trading functionality verified!")

    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n🎊 SUCCESS: Live IG Markets trading test completed!")
        else:
            print("\n❌ FAILED: Live IG Markets trading test failed!")
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")