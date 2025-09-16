#!/usr/bin/env python3
"""
Execute 3 Live Trades on IG Markets | 在IG Markets執行3筆實時交易
===============================================================

Execute 3 real USD/JPY positions on your IG demo account.
在您的IG模擬帳戶上執行3個真實的美元/日圓倉位。
"""

import requests
import json
import time
from datetime import datetime

def load_session_tokens():
    """Load session tokens from file"""
    try:
        with open('ig_session_tokens.json', 'r') as f:
            return json.load(f)
    except:
        return None

def get_market_info(tokens, epic):
    """Get current market information"""
    headers = {
        'Accept': 'application/json; charset=UTF-8',
        'X-IG-API-KEY': tokens['api_key'],
        'CST': tokens['cst'],
        'X-SECURITY-TOKEN': tokens['x_security_token'],
        'Version': '3'
    }

    try:
        response = requests.get(f'https://demo-api.ig.com/gateway/deal/markets/{epic}',
                              headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            snapshot = data.get('snapshot', {})
            instrument = data.get('instrument', {})

            return {
                'bid': snapshot.get('bid'),
                'ask': snapshot.get('offer'),
                'name': instrument.get('name'),
                'currency': instrument.get('currencies', [{}])[0].get('code', 'USD'),
                'min_size': instrument.get('lotSize', 0.1)
            }
        else:
            print(f"❌ Market info error: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"❌ Market info exception: {e}")
        return None

def create_position(tokens, epic, direction, size, currency="USD"):
    """Create a trading position"""
    headers = {
        'Accept': 'application/json; charset=UTF-8',
        'Content-Type': 'application/json; charset=UTF-8',
        'X-IG-API-KEY': tokens['api_key'],
        'CST': tokens['cst'],
        'X-SECURITY-TOKEN': tokens['x_security_token'],
        'Version': '2'
    }

    # Updated payload with currency code
    payload = {
        "epic": epic,
        "expiry": "DFB",
        "direction": direction,
        "size": size,
        "orderType": "MARKET",
        "timeInForce": "EXECUTE_AND_ELIMINATE",
        "currencyCode": currency,  # Add currency code
        "guaranteedStop": False,
        "forceOpen": True,
        "stopDistance": 25,  # 25 pips stop loss
        "limitDistance": 40   # 40 pips take profit
    }

    try:
        response = requests.post('https://demo-api.ig.com/gateway/deal/positions/otc',
                               json=payload,
                               headers=headers,
                               timeout=15)

        print(f"📊 Trade Response: {response.status_code}")

        if response.status_code in [200, 201]:
            trade_data = response.json()
            deal_reference = trade_data.get('dealReference')
            print(f"✅ Position created: {direction} {size} {epic}")
            print(f"📋 Deal Reference: {deal_reference}")
            return deal_reference
        else:
            print(f"❌ Trade failed: {response.text}")
            return None

    except Exception as e:
        print(f"❌ Trade exception: {e}")
        return None

def get_positions(tokens):
    """Get current open positions"""
    headers = {
        'Accept': 'application/json; charset=UTF-8',
        'X-IG-API-KEY': tokens['api_key'],
        'CST': tokens['cst'],
        'X-SECURITY-TOKEN': tokens['x_security_token'],
        'Version': '2'
    }

    try:
        response = requests.get('https://demo-api.ig.com/gateway/deal/positions',
                              headers=headers, timeout=10)

        if response.status_code == 200:
            positions = response.json().get('positions', [])
            return positions
        else:
            print(f"❌ Position query error: {response.status_code}")
            return []

    except Exception as e:
        print(f"❌ Position query exception: {e}")
        return []

def display_positions(positions):
    """Display current positions"""
    if not positions:
        print("📊 No open positions")
        return

    print(f"📊 Found {len(positions)} open positions:")

    total_pnl = 0
    for i, pos in enumerate(positions, 1):
        market = pos.get('market', {})
        position = pos.get('position', {})

        name = market.get('instrumentName', 'Unknown')
        direction = position.get('direction', 'N/A')
        size = position.get('size', 0)
        open_level = position.get('openLevel', 0)
        pnl = position.get('unrealisedPL', 0)
        total_pnl += pnl

        direction_emoji = "📈" if direction == "BUY" else "📉"
        pnl_emoji = "✅" if pnl >= 0 else "❌"

        print(f"   {i}. {direction_emoji} {name}: {direction} {size} @ {open_level}")
        print(f"      {pnl_emoji} P&L: ${pnl:.2f}")

    print(f"💰 Total Unrealized P&L: ${total_pnl:.2f}")

def main():
    """Execute 3 trading positions"""
    print("🚀 EXECUTING 3 LIVE USD/JPY TRADES")
    print("🚀 執行3筆實時美元/日圓交易")
    print("=" * 50)

    # Load tokens
    tokens = load_session_tokens()
    if not tokens:
        print("❌ No session tokens found. Run create_session_tokens.py first")
        return False

    print("✅ Session tokens loaded")

    # USD/JPY epic
    usd_jpy_epic = "CS.D.USDJPY.MINI.IP"

    # Step 1: Get market info
    print(f"\n1️⃣ Getting USD/JPY market information...")
    market_info = get_market_info(tokens, usd_jpy_epic)
    if not market_info:
        print("❌ Failed to get market info")
        return False

    print(f"📈 {market_info['name']}")
    print(f"💹 Bid: {market_info['bid']} | Ask: {market_info['ask']}")
    print(f"💱 Currency: {market_info['currency']}")
    print(f"📊 Min Size: {market_info['min_size']}")

    # Step 2: Execute 3 positions
    print(f"\n2️⃣ Creating 3 USD/JPY positions...")

    positions_to_create = [
        {"direction": "BUY", "size": 0.2},
        {"direction": "SELL", "size": 0.3},
        {"direction": "BUY", "size": 0.1}
    ]

    deal_references = []
    currency = market_info['currency']

    for i, pos_config in enumerate(positions_to_create, 1):
        print(f"\n📊 Creating Position {i}/3...")
        print(f"   Direction: {pos_config['direction']}")
        print(f"   Size: {pos_config['size']}")

        deal_ref = create_position(tokens, usd_jpy_epic, pos_config['direction'], pos_config['size'], currency)

        if deal_ref:
            deal_references.append(deal_ref)
            print(f"✅ Position {i} created successfully!")
        else:
            print(f"❌ Position {i} failed!")

        time.sleep(3)  # Wait between orders

    print(f"\n📊 Created {len(deal_references)} out of 3 positions")

    # Step 3: Check positions
    print(f"\n3️⃣ Checking open positions...")
    time.sleep(5)  # Wait for positions to appear

    current_positions = get_positions(tokens)
    display_positions(current_positions)

    # Step 4: Monitor for a short time
    print(f"\n4️⃣ Monitoring positions for 60 seconds...")
    for i in range(12):  # 12 * 5 = 60 seconds
        time.sleep(5)
        positions = get_positions(tokens)

        if i % 3 == 0:  # Update every 15 seconds
            print(f"   📊 Update {i//3 + 1}/4:")
            display_positions(positions)

    # Final summary
    print(f"\n5️⃣ Final Summary...")
    final_positions = get_positions(tokens)
    display_positions(final_positions)

    # Show account balance change
    print(f"\n💰 Account Status:")
    account_response = requests.get('https://demo-api.ig.com/gateway/deal/accounts',
                                  headers={
                                      'Accept': 'application/json; charset=UTF-8',
                                      'X-IG-API-KEY': tokens['api_key'],
                                      'CST': tokens['cst'],
                                      'X-SECURITY-TOKEN': tokens['x_security_token'],
                                      'Version': '1'
                                  }, timeout=10)

    if account_response.status_code == 200:
        accounts = account_response.json().get('accounts', [])
        if accounts:
            account = accounts[0]
            balance = account.get('balance', {})
            print(f"💰 Current Balance: ${balance.get('balance', 'N/A')}")
            print(f"💵 Available: ${balance.get('available', 'N/A')}")

    print(f"\n🎉 LIVE TRADING TEST COMPLETE!")
    print(f"📊 Positions executed: {len(deal_references)}")
    print(f"📈 Open positions: {len(final_positions)}")
    print(f"✅ Live IG Markets trading verified!")

    return len(deal_references) > 0

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎊 SUCCESS: Live trading execution completed!")
    else:
        print("\n❌ FAILED: Live trading execution failed!")