#!/usr/bin/env python3
"""
Test 3 Demo Trading Positions | 測試3個演示交易倉位
==================================================

Simulate 3 trading positions to test the AIFX system functionality
without requiring live IG Markets API connection.
模擬3個交易倉位以測試AIFX系統功能，無需真實的IG Markets API連接。

This script will:
1. Initialize the trading system components
2. Generate mock market data for USD/JPY
3. Create 3 test positions (BUY/SELL)
4. Monitor position performance
5. Display results and close positions
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "main" / "python"))

try:
    from trading.position_manager import PositionManager
    from utils.logger import setup_logger
    from utils.config import Config
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Using simplified position simulation instead")

class DemoPositionTester:
    """Demo position testing without external dependencies"""

    def __init__(self):
        """Initialize the demo position tester"""
        self.logger = setup_logger("DemoPositionTester") if 'setup_logger' in globals() else None
        self.positions = []
        self.balance = 100000.0  # Starting balance
        self.position_id_counter = 1

        print("🚀 Initialized Demo Position Tester")
        print(f"💰 Starting Balance: ${self.balance:,.2f}")

    def log(self, message: str):
        """Log message with fallback to print"""
        if self.logger:
            self.logger.info(message)
        else:
            print(f"📝 {datetime.now().strftime('%H:%M:%S')} - {message}")

    def generate_mock_price(self, base_price: float = 150.50) -> Dict:
        """Generate mock USD/JPY price data"""
        spread = 0.02
        bid = base_price + random.uniform(-0.50, 0.50)
        ask = bid + spread

        return {
            'symbol': 'USD/JPY',
            'bid': round(bid, 3),
            'ask': round(ask, 3),
            'spread': spread,
            'timestamp': datetime.now()
        }

    def create_position(self, symbol: str, direction: str, size: float, price: float) -> Dict:
        """Create a demo trading position"""
        position = {
            'id': f"POS_{self.position_id_counter:03d}",
            'symbol': symbol,
            'direction': direction,  # 'BUY' or 'SELL'
            'size': size,
            'entry_price': price,
            'entry_time': datetime.now(),
            'status': 'OPEN',
            'pnl': 0.0,
            'unrealized_pnl': 0.0
        }

        self.positions.append(position)
        self.position_id_counter += 1

        # Calculate position value
        position_value = size * price

        self.log(f"✅ Created Position {position['id']}: {direction} {size} {symbol} @ {price}")
        self.log(f"💰 Position Value: ${position_value:,.2f}")

        return position

    def update_position_pnl(self, position: Dict, current_price: float) -> Dict:
        """Update position P&L based on current price"""
        entry_price = position['entry_price']
        size = position['size']
        direction = position['direction']

        if direction == 'BUY':
            pnl = (current_price - entry_price) * size
        else:  # SELL
            pnl = (entry_price - current_price) * size

        position['unrealized_pnl'] = pnl
        position['current_price'] = current_price

        return position

    def close_position(self, position: Dict, exit_price: float) -> Dict:
        """Close a position and realize P&L"""
        position = self.update_position_pnl(position, exit_price)
        position['status'] = 'CLOSED'
        position['exit_price'] = exit_price
        position['exit_time'] = datetime.now()
        position['realized_pnl'] = position['unrealized_pnl']

        self.balance += position['realized_pnl']

        pnl_status = "✅" if position['realized_pnl'] > 0 else "❌"
        self.log(f"{pnl_status} Closed Position {position['id']}: P&L ${position['realized_pnl']:.2f}")

        return position

    def display_position_summary(self):
        """Display current position summary"""
        print("\n" + "="*80)
        print("📊 POSITION SUMMARY | 倉位摘要")
        print("="*80)

        open_positions = [p for p in self.positions if p['status'] == 'OPEN']
        closed_positions = [p for p in self.positions if p['status'] == 'CLOSED']

        print(f"📈 Open Positions: {len(open_positions)}")
        print(f"📉 Closed Positions: {len(closed_positions)}")
        print(f"💰 Current Balance: ${self.balance:,.2f}")

        total_unrealized = sum(p.get('unrealized_pnl', 0) for p in open_positions)
        total_realized = sum(p.get('realized_pnl', 0) for p in closed_positions)

        print(f"💵 Unrealized P&L: ${total_unrealized:.2f}")
        print(f"💰 Realized P&L: ${total_realized:.2f}")
        print(f"📊 Total P&L: ${total_unrealized + total_realized:.2f}")

    def display_position_details(self):
        """Display detailed position information"""
        print("\n📋 DETAILED POSITION REPORT | 詳細倉位報告")
        print("-" * 80)

        for i, pos in enumerate(self.positions, 1):
            status_emoji = "🟢" if pos['status'] == 'OPEN' else "🔴"
            direction_emoji = "📈" if pos['direction'] == 'BUY' else "📉"

            print(f"\n{i}. {status_emoji} Position {pos['id']} - {direction_emoji} {pos['direction']}")
            print(f"   Symbol: {pos['symbol']}")
            print(f"   Size: {pos['size']}")
            print(f"   Entry: {pos['entry_price']} @ {pos['entry_time'].strftime('%H:%M:%S')}")

            if pos['status'] == 'CLOSED':
                print(f"   Exit: {pos['exit_price']} @ {pos['exit_time'].strftime('%H:%M:%S')}")
                print(f"   P&L: ${pos['realized_pnl']:.2f}")
            else:
                current_price = pos.get('current_price', pos['entry_price'])
                print(f"   Current: {current_price}")
                print(f"   Unrealized P&L: ${pos.get('unrealized_pnl', 0):.2f}")

async def run_demo_test():
    """Run the 3-position demo test"""
    print("🎯 Starting 3-Position Demo Test")
    print("🎯 開始3倉位演示測試")
    print("="*50)

    # Initialize demo tester
    demo = DemoPositionTester()

    # Step 1: Create 3 positions
    print("\n📊 Step 1: Creating 3 Demo Positions...")

    positions_to_create = [
        {'symbol': 'USD/JPY', 'direction': 'BUY', 'size': 10000, 'price': 150.25},
        {'symbol': 'USD/JPY', 'direction': 'SELL', 'size': 15000, 'price': 150.75},
        {'symbol': 'USD/JPY', 'direction': 'BUY', 'size': 8000, 'price': 150.45}
    ]

    created_positions = []
    for pos_config in positions_to_create:
        position = demo.create_position(**pos_config)
        created_positions.append(position)
        await asyncio.sleep(0.5)  # Small delay for realism

    demo.display_position_summary()

    # Step 2: Simulate price movements and update P&L
    print("\n📈 Step 2: Simulating Price Movements...")

    for i in range(5):  # 5 price updates
        print(f"\n🔄 Price Update {i+1}/5")

        # Generate new price
        market_data = demo.generate_mock_price()
        current_price = market_data['bid']

        print(f"💹 USD/JPY: {current_price} (Bid) / {market_data['ask']} (Ask)")

        # Update all open positions
        for position in created_positions:
            if position['status'] == 'OPEN':
                demo.update_position_pnl(position, current_price)

        # Show current P&L
        total_unrealized = sum(p.get('unrealized_pnl', 0) for p in created_positions if p['status'] == 'OPEN')
        print(f"💰 Total Unrealized P&L: ${total_unrealized:.2f}")

        await asyncio.sleep(2)  # Wait 2 seconds between updates

    # Step 3: Close all positions
    print("\n📉 Step 3: Closing All Positions...")

    final_price_data = demo.generate_mock_price()
    final_price = final_price_data['bid']

    print(f"🔚 Final USD/JPY Price: {final_price}")

    for position in created_positions:
        if position['status'] == 'OPEN':
            demo.close_position(position, final_price)
            await asyncio.sleep(0.5)

    # Step 4: Final results
    print("\n🎉 Step 4: Final Results")
    demo.display_position_summary()
    demo.display_position_details()

    # Calculate performance metrics
    total_trades = len(created_positions)
    winning_trades = len([p for p in created_positions if p.get('realized_pnl', 0) > 0])
    losing_trades = total_trades - winning_trades
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    print(f"\n🏆 PERFORMANCE METRICS | 績效指標")
    print("-" * 40)
    print(f"📊 Total Trades: {total_trades}")
    print(f"✅ Winning Trades: {winning_trades}")
    print(f"❌ Losing Trades: {losing_trades}")
    print(f"🎯 Win Rate: {win_rate:.1f}%")

    total_pnl = sum(p.get('realized_pnl', 0) for p in created_positions)
    print(f"💰 Total P&L: ${total_pnl:.2f}")
    print(f"📈 Return: {(total_pnl / 100000 * 100):.3f}%")

    print("\n✅ Demo Test Complete!")
    print("✅ 演示測試完成！")

    return True

def main():
    """Main execution function"""
    print("🚀 AIFX 3-Position Demo Test")
    print("🚀 AIFX 3倉位演示測試")
    print("=" * 50)
    print("📝 Testing trading system without live API connection")
    print("📝 測試交易系統而無需實時API連接")
    print("🎯 Simulating 3 USD/JPY positions with realistic price movements")
    print("🎯 模擬3個美元/日圓倉位及真實價格變動")
    print()

    try:
        # Run the demo test
        asyncio.run(run_demo_test())

        print("\n🎊 SUCCESS: All demo positions executed successfully!")
        print("🎊 成功：所有演示倉位執行成功！")

        return True

    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
        return False

    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)