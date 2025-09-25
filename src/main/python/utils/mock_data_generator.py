"""
Mock Data Generator for AIFX Trading Signals
AIFX交易信號模擬數據生成器

Generates realistic forex data when real data sources are unavailable.
當真實數據源不可用時生成真實的外匯數據。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List

class MockDataGenerator:
    """
    Generates mock forex data for testing and demonstration
    生成用於測試和演示的模擬外匯數據
    """

    def __init__(self):
        # Base prices for major currency pairs
        self.base_prices = {
            'EURUSD=X': 1.0850,
            'USDJPY=X': 149.50,
            'GBPUSD=X': 1.2650,
            'AUDUSD=X': 0.6750,
            'USDCAD=X': 1.3650,
            'USDCHF=X': 0.8850
        }

    def generate_realistic_ohlcv(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval_hours: int = 1
    ) -> pd.DataFrame:
        """
        Generate realistic OHLCV data with proper forex characteristics
        生成具有適當外匯特性的真實OHLCV數據
        """

        # Calculate number of periods
        total_hours = int((end_date - start_date).total_seconds() / 3600)
        periods = total_hours // interval_hours

        # Base price for the symbol
        base_price = self.base_prices.get(symbol, 1.0000)

        # Generate timestamps
        timestamps = []
        current_time = start_date
        for _ in range(periods):
            timestamps.append(current_time)
            current_time += timedelta(hours=interval_hours)

        # Generate price data with realistic characteristics
        data = []
        current_price = base_price

        for i, timestamp in enumerate(timestamps):
            # Add some trending and mean-reverting behavior
            trend = np.sin(i * 0.01) * 0.001  # Long-term trend
            noise = np.random.normal(0, 0.002)  # Random noise

            # Price change
            price_change = trend + noise
            current_price = current_price * (1 + price_change)

            # Generate OHLC around current price
            volatility = 0.001  # 0.1% volatility per period
            spread = current_price * volatility

            high = current_price + random.uniform(0, spread)
            low = current_price - random.uniform(0, spread)
            open_price = current_price + random.uniform(-spread/2, spread/2)
            close_price = current_price + random.uniform(-spread/2, spread/2)

            # Ensure OHLC relationships are valid
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)

            # Volume (forex doesn't have volume, but some systems expect it)
            volume = random.randint(1000000, 5000000)

            data.append({
                'Open': round(open_price, 5),
                'High': round(high, 5),
                'Low': round(low, 5),
                'Close': round(close_price, 5),
                'Volume': volume
            })

            current_price = close_price

        # Create DataFrame
        df = pd.DataFrame(data, index=pd.to_datetime(timestamps))
        df.index.name = 'Datetime'

        return df

    def generate_multiple_symbols(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate data for multiple symbols
        為多個品種生成數據
        """
        result = {}
        for symbol in symbols:
            result[symbol] = self.generate_realistic_ohlcv(symbol, start_date, end_date)

        return result

def create_mock_data_for_service(symbols: List[str], days: int = 30) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to create mock data for the trading service
    為交易服務創建模擬數據的便利函數
    """
    generator = MockDataGenerator()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    return generator.generate_multiple_symbols(symbols, start_date, end_date)