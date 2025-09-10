"""
AIFX Technical Indicators | AIFX技術指標

This module provides comprehensive technical analysis indicators for forex trading.
此模組為外匯交易提供全面的技術分析指標。

Features | 功能:
- Moving averages (SMA, EMA, WMA) | 移動平均線
- Momentum indicators (RSI, MACD, Stochastic) | 動量指標
- Volatility indicators (Bollinger Bands, ATR) | 波動率指標
- Volume indicators | 成交量指標
- Custom forex-specific indicators | 自定義外匯專用指標
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

from .logger import get_logger


class TechnicalIndicators:
    """
    Technical indicators calculation class | 技術指標計算類
    """
    
    def __init__(self):
        """
        Initialize technical indicators calculator | 初始化技術指標計算器
        """
        self.logger = get_logger("TechnicalIndicators")
    
    # =============================================================================
    # MOVING AVERAGES | 移動平均線
    # =============================================================================
    
    def sma(self, data: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average | 簡單移動平均線
        
        Args:
            data: Price series | 價格序列
            period: Moving average period | 移動平均週期
            
        Returns:
            SMA series | SMA序列
        """
        return data.rolling(window=period, min_periods=1).mean()
    
    def ema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average | 指數移動平均線
        
        Args:
            data: Price series | 價格序列
            period: EMA period | EMA週期
            
        Returns:
            EMA series | EMA序列
        """
        return data.ewm(span=period, adjust=False).mean()
    
    def wma(self, data: pd.Series, period: int) -> pd.Series:
        """
        Weighted Moving Average | 加權移動平均線
        
        Args:
            data: Price series | 價格序列
            period: WMA period | WMA週期
            
        Returns:
            WMA series | WMA序列
        """
        weights = np.arange(1, period + 1)
        return data.rolling(window=period).apply(
            lambda x: np.dot(x, weights) / weights.sum() if len(x) == period else np.nan,
            raw=True
        )
    
    def dema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Double Exponential Moving Average | 雙指數移動平均線
        
        Args:
            data: Price series | 價格序列
            period: DEMA period | DEMA週期
            
        Returns:
            DEMA series | DEMA序列
        """
        ema1 = self.ema(data, period)
        ema2 = self.ema(ema1, period)
        return 2 * ema1 - ema2
    
    def tema(self, data: pd.Series, period: int) -> pd.Series:
        """
        Triple Exponential Moving Average | 三重指數移動平均線
        
        Args:
            data: Price series | 價格序列
            period: TEMA period | TEMA週期
            
        Returns:
            TEMA series | TEMA序列
        """
        ema1 = self.ema(data, period)
        ema2 = self.ema(ema1, period)
        ema3 = self.ema(ema2, period)
        return 3 * ema1 - 3 * ema2 + ema3
    
    # =============================================================================
    # MOMENTUM INDICATORS | 動量指標
    # =============================================================================
    
    def rsi(self, data: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index | 相對強弱指標
        
        Args:
            data: Price series | 價格序列
            period: RSI period | RSI週期
            
        Returns:
            RSI series (0-100) | RSI序列（0-100）
        """
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Fill initial NaN with neutral value | 用中性值填充初始NaN
    
    def macd(
        self,
        data: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence) | MACD指標
        
        Args:
            data: Price series | 價格序列
            fast_period: Fast EMA period | 快速EMA週期
            slow_period: Slow EMA period | 慢速EMA週期
            signal_period: Signal line EMA period | 信號線EMA週期
            
        Returns:
            Dictionary with MACD, Signal, and Histogram | 包含MACD、信號線和柱狀圖的字典
        """
        fast_ema = self.ema(data, fast_period)
        slow_ema = self.ema(data, slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = self.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Dict[str, pd.Series]:
        """
        Stochastic Oscillator | 隨機振蕩器
        
        Args:
            high: High price series | 最高價序列
            low: Low price series | 最低價序列
            close: Close price series | 收盤價序列
            k_period: %K period | %K週期
            d_period: %D period | %D週期
            
        Returns:
            Dictionary with %K and %D lines | 包含%K和%D線的字典
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k_percent': k_percent.fillna(50),
            'd_percent': d_percent.fillna(50)
        }
    
    def williams_r(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Williams %R | 威廉指標
        
        Args:
            high: High price series | 最高價序列
            low: Low price series | 最低價序列
            close: Close price series | 收盤價序列
            period: Williams %R period | 威廉指標週期
            
        Returns:
            Williams %R series (-100 to 0) | 威廉指標序列（-100到0）
        """
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        wr = -100 * ((highest_high - close) / (highest_high - lowest_low))
        
        return wr.fillna(-50)
    
    def momentum(self, data: pd.Series, period: int = 10) -> pd.Series:
        """
        Price Momentum | 價格動量
        
        Args:
            data: Price series | 價格序列
            period: Momentum period | 動量週期
            
        Returns:
            Momentum series | 動量序列
        """
        return data / data.shift(period) - 1
    
    def roc(self, data: pd.Series, period: int = 10) -> pd.Series:
        """
        Rate of Change | 變化率
        
        Args:
            data: Price series | 價格序列
            period: ROC period | ROC週期
            
        Returns:
            ROC series (percentage) | ROC序列（百分比）
        """
        return ((data - data.shift(period)) / data.shift(period)) * 100
    
    # =============================================================================
    # VOLATILITY INDICATORS | 波動率指標
    # =============================================================================
    
    def bollinger_bands(
        self,
        data: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Bollinger Bands | 布林帶
        
        Args:
            data: Price series | 價格序列
            period: Moving average period | 移動平均週期
            std_dev: Standard deviation multiplier | 標準差倍數
            
        Returns:
            Dictionary with upper, middle, and lower bands | 包含上軌、中軌和下軌的字典
        """
        sma = self.sma(data, period)
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Band width and position | 帶寬和位置
        band_width = (upper_band - lower_band) / sma
        band_position = (data - lower_band) / (upper_band - lower_band)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band,
            'width': band_width,
            'position': band_position.fillna(0.5)
        }
    
    def atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """
        Average True Range | 平均真實區間
        
        Args:
            high: High price series | 最高價序列
            low: Low price series | 最低價序列
            close: Close price series | 收盤價序列
            period: ATR period | ATR週期
            
        Returns:
            ATR series | ATR序列
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr.fillna(true_range)
    
    def keltner_channels(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20,
        atr_period: int = 10,
        multiplier: float = 2.0
    ) -> Dict[str, pd.Series]:
        """
        Keltner Channels | 肯特納通道
        
        Args:
            high: High price series | 最高價序列
            low: Low price series | 最低價序列
            close: Close price series | 收盤價序列
            period: EMA period | EMA週期
            atr_period: ATR period | ATR週期
            multiplier: ATR multiplier | ATR倍數
            
        Returns:
            Dictionary with upper, middle, and lower channels | 包含上通道、中通道和下通道的字典
        """
        typical_price = (high + low + close) / 3
        middle_line = self.ema(typical_price, period)
        atr_value = self.atr(high, low, close, atr_period)
        
        upper_channel = middle_line + (atr_value * multiplier)
        lower_channel = middle_line - (atr_value * multiplier)
        
        return {
            'upper': upper_channel,
            'middle': middle_line,
            'lower': lower_channel
        }
    
    def donchian_channels(
        self,
        high: pd.Series,
        low: pd.Series,
        period: int = 20
    ) -> Dict[str, pd.Series]:
        """
        Donchian Channels | 唐奇安通道
        
        Args:
            high: High price series | 最高價序列
            low: Low price series | 最低價序列
            period: Channel period | 通道週期
            
        Returns:
            Dictionary with upper, middle, and lower channels | 包含上通道、中通道和下通道的字典
        """
        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()
        middle_channel = (upper_channel + lower_channel) / 2
        
        return {
            'upper': upper_channel,
            'middle': middle_channel,
            'lower': lower_channel
        }
    
    # =============================================================================
    # VOLUME INDICATORS | 成交量指標
    # =============================================================================
    
    def volume_sma(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Volume Simple Moving Average | 成交量簡單移動平均
        """
        return volume.rolling(window=period).mean()
    
    def volume_ratio(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        Volume Ratio | 成交量比率
        """
        vol_sma = self.volume_sma(volume, period)
        return volume / vol_sma
    
    def on_balance_volume(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On Balance Volume | 能量潮指標
        
        Args:
            close: Close price series | 收盤價序列
            volume: Volume series | 成交量序列
            
        Returns:
            OBV series | OBV序列
        """
        price_change = close.diff()
        obv = volume.copy()
        
        obv[price_change < 0] = -volume[price_change < 0]
        obv[price_change == 0] = 0
        
        return obv.cumsum()
    
    # =============================================================================
    # FOREX-SPECIFIC INDICATORS | 外匯專用指標
    # =============================================================================
    
    def currency_strength_index(
        self,
        data_dict: Dict[str, pd.DataFrame],
        base_currency: str = 'USD'
    ) -> Dict[str, pd.Series]:
        """
        Currency Strength Index | 貨幣強度指標
        Calculate relative strength of currencies based on multiple pairs
        基於多個貨幣對計算貨幣的相對強度
        
        Args:
            data_dict: Dictionary of currency pair DataFrames | 貨幣對DataFrame字典
            base_currency: Base currency for comparison | 比較基準貨幣
            
        Returns:
            Dictionary of currency strength indices | 貨幣強度指標字典
        """
        currencies = set()
        for pair in data_dict.keys():
            if len(pair) >= 6:
                currencies.add(pair[:3])  # First currency | 第一個貨幣
                currencies.add(pair[3:6])  # Second currency | 第二個貨幣
        
        strength_dict = {}
        
        for currency in currencies:
            if currency == base_currency:
                continue
                
            strength_values = []
            
            for pair, df in data_dict.items():
                if len(pair) >= 6:
                    base_curr = pair[:3]
                    quote_curr = pair[3:6]
                    
                    if currency == base_curr:
                        # Currency is base, so strength = 1/price change | 貨幣是基準，強度=1/價格變化
                        strength_values.append(df['Close'].pct_change())
                    elif currency == quote_curr:
                        # Currency is quote, so strength = -price change | 貨幣是報價，強度=-價格變化
                        strength_values.append(-df['Close'].pct_change())
            
            if strength_values:
                # Average strength across all pairs | 所有貨幣對的平均強度
                avg_strength = pd.concat(strength_values, axis=1).mean(axis=1)
                strength_dict[currency] = avg_strength.rolling(20).mean()
        
        return strength_dict
    
    def pivot_points(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> Dict[str, pd.Series]:
        """
        Pivot Points | 樞軸點
        
        Args:
            high: High price series | 最高價序列
            low: Low price series | 最低價序列
            close: Close price series | 收盤價序列
            
        Returns:
            Dictionary with pivot levels | 包含樞軸水平的字典
        """
        # Use previous day's data | 使用前一日數據
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)
        
        # Standard pivot point | 標準樞軸點
        pivot = (prev_high + prev_low + prev_close) / 3
        
        # Support and resistance levels | 支撐和阻力水平
        s1 = 2 * pivot - prev_high
        r1 = 2 * pivot - prev_low
        s2 = pivot - (prev_high - prev_low)
        r2 = pivot + (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
        r3 = prev_high + 2 * (pivot - prev_low)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    # =============================================================================
    # COMPOSITE INDICATORS | 複合指標
    # =============================================================================
    
    def add_all_indicators(
        self,
        df: pd.DataFrame,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Add all technical indicators to DataFrame | 向DataFrame添加所有技術指標
        
        Args:
            df: OHLCV DataFrame | OHLCV DataFrame
            include_volume: Include volume-based indicators | 包含基於成交量的指標
            
        Returns:
            DataFrame with all indicators | 包含所有指標的DataFrame
        """
        result_df = df.copy()
        
        try:
            # Moving averages | 移動平均線
            for period in [5, 10, 20, 50, 100]:
                result_df[f'sma_{period}'] = self.sma(df['Close'], period)
                result_df[f'ema_{period}'] = self.ema(df['Close'], period)
            
            # Price ratios to moving averages | 價格與移動平均線比率
            for period in [10, 20, 50]:
                result_df[f'close_sma_{period}_ratio'] = df['Close'] / result_df[f'sma_{period}']
                result_df[f'close_ema_{period}_ratio'] = df['Close'] / result_df[f'ema_{period}']
            
            # MACD | MACD指標
            macd_dict = self.macd(df['Close'])
            for key, value in macd_dict.items():
                result_df[f'macd_{key}'] = value
            
            # RSI | RSI指標
            for period in [7, 14, 21]:
                result_df[f'rsi_{period}'] = self.rsi(df['Close'], period)
            
            # Stochastic | 隨機指標
            stoch_dict = self.stochastic(df['High'], df['Low'], df['Close'])
            for key, value in stoch_dict.items():
                result_df[f'stoch_{key}'] = value
            
            # Williams %R | 威廉指標
            result_df['williams_r'] = self.williams_r(df['High'], df['Low'], df['Close'])
            
            # Bollinger Bands | 布林帶
            bb_dict = self.bollinger_bands(df['Close'])
            for key, value in bb_dict.items():
                result_df[f'bb_{key}'] = value
            
            # ATR | 平均真實區間
            result_df['atr'] = self.atr(df['High'], df['Low'], df['Close'])
            result_df['atr_20'] = self.atr(df['High'], df['Low'], df['Close'], 20)
            
            # Keltner Channels | 肯特納通道
            kelt_dict = self.keltner_channels(df['High'], df['Low'], df['Close'])
            for key, value in kelt_dict.items():
                result_df[f'keltner_{key}'] = value
            
            # Momentum indicators | 動量指標
            for period in [5, 10, 20]:
                result_df[f'momentum_{period}'] = self.momentum(df['Close'], period)
                result_df[f'roc_{period}'] = self.roc(df['Close'], period)
            
            # Volume indicators | 成交量指標
            if include_volume and 'Volume' in df.columns:
                result_df['volume_sma_10'] = self.volume_sma(df['Volume'], 10)
                result_df['volume_sma_20'] = self.volume_sma(df['Volume'], 20)
                result_df['volume_ratio_10'] = self.volume_ratio(df['Volume'], 10)
                result_df['volume_ratio_20'] = self.volume_ratio(df['Volume'], 20)
                result_df['obv'] = self.on_balance_volume(df['Close'], df['Volume'])
            
            # Pivot points | 樞軸點
            pivot_dict = self.pivot_points(df['High'], df['Low'], df['Close'])
            for key, value in pivot_dict.items():
                result_df[f'pivot_{key}'] = value
            
            self.logger.info(f"Added technical indicators | 已添加技術指標: {len(result_df.columns) - len(df.columns)} new features")
            
        except Exception as e:
            self.logger.error(f"Error adding technical indicators | 添加技術指標時出錯: {str(e)}")
            return df
        
        return result_df
    
    def get_trading_signals(
        self,
        df: pd.DataFrame,
        symbol: str = "FOREX"
    ) -> pd.DataFrame:
        """
        Generate basic trading signals from indicators | 從指標生成基本交易信號
        
        Args:
            df: DataFrame with technical indicators | 包含技術指標的DataFrame
            symbol: Trading symbol for logging | 用於日誌的交易品種
            
        Returns:
            DataFrame with signal columns | 包含信號列的DataFrame
        """
        signals_df = df.copy()
        
        try:
            # RSI signals | RSI信號
            if 'rsi_14' in df.columns:
                signals_df['signal_rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
                signals_df['signal_rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
            
            # MACD signals | MACD信號
            if all(col in df.columns for col in ['macd_macd', 'macd_signal']):
                signals_df['signal_macd_bullish'] = (
                    (df['macd_macd'] > df['macd_signal']) & 
                    (df['macd_macd'].shift(1) <= df['macd_signal'].shift(1))
                ).astype(int)
                
                signals_df['signal_macd_bearish'] = (
                    (df['macd_macd'] < df['macd_signal']) & 
                    (df['macd_macd'].shift(1) >= df['macd_signal'].shift(1))
                ).astype(int)
            
            # Bollinger Bands signals | 布林帶信號
            if all(col in df.columns for col in ['bb_upper', 'bb_lower']):
                signals_df['signal_bb_lower_touch'] = (df['Close'] <= df['bb_lower']).astype(int)
                signals_df['signal_bb_upper_touch'] = (df['Close'] >= df['bb_upper']).astype(int)
            
            # Moving average crossover signals | 移動平均交叉信號
            if all(col in df.columns for col in ['sma_10', 'sma_20']):
                signals_df['signal_ma_golden_cross'] = (
                    (df['sma_10'] > df['sma_20']) & 
                    (df['sma_10'].shift(1) <= df['sma_20'].shift(1))
                ).astype(int)
                
                signals_df['signal_ma_death_cross'] = (
                    (df['sma_10'] < df['sma_20']) & 
                    (df['sma_10'].shift(1) >= df['sma_20'].shift(1))
                ).astype(int)
            
            # Stochastic signals | 隨機指標信號
            if all(col in df.columns for col in ['stoch_k_percent', 'stoch_d_percent']):
                signals_df['signal_stoch_oversold'] = (
                    (df['stoch_k_percent'] < 20) & (df['stoch_d_percent'] < 20)
                ).astype(int)
                
                signals_df['signal_stoch_overbought'] = (
                    (df['stoch_k_percent'] > 80) & (df['stoch_d_percent'] > 80)
                ).astype(int)
            
            signal_cols = [col for col in signals_df.columns if col.startswith('signal_')]
            self.logger.info(f"Generated {len(signal_cols)} trading signals for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error generating trading signals | 生成交易信號時出錯: {str(e)}")
            return df
        
        return signals_df
    
    # =============================================================================
    # MISSING METHOD ALIASES | 缺失方法別名
    # =============================================================================
    
    def cci(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 20
    ) -> pd.Series:
        """
        Commodity Channel Index | 商品通道指標
        
        Args:
            high: High price series | 最高價序列
            low: Low price series | 最低價序列  
            close: Close price series | 收盤價序列
            period: Period for calculation | 計算週期
            
        Returns:
            CCI series | CCI序列
        """
        try:
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mean_deviation = typical_price.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=False
            )
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            return cci.fillna(0)
        except Exception as e:
            self.logger.error(f"Error calculating CCI | 計算CCI時出錯: {str(e)}")
            return pd.Series(index=close.index, data=0)
    
    def obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume alias for on_balance_volume | 能量潮指標別名
        
        Args:
            close: Close price series | 收盤價序列
            volume: Volume series | 成交量序列
            
        Returns:
            OBV series | OBV序列
        """
        return self.on_balance_volume(close, volume)


# Convenience functions | 便利函數
def calculate_all_indicators(df: pd.DataFrame, include_volume: bool = True) -> pd.DataFrame:
    """
    Convenience function to calculate all technical indicators | 計算所有技術指標的便利函數
    
    Args:
        df: OHLCV DataFrame | OHLCV DataFrame
        include_volume: Include volume indicators | 包含成交量指標
        
    Returns:
        DataFrame with all indicators | 包含所有指標的DataFrame
    """
    ti = TechnicalIndicators()
    return ti.add_all_indicators(df, include_volume)


def get_basic_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to get basic trading signals | 獲取基本交易信號的便利函數
    
    Args:
        df: DataFrame with indicators | 包含指標的DataFrame
        
    Returns:
        DataFrame with signals | 包含信號的DataFrame
    """
    ti = TechnicalIndicators()
    return ti.get_trading_signals(df)


if __name__ == "__main__":
    # Example usage | 使用示例
    from ..utils.data_loader import DataLoader
    
    # Load sample data | 載入樣本數據
    loader = DataLoader()
    data = loader.download_data(['EURUSD'], period='3m', interval='1h')
    
    if 'EURUSD' in data:
        # Calculate all indicators | 計算所有指標
        ti = TechnicalIndicators()
        df_with_indicators = ti.add_all_indicators(data['EURUSD'])
        
        print(f"Original columns | 原始列數: {len(data['EURUSD'].columns)}")
        print(f"With indicators | 包含指標: {len(df_with_indicators.columns)}")
        
        # Generate signals | 生成信號
        df_with_signals = ti.get_trading_signals(df_with_indicators, 'EURUSD')
        signal_cols = [col for col in df_with_signals.columns if col.startswith('signal_')]
        print(f"Generated signals | 生成的信號: {len(signal_cols)}")
        
        # Show recent data | 顯示最近數據
        print("\\nRecent indicators | 最近指標:")
        indicator_cols = ['Close', 'rsi_14', 'macd_macd', 'bb_upper', 'bb_lower', 'atr']
        available_cols = [col for col in indicator_cols if col in df_with_indicators.columns]
        print(df_with_indicators[available_cols].tail())