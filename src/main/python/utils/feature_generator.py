"""
AIFX Feature Generator | AIFX特徵生成器

This module generates features for AI models by combining technical indicators,
market data preprocessing, and time-based feature engineering.
此模組通過結合技術指標、市場數據預處理和基於時間的特徵工程為AI模型生成特徵。

The FeatureGenerator bridges the gap between raw market data and AI model input.
FeatureGenerator在原始市場數據和AI模型輸入之間建立橋樑。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
warnings.filterwarnings('ignore')

from .logger import get_logger
from .data_preprocessor import DataPreprocessor
from .technical_indicators import TechnicalIndicators


class FeatureGenerator:
    """
    Feature generation class for AI models | AI模型特徵生成類
    
    Combines technical indicators, market data preprocessing, and time-based
    features to create comprehensive feature sets for AI model training and prediction.
    結合技術指標、市場數據預處理和基於時間的特徵，為AI模型訓練和預測創建綜合特徵集。
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize feature generator | 初始化特徵生成器
        
        Args:
            lookback_periods: List of periods for technical indicators | 技術指標的週期列表
        """
        self.logger = get_logger(__name__)
        self.data_preprocessor = DataPreprocessor()
        self.technical_indicators = TechnicalIndicators()
        
        # Default lookback periods for technical indicators | 技術指標的默認回望期
        self.lookback_periods = lookback_periods or [5, 10, 20, 50]
        
        # Feature names for tracking | 用於跟蹤的特徵名稱
        self.feature_names = []
        
        self.logger.info("✅ FeatureGenerator initialized | 特徵生成器已初始化")
    
    def generate_features(self, market_data: pd.DataFrame, symbol: str = "EURUSD") -> pd.DataFrame:
        """
        Generate comprehensive features from market data | 從市場數據生成綜合特徵
        
        Args:
            market_data: OHLCV market data | OHLCV市場數據
            symbol: Trading symbol | 交易品種
            
        Returns:
            DataFrame with generated features | 包含生成特徵的數據框
        """
        try:
            if market_data.empty:
                self.logger.warning("⚠️ Empty market data provided | 提供了空的市場數據")
                return pd.DataFrame()
            
            # Ensure required columns exist | 確保所需列存在
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in market_data.columns for col in required_columns):
                self.logger.error(f"❌ Missing required columns in market data | 市場數據中缺少必需列")
                return pd.DataFrame()
            
            self.logger.info(f"🔄 Generating features for {symbol} with {len(market_data)} data points")
            
            # Start with original market data | 從原始市場數據開始
            features_df = market_data.copy()
            
            # 1. Basic price features | 基本價格特徵
            features_df = self._add_price_features(features_df)
            
            # 2. Technical indicators | 技術指標
            features_df = self._add_technical_indicators(features_df)
            
            # 3. Time-based features | 基於時間的特徵
            features_df = self._add_time_features(features_df)
            
            # 4. Volatility features | 波動率特徵
            features_df = self._add_volatility_features(features_df)
            
            # 5. Momentum features | 動量特徵
            features_df = self._add_momentum_features(features_df)
            
            # 6. Volume features | 成交量特徵
            features_df = self._add_volume_features(features_df)
            
            # 7. Statistical features | 統計特徵
            features_df = self._add_statistical_features(features_df)
            
            # Clean and validate features | 清理和驗證特徵
            features_df = self._clean_features(features_df)
            
            # Store feature names | 存儲特徵名稱
            self.feature_names = list(features_df.columns)
            
            self.logger.info(f"✅ Generated {len(features_df.columns)} features for {symbol}")
            return features_df
            
        except Exception as e:
            self.logger.error(f"❌ Error generating features: {str(e)}")
            return pd.DataFrame()
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-related features | 添加基本價格相關特徵"""
        try:
            # Price changes | 價格變化
            df['price_change'] = df['Close'] - df['Open']
            df['price_change_pct'] = (df['Close'] - df['Open']) / df['Open'] * 100
            
            # High-Low range | 最高最低價範圍
            df['hl_range'] = df['High'] - df['Low']
            df['hl_range_pct'] = (df['High'] - df['Low']) / df['Open'] * 100
            
            # Body and shadow ratios | 實體和影線比率
            df['body_size'] = abs(df['Close'] - df['Open'])
            df['upper_shadow'] = df['High'] - np.maximum(df['Open'], df['Close'])
            df['lower_shadow'] = np.minimum(df['Open'], df['Close']) - df['Low']
            
            # Typical price | 典型價格
            df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
            
            return df
        except Exception as e:
            self.logger.error(f"❌ Error adding price features: {str(e)}")
            return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators | 添加技術指標"""
        try:
            # Moving averages | 移動平均線
            for period in self.lookback_periods:
                if len(df) >= period:
                    # Simple Moving Average
                    df[f'sma_{period}'] = self.technical_indicators.sma(df['Close'], period)
                    # Exponential Moving Average
                    df[f'ema_{period}'] = self.technical_indicators.ema(df['Close'], period)
                    
                    # Price relative to moving averages | 價格相對於移動平均線
                    df[f'close_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
                    df[f'close_ema_{period}_ratio'] = df['Close'] / df[f'ema_{period}']
            
            # RSI | 相對強弱指數
            df['rsi_14'] = self.technical_indicators.rsi(df['Close'], 14)
            df['rsi_30'] = self.technical_indicators.rsi(df['Close'], 30)
            
            # MACD | 指數平滑移動平均線
            macd_result = self.technical_indicators.macd(df['Close'])
            df['macd'] = macd_result['macd']
            df['macd_signal'] = macd_result['signal']
            df['macd_histogram'] = macd_result['histogram']
            
            # Bollinger Bands | 布林帶
            bb_result = self.technical_indicators.bollinger_bands(df['Close'])
            df['bb_upper'] = bb_result['upper']
            df['bb_middle'] = bb_result['middle']
            df['bb_lower'] = bb_result['lower']
            df['bb_width'] = bb_result['width']
            df['bb_position'] = (df['Close'] - bb_result['lower']) / (bb_result['upper'] - bb_result['lower'])
            
            # ATR | 平均真實範圍
            df['atr_14'] = self.technical_indicators.atr(df['High'], df['Low'], df['Close'], 14)
            
            return df
        except Exception as e:
            self.logger.error(f"❌ Error adding technical indicators: {str(e)}")
            return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features | 添加基於時間的特徵"""
        try:
            if df.index.dtype != 'datetime64[ns]':
                # Try to convert index to datetime if it's not already | 如果還不是，嘗試將索引轉換為datetime
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    self.logger.warning("⚠️ Could not convert index to datetime | 無法將索引轉換為datetime")
                    return df
            
            # Hour of day | 一天中的小時
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
            
            # Trading session indicators | 交易時段指標
            # European session (7-16 GMT) | 歐洲時段
            df['european_session'] = ((df.index.hour >= 7) & (df.index.hour <= 16)).astype(int)
            # American session (13-22 GMT) | 美國時段
            df['american_session'] = ((df.index.hour >= 13) & (df.index.hour <= 22)).astype(int)
            # Asian session (21-6 GMT next day) | 亞洲時段
            df['asian_session'] = ((df.index.hour >= 21) | (df.index.hour <= 6)).astype(int)
            
            # Weekend indicator | 週末指標
            df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            return df
        except Exception as e:
            self.logger.error(f"❌ Error adding time features: {str(e)}")
            return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features | 添加波動率特徵"""
        try:
            # Price volatility over different periods | 不同週期的價格波動率
            for period in [5, 10, 20]:
                if len(df) >= period:
                    returns = df['Close'].pct_change()
                    df[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(period)
            
            # Parkinson's volatility (using High and Low) | 帕金森波動率（使用最高價和最低價）
            df['parkinson_vol'] = np.sqrt(0.361 * np.log(df['High'] / df['Low']) ** 2)
            
            # Garman-Klass volatility | 加曼-克拉斯波動率
            df['gk_volatility'] = 0.5 * np.log(df['High'] / df['Low']) ** 2 - (2 * np.log(2) - 1) * np.log(df['Close'] / df['Open']) ** 2
            
            return df
        except Exception as e:
            self.logger.error(f"❌ Error adding volatility features: {str(e)}")
            return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features | 添加動量特徵"""
        try:
            # Rate of Change over different periods | 不同週期的變化率
            for period in [5, 10, 20]:
                if len(df) >= period:
                    df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period) * 100
            
            # Williams %R | 威廉指標
            for period in [14, 30]:
                if len(df) >= period:
                    highest_high = df['High'].rolling(window=period).max()
                    lowest_low = df['Low'].rolling(window=period).min()
                    df[f'williams_r_{period}'] = (highest_high - df['Close']) / (highest_high - lowest_low) * -100
            
            # Commodity Channel Index | 商品通道指數
            df['cci'] = self.technical_indicators.cci(df['High'], df['Low'], df['Close'])
            
            return df
        except Exception as e:
            self.logger.error(f"❌ Error adding momentum features: {str(e)}")
            return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features | 添加基於成交量的特徵"""
        try:
            if 'Volume' not in df.columns or df['Volume'].sum() == 0:
                # For forex data, volume might not be meaningful, add proxy features | 對於外匯數據，成交量可能沒有意義，添加代理特徵
                df['volume_proxy'] = df['hl_range'] * 1000  # Use range as volume proxy | 使用範圍作為成交量代理
                return df
            
            # Volume moving averages | 成交量移動平均線
            for period in [5, 10, 20]:
                if len(df) >= period:
                    df[f'volume_sma_{period}'] = df['Volume'].rolling(window=period).mean()
                    df[f'volume_ratio_{period}'] = df['Volume'] / df[f'volume_sma_{period}']
            
            # On-Balance Volume | 能量潮
            df['obv'] = self.technical_indicators.obv(df['Close'], df['Volume'])
            
            # Volume Price Trend | 成交量價格趨勢
            df['vpt'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1) * df['Volume']).cumsum()
            
            return df
        except Exception as e:
            self.logger.error(f"❌ Error adding volume features: {str(e)}")
            return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features | 添加統計特徵"""
        try:
            # Rolling statistics | 滾動統計
            for period in [10, 20]:
                if len(df) >= period:
                    returns = df['Close'].pct_change()
                    
                    # Skewness and Kurtosis | 偏度和峰度
                    df[f'skewness_{period}'] = returns.rolling(window=period).skew()
                    df[f'kurtosis_{period}'] = returns.rolling(window=period).kurt()
                    
                    # Z-score | Z分數
                    df[f'zscore_{period}'] = (df['Close'] - df['Close'].rolling(window=period).mean()) / df['Close'].rolling(window=period).std()
            
            # Support and Resistance levels | 支撑和阻力位
            df['support_20'] = df['Low'].rolling(window=20).min()
            df['resistance_20'] = df['High'].rolling(window=20).max()
            df['support_distance'] = (df['Close'] - df['support_20']) / df['Close'] * 100
            df['resistance_distance'] = (df['resistance_20'] - df['Close']) / df['Close'] * 100
            
            return df
        except Exception as e:
            self.logger.error(f"❌ Error adding statistical features: {str(e)}")
            return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features | 清理和驗證特徵"""
        try:
            # Replace infinite values with NaN | 將無限值替換為NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill NaN values (limited) | 前向填充NaN值（有限）
            df = df.fillna(method='ffill', limit=5)
            
            # Drop rows with too many NaN values | 刪除NaN值過多的行
            nan_threshold = 0.5  # Allow up to 50% NaN values | 允許最多50%的NaN值
            df = df.dropna(thresh=int(len(df.columns) * (1 - nan_threshold)))
            
            # Remove columns with all NaN values | 刪除全部為NaN值的列
            df = df.dropna(axis=1, how='all')
            
            self.logger.info(f"🔄 Feature cleaning complete: {df.shape[0]} rows, {df.shape[1]} features")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Error cleaning features: {str(e)}")
            return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names | 獲取特徵名稱列表"""
        return self.feature_names.copy()
    
    def get_feature_importance_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get feature importance data for analysis | 獲取用於分析的特徵重要性數據"""
        try:
            feature_info = {}
            
            for col in df.columns:
                if col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    continue  # Skip original OHLCV columns | 跳過原始OHLCV列
                
                feature_info[col] = {
                    'type': self._classify_feature_type(col),
                    'non_null_count': df[col].count(),
                    'null_percentage': (df[col].isnull().sum() / len(df)) * 100,
                    'mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else None,
                    'std': df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else None
                }
            
            return feature_info
            
        except Exception as e:
            self.logger.error(f"❌ Error getting feature importance data: {str(e)}")
            return {}
    
    def _classify_feature_type(self, feature_name: str) -> str:
        """Classify feature type based on name | 根據名稱分類特徵類型"""
        feature_name = feature_name.lower()
        
        if any(term in feature_name for term in ['sma', 'ema', 'ma_']):
            return 'moving_average'
        elif any(term in feature_name for term in ['rsi', 'macd', 'cci', 'williams']):
            return 'momentum'
        elif any(term in feature_name for term in ['bb_', 'bollinger', 'atr', 'volatility']):
            return 'volatility'
        elif any(term in feature_name for term in ['volume', 'obv', 'vpt']):
            return 'volume'
        elif any(term in feature_name for term in ['hour', 'day_', 'month', 'session', 'weekend']):
            return 'time_based'
        elif any(term in feature_name for term in ['price', 'change', 'range', 'body', 'shadow']):
            return 'price_based'
        elif any(term in feature_name for term in ['skewness', 'kurtosis', 'zscore', 'support', 'resistance']):
            return 'statistical'
        else:
            return 'other'


# For backward compatibility | 為了向後兼容
def create_feature_generator(lookback_periods: List[int] = None) -> FeatureGenerator:
    """Create a FeatureGenerator instance | 創建FeatureGenerator實例"""
    return FeatureGenerator(lookback_periods)