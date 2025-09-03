"""
AIFX Data Preprocessing Pipeline | AIFX數據預處理管道

This module provides data preprocessing capabilities for the AIFX trading system.
此模組為AIFX交易系統提供數據預處理功能。

Features | 功能:
- Data cleaning and validation | 數據清理和驗證
- Feature engineering | 特徵工程
- Data normalization and scaling | 數據標準化和縮放
- Time-based feature extraction | 基於時間的特徵提取
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

from .logger import get_logger
from .config import Config


class DataPreprocessor:
    """
    Data preprocessing class for AIFX system | AIFX系統數據預處理類
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize data preprocessor | 初始化數據預處理器
        
        Args:
            config: Configuration object | 配置對象
        """
        self.config = config or Config()
        self.logger = get_logger("DataPreprocessor")
        
        # Scalers for different features | 不同特徵的縮放器
        self.scalers = {
            'price': StandardScaler(),
            'volume': RobustScaler(),
            'returns': StandardScaler(),
            'indicators': MinMaxScaler()
        }
        
        # Feature names for tracking | 用於跟踪的特徵名稱
        self.feature_columns = []
        self.target_column = 'target'
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        add_features: bool = True,
        normalize: bool = True,
        remove_outliers: bool = True,
        create_target: bool = True
    ) -> pd.DataFrame:
        """
        Complete data preprocessing pipeline | 完整數據預處理管道
        
        Args:
            data: Input OHLCV DataFrame | 輸入OHLCV DataFrame
            add_features: Add technical and time-based features | 添加技術和時間特徵
            normalize: Apply normalization to features | 對特徵應用標準化
            remove_outliers: Remove statistical outliers | 移除統計異常值
            create_target: Create target variable for prediction | 創建預測目標變量
            
        Returns:
            Processed DataFrame with features | 包含特徵的處理後DataFrame
        """
        self.logger.info(f"Starting data preprocessing | 開始數據預處理: {len(data)} records")
        
        df = data.copy()
        
        # 1. Basic data validation | 基礎數據驗證
        df = self._validate_ohlcv_data(df)
        
        # 2. Add basic price features | 添加基礎價格特徵
        df = self._add_price_features(df)
        
        # 3. Add time-based features | 添加時間特徵
        if add_features:
            df = self._add_time_features(df)
        
        # 4. Add technical indicators | 添加技術指標
        if add_features:
            df = self._add_basic_technical_features(df)
        
        # 5. Create target variable | 創建目標變量
        if create_target:
            df = self._create_target_variable(df)
        
        # 6. Remove outliers | 移除異常值
        if remove_outliers:
            df = self._remove_outliers(df)
        
        # 7. Handle missing values | 處理缺失值
        df = self._handle_missing_values(df)
        
        # 8. Normalize features | 標準化特徵
        if normalize:
            df = self._normalize_features(df)
        
        # 9. Final validation | 最終驗證
        df = self._final_validation(df)
        
        self.logger.info(f"Data preprocessing completed | 數據預處理完成: {len(df)} records, {len(df.columns)} features")
        
        return df
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate OHLCV data integrity | 驗證OHLCV數據完整性
        """
        # Ensure required columns exist | 確保必需列存在
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns | 缺少必需列: {missing_cols}")
        
        # Validate OHLC relationships | 驗證OHLC關係
        invalid_high = df['High'] < df[['Open', 'Close']].max(axis=1)
        invalid_low = df['Low'] > df[['Open', 'Close']].min(axis=1)
        
        if invalid_high.any() or invalid_low.any():
            invalid_count = (invalid_high | invalid_low).sum()
            self.logger.warning(f"Found {invalid_count} invalid OHLC relationships | 發現{invalid_count}個無效OHLC關係")
            
            # Fix invalid relationships | 修復無效關係
            df.loc[invalid_high, 'High'] = df.loc[invalid_high, ['Open', 'Close']].max(axis=1)
            df.loc[invalid_low, 'Low'] = df.loc[invalid_low, ['Open', 'Close']].min(axis=1)
        
        # Remove rows with zero or negative prices | 移除零價或負價行
        valid_prices = (df[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)
        df = df[valid_prices]
        
        return df.sort_index()
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic price-derived features | 添加基礎價格衍生特徵
        """
        # Typical price | 典型價格
        df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Price ranges | 價格區間
        df['hl_range'] = df['High'] - df['Low']
        df['oc_range'] = abs(df['Open'] - df['Close'])
        df['ho_range'] = abs(df['High'] - df['Open'])
        df['lc_range'] = abs(df['Low'] - df['Close'])
        
        # Price position within range | 價格在區間內的位置
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        df['close_position'] = df['close_position'].fillna(0.5)
        
        # Returns | 收益率
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility (rolling standard deviation of returns) | 波動率（收益率的滾動標準差）
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features | 添加時間特徵
        """
        # Hour of day | 一天中的小時
        df['hour'] = df.index.hour
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week | 一週中的天
        df['day_of_week'] = df.index.dayofweek
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month | 月份
        df['month'] = df.index.month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Trading session indicators | 交易時段指標
        # European session: 7-16 UTC | 歐洲時段
        # US session: 13-22 UTC | 美國時段  
        # Asian session: 23-8 UTC | 亞洲時段
        df['european_session'] = ((df['hour'] >= 7) & (df['hour'] < 16)).astype(int)
        df['us_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        df['asian_session'] = ((df['hour'] >= 23) | (df['hour'] < 8)).astype(int)
        
        # Market overlap periods | 市場重疊期間
        df['euro_us_overlap'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        df['asia_euro_overlap'] = ((df['hour'] >= 7) & (df['hour'] < 8)).astype(int)
        
        return df
    
    def _add_basic_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical indicator features | 添加基礎技術指標特徵
        Note: This adds simple technical indicators. More advanced indicators 
        will be implemented in the technical_indicators.py module.
        注意：這添加簡單的技術指標。更高級的指標將在technical_indicators.py模組中實現。
        """
        # Simple Moving Averages | 簡單移動平均線
        periods = [5, 10, 20, 50]
        for period in periods:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'price_sma_{period}_ratio'] = df['Close'] / df[f'sma_{period}']
        
        # Exponential Moving Averages | 指數移動平均線
        for period in [10, 20]:
            df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
            df[f'price_ema_{period}_ratio'] = df['Close'] / df[f'ema_{period}']
        
        # Volume features | 成交量特徵
        df['volume_sma_10'] = df['Volume'].rolling(10).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_10']
        
        # Price momentum | 價格動量
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Rate of Change | 變化率
        df['roc_5'] = ((df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)) * 100
        df['roc_10'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        return df
    
    def _create_target_variable(self, df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
        """
        Create target variable for prediction | 創建預測目標變量
        
        Args:
            df: Input DataFrame | 輸入DataFrame
            horizon: Prediction horizon in periods | 預測時間範圍（週期數）
        """
        # Future price change direction | 未來價格變化方向
        future_close = df['Close'].shift(-horizon)
        current_close = df['Close']
        
        # Binary classification target (1 = up, 0 = down) | 二分類目標（1=上漲，0=下跌）
        df['target_binary'] = (future_close > current_close).astype(int)
        
        # Regression target (future return) | 回歸目標（未來收益）
        df['target_return'] = (future_close - current_close) / current_close
        
        # Multi-class target based on return thresholds | 基於收益閾值的多分類目標
        thresholds = [-0.002, 0.002]  # -0.2% and +0.2%
        conditions = [
            df['target_return'] < thresholds[0],  # Strong down | 強烈下跌
            (df['target_return'] >= thresholds[0]) & (df['target_return'] <= thresholds[1]),  # Neutral | 中性
            df['target_return'] > thresholds[1]   # Strong up | 強烈上漲
        ]
        df['target_multiclass'] = np.select(conditions, [0, 1, 2], default=1)
        
        # Set default target | 設置默認目標
        df[self.target_column] = df['target_binary']
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """
        Remove statistical outliers | 移除統計異常值
        
        Args:
            df: Input DataFrame | 輸入DataFrame
            method: Outlier detection method (iqr, zscore) | 異常值檢測方法
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_mask = pd.Series(False, index=df.index)
        
        for col in numeric_cols:
            if col in ['target', 'target_binary', 'target_return', 'target_multiclass']:
                continue  # Skip target columns | 跳過目標列
            
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # More conservative than 1.5 * IQR | 比1.5*IQR更保守
                upper_bound = Q3 + 3 * IQR
                
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_outliers = z_scores > 4  # More conservative than 3 | 比3更保守
            
            else:
                continue
            
            outlier_mask |= col_outliers
        
        if outlier_mask.any():
            outlier_count = outlier_mask.sum()
            self.logger.info(f"Removing {outlier_count} outlier records | 移除{outlier_count}個異常記錄")
            df = df[~outlier_mask]
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset | 處理數據集中的缺失值
        """
        # Check missing values | 檢查缺失值
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        
        if len(missing_cols) > 0:
            self.logger.info(f"Handling missing values in columns | 處理列中的缺失值: {missing_cols.to_dict()}")
            
            # Forward fill for price and volume data | 價格和成交量數據使用前向填充
            price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(method='ffill')
            
            # Interpolate for technical indicators | 技術指標使用插值
            technical_cols = [col for col in df.columns if any(indicator in col.lower() 
                            for indicator in ['sma', 'ema', 'rsi', 'macd', 'bb'])]
            for col in technical_cols:
                df[col] = df[col].interpolate(method='linear')
            
            # Fill remaining with median for numeric columns | 數值列的剩餘部分用中位數填充
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())
            
            # Final check and drop rows with remaining missing values | 最終檢查並刪除剩餘缺失值行
            df = df.dropna()
        
        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize/scale features | 標準化/縮放特徵
        """
        # Identify different types of features | 識別不同類型的特徵
        price_features = [col for col in df.columns if any(word in col.lower() 
                         for word in ['price', 'open', 'high', 'low', 'close', 'sma', 'ema'])]
        
        volume_features = [col for col in df.columns if 'volume' in col.lower()]
        
        return_features = [col for col in df.columns if any(word in col.lower() 
                          for word in ['return', 'roc', 'momentum'])]
        
        indicator_features = [col for col in df.columns if any(word in col.lower() 
                             for word in ['rsi', 'macd', 'bb', 'ratio', 'position'])]
        
        # Apply different scaling strategies | 應用不同的縮放策略
        feature_groups = [
            (price_features, 'price'),
            (volume_features, 'volume'),
            (return_features, 'returns'),
            (indicator_features, 'indicators')
        ]
        
        for features, scaler_type in feature_groups:
            valid_features = [f for f in features if f in df.columns and f not in 
                            ['target', 'target_binary', 'target_return', 'target_multiclass']]
            
            if valid_features:
                scaler = self.scalers[scaler_type]
                df[valid_features] = scaler.fit_transform(df[valid_features])
        
        return df
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final validation of processed data | 處理後數據的最終驗證
        """
        # Remove infinite values | 移除無限值
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        # Ensure minimum data requirements | 確保最小數據要求
        min_records = self.config.model.feature_window * 2
        if len(df) < min_records:
            raise ValueError(f"Insufficient data after preprocessing | 預處理後數據不足: {len(df)} < {min_records}")
        
        # Store feature columns | 存儲特徵列
        target_cols = ['target', 'target_binary', 'target_return', 'target_multiclass']
        self.feature_columns = [col for col in df.columns if col not in target_cols]
        
        self.logger.info(f"Final dataset shape | 最終數據集形狀: {df.shape}")
        self.logger.info(f"Feature columns count | 特徵列數量: {len(self.feature_columns)}")
        
        return df
    
    def get_features_and_target(
        self,
        df: pd.DataFrame,
        target_type: str = 'binary'
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get features and target for model training | 獲取模型訓練的特徵和目標
        
        Args:
            df: Processed DataFrame | 處理後的DataFrame
            target_type: Type of target (binary, return, multiclass) | 目標類型
            
        Returns:
            Tuple of (features, target) | (特徵, 目標)的元組
        """
        target_map = {
            'binary': 'target_binary',
            'return': 'target_return',
            'multiclass': 'target_multiclass'
        }
        
        target_col = target_map.get(target_type, 'target_binary')
        
        # Remove rows where target is NaN | 移除目標為NaN的行
        valid_mask = df[target_col].notna()
        df_clean = df[valid_mask].copy()
        
        X = df_clean[self.feature_columns]
        y = df_clean[target_col]
        
        return X, y
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted scalers | 使用擬合的縮放器轉換新數據
        
        Args:
            df: New data to transform | 要轉換的新數據
            
        Returns:
            Transformed DataFrame | 轉換後的DataFrame
        """
        df_transformed = self.preprocess_data(
            df,
            add_features=True,
            normalize=False,  # Don't fit new scalers | 不擬合新的縮放器
            remove_outliers=False,  # Don't remove outliers from prediction data | 不從預測數據中移除異常值
            create_target=False
        )
        
        # Apply existing scalers | 應用現有的縮放器
        feature_groups = [
            ([f for f in df_transformed.columns if any(word in f.lower() 
              for word in ['price', 'open', 'high', 'low', 'close', 'sma', 'ema'])], 'price'),
            ([f for f in df_transformed.columns if 'volume' in f.lower()], 'volume'),
            ([f for f in df_transformed.columns if any(word in f.lower() 
              for word in ['return', 'roc', 'momentum'])], 'returns'),
            ([f for f in df_transformed.columns if any(word in f.lower() 
              for word in ['rsi', 'macd', 'bb', 'ratio', 'position'])], 'indicators')
        ]
        
        for features, scaler_type in feature_groups:
            valid_features = [f for f in features if f in df_transformed.columns]
            if valid_features and hasattr(self.scalers[scaler_type], 'scale_'):
                df_transformed[valid_features] = self.scalers[scaler_type].transform(df_transformed[valid_features])
        
        return df_transformed[self.feature_columns]


if __name__ == "__main__":
    # Example usage | 使用示例
    from .data_loader import DataLoader
    
    # Load sample data | 載入樣本數據
    loader = DataLoader()
    data = loader.download_data(['EURUSD'], period='6m', interval='1h')
    
    if 'EURUSD' in data:
        # Process the data | 處理數據
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess_data(data['EURUSD'])
        
        print(f"Original data shape | 原始數據形狀: {data['EURUSD'].shape}")
        print(f"Processed data shape | 處理後數據形狀: {processed_data.shape}")
        print(f"Feature columns | 特徵列: {len(preprocessor.feature_columns)}")
        
        # Get features and target | 獲取特徵和目標
        X, y = preprocessor.get_features_and_target(processed_data, target_type='binary')
        print(f"Features shape | 特徵形狀: {X.shape}")
        print(f"Target distribution | 目標分佈:\\n{y.value_counts()}")