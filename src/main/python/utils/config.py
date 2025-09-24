"""
AIFX Configuration Management | AIFX配置管理

This module handles all configuration settings for the AIFX trading system.
此模組處理AIFX交易系統的所有配置設置。

Features | 功能:
- Trading parameters | 交易參數
- Data source configurations | 數據源配置
- Model hyperparameters | 模型超參數
- Risk management settings | 風險管理設置
"""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pathlib import Path


@dataclass
class TradingConfig:
    """
    Trading configuration settings | 交易配置設置
    """
    # Trading pairs | 交易對
    symbols: List[str] = None
    
    # Timeframe | 時間框架
    timeframe: str = "1h"
    
    # Risk management | 風險管理
    risk_per_trade: float = 0.01  # 1% risk per trade | 每筆交易1%風險
    max_positions: int = 1        # Maximum concurrent positions | 最大併發倉位
    
    # Stop loss and take profit multipliers (ATR based) | 止損和止盈倍數（基於ATR）
    stop_loss_atr_multiplier: float = 2.0
    take_profit_atr_multiplier: float = 3.0
    
    # Signal confidence threshold | 信號信心閾值
    min_confidence_threshold: float = 0.6
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["EURUSD=X", "USDJPY=X"]


@dataclass  
class DataConfig:
    """
    Data configuration settings | 數據配置設置
    """
    # Data sources | 數據源
    data_source: str = "yahoo"  # yahoo, alpha_vantage, etc.
    
    # Data storage paths | 數據存儲路徑
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    
    # Historical data period | 歷史數據週期
    lookback_years: int = 3
    
    # Data validation parameters | 數據驗證參數
    min_data_points: int = 1000
    max_missing_percentage: float = 0.05  # 5% max missing data | 最多5%缺失數據


@dataclass
class ModelConfig:
    """
    Model configuration settings | 模型配置設置
    """
    # XGBoost parameters | XGBoost參數
    xgb_params: Dict = None
    
    # Random Forest parameters | 隨機森林參數  
    rf_params: Dict = None
    
    # LSTM parameters | LSTM參數
    lstm_params: Dict = None
    
    # Feature engineering | 特徵工程
    feature_window: int = 20  # Technical indicator window | 技術指標窗口
    prediction_horizon: int = 1  # Predict next N periods | 預測未來N個週期
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            
        if self.rf_params is None:
            self.rf_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
            
        if self.lstm_params is None:
            self.lstm_params = {
                'sequence_length': 60,
                'lstm_units': [50, 30],
                'dropout': 0.2,
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001
            }


@dataclass
class BacktestConfig:
    """
    Backtesting configuration settings | 回測配置設置
    """
    # Initial capital | 初始資本
    initial_cash: float = 100000.0
    
    # Commission settings | 佣金設置
    commission: float = 0.0002  # 0.02% per trade | 每筆交易0.02%
    
    # Slippage | 滑點
    slippage: float = 0.0001  # 0.01% slippage | 0.01%滑點
    
    # Backtesting period | 回測週期
    start_date: str = "2021-01-01"
    end_date: str = "2024-01-01"


class Config:
    """
    Main configuration class for AIFX system | AIFX系統主配置類
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration | 初始化配置
        
        Args:
            config_file: Path to configuration file | 配置文件路徑
        """
        # Default configurations | 默認配置
        self.trading = TradingConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.backtest = BacktestConfig()
        
        # Project paths | 項目路徑
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.src_path = self.project_root / "src"
        self.data_path = self.project_root / "data"
        self.models_path = self.project_root / "models"
        self.output_path = self.project_root / "output"
        self.logs_path = self.project_root / "logs"
        
        # Load custom configuration if provided | 如果提供則載入自定義配置
        if config_file:
            self.load_config(config_file)
    
    def load_config(self, config_file: str) -> None:
        """
        Load configuration from YAML file | 從YAML文件載入配置
        
        Args:
            config_file: Path to YAML configuration file | YAML配置文件路徑
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Update configurations | 更新配置
            if 'trading' in config_data:
                self._update_dataclass(self.trading, config_data['trading'])
            
            if 'data' in config_data:
                self._update_dataclass(self.data, config_data['data'])
                
            if 'model' in config_data:
                self._update_dataclass(self.model, config_data['model'])
                
            if 'backtest' in config_data:
                self._update_dataclass(self.backtest, config_data['backtest'])
                
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")
    
    def save_config(self, config_file: str) -> None:
        """
        Save current configuration to YAML file | 將當前配置保存到YAML文件
        
        Args:
            config_file: Output YAML file path | 輸出YAML文件路徑
        """
        config_data = {
            'trading': self._dataclass_to_dict(self.trading),
            'data': self._dataclass_to_dict(self.data),
            'model': self._dataclass_to_dict(self.model),
            'backtest': self._dataclass_to_dict(self.backtest)
        }
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def _update_dataclass(self, obj, data: Dict) -> None:
        """
        Update dataclass object with dictionary data | 用字典數據更新dataclass對象
        """
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def _dataclass_to_dict(self, obj) -> Dict:
        """
        Convert dataclass to dictionary | 將dataclass轉換為字典
        """
        result = {}
        for field_name in obj.__dataclass_fields__:
            result[field_name] = getattr(obj, field_name)
        return result
    
    def create_directories(self) -> None:
        """
        Create necessary directories | 創建必要目錄
        """
        directories = [
            self.data_path / "raw",
            self.data_path / "processed", 
            self.data_path / "external",
            self.models_path / "trained",
            self.models_path / "checkpoints",
            self.output_path,
            self.logs_path
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


class ConfigurationError(Exception):
    """
    Custom exception for configuration errors | 配置錯誤的自定義異常
    """
    pass


# Global configuration instance | 全局配置實例
config = Config()

# Environment-specific configurations | 特定環境配置
def get_config(environment: str = "development") -> Config:
    """
    Get configuration for specific environment | 獲取特定環境的配置
    
    Args:
        environment: Environment name (development, testing, production) | 環境名稱
        
    Returns:
        Configuration instance | 配置實例
    """
    config_file = f"src/main/resources/config/{environment}.yaml"
    
    if os.path.exists(config_file):
        return Config(config_file)
    else:
        return Config()


# Backward compatibility aliases | 向後兼容別名
Configuration = Config  # Main configuration class alias

if __name__ == "__main__":
    # Example usage | 使用示例
    cfg = get_config("development")
    cfg.create_directories()

    print(f"Trading symbols | 交易品種: {cfg.trading.symbols}")
    print(f"Risk per trade | 每筆交易風險: {cfg.trading.risk_per_trade}")
    print(f"Data source | 數據源: {cfg.data.data_source}")
    print(f"Initial cash | 初始資金: {cfg.backtest.initial_cash}")