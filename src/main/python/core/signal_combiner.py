"""
Signal Combination Framework | 信號組合框架

Abstract base classes and interfaces for combining multiple trading signals in AIFX system.
AIFX系統中用於組合多個交易信號的抽象基類和接口。

This module provides the foundational framework for integrating:
此模組為以下整合提供基礎框架：
- AI model predictions (XGBoost, Random Forest, LSTM) | AI模型預測
- Technical indicator signals (MA, MACD, RSI, Bollinger Bands) | 技術指標信號
- Confidence scoring and weighting mechanisms | 信心評分和權重機制
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime
import logging

# Configure logger | 配置日誌器
logger = logging.getLogger(__name__)


class SignalType(Enum):
    """
    Enumeration of signal types | 信號類型枚舉
    """
    BUY = 1
    SELL = -1
    HOLD = 0


class SignalStrength(Enum):
    """
    Enumeration of signal strength levels | 信號強度等級枚舉
    """
    VERY_WEAK = 0.2
    WEAK = 0.4
    MODERATE = 0.6
    STRONG = 0.8
    VERY_STRONG = 1.0


class TradingSignal:
    """
    Individual trading signal container | 個別交易信號容器
    
    Represents a single trading signal with metadata.
    代表帶有元數據的單一交易信號。
    """
    
    def __init__(self, 
                 signal_type: SignalType,
                 strength: float,
                 confidence: float,
                 source: str,
                 timestamp: Optional[datetime] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize trading signal | 初始化交易信號
        
        Args:
            signal_type: Type of signal (BUY/SELL/HOLD) | 信號類型
            strength: Signal strength (0.0-1.0) | 信號強度
            confidence: Confidence level (0.0-1.0) | 信心水準
            source: Signal source identifier | 信號來源識別符
            timestamp: Signal timestamp | 信號時間戳
            metadata: Additional metadata | 額外元數據
        """
        self.signal_type = signal_type
        self.strength = max(0.0, min(1.0, strength))  # Clamp to [0, 1]
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        self.source = source
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        
    def get_weighted_signal(self) -> float:
        """
        Get signal value weighted by confidence | 獲取信心權重的信號值
        
        Returns:
            Weighted signal value | 權重信號值
        """
        base_value = self.signal_type.value * self.strength
        return base_value * self.confidence
    
    def __str__(self) -> str:
        """String representation | 字符串表示"""
        return (f"TradingSignal({self.signal_type.name}, "
                f"strength={self.strength:.2f}, "
                f"confidence={self.confidence:.2f}, "
                f"source={self.source})")


class BaseSignalCombiner(ABC):
    """
    Abstract base class for signal combination strategies | 信號組合策略的抽象基類
    
    Provides common interface for combining multiple trading signals.
    為組合多個交易信號提供通用接口。
    """
    
    def __init__(self, combiner_name: str, version: str = "1.0"):
        """
        Initialize signal combiner | 初始化信號組合器
        
        Args:
            combiner_name: Name of the combiner | 組合器名稱
            version: Combiner version | 組合器版本
        """
        self.combiner_name = combiner_name
        self.version = version
        self.weights = {}  # Signal source weights
        self.combination_history = []
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'combiner_name': combiner_name,
            'version': version,
            'signals_processed': 0,
            'performance_stats': {}
        }
        
    @abstractmethod
    def combine_signals(self, signals: List[TradingSignal]) -> TradingSignal:
        """
        Combine multiple signals into a single signal | 將多個信號組合成單一信號
        
        Args:
            signals: List of trading signals | 交易信號列表
            
        Returns:
            Combined trading signal | 組合交易信號
        """
        pass
    
    @abstractmethod
    def calculate_confidence(self, signals: List[TradingSignal]) -> float:
        """
        Calculate overall confidence for combined signal | 計算組合信號的整體信心
        
        Args:
            signals: List of trading signals | 交易信號列表
            
        Returns:
            Combined confidence score | 組合信心評分
        """
        pass
    
    def set_weights(self, weights: Dict[str, float]):
        """
        Set weights for different signal sources | 設置不同信號來源的權重
        
        Args:
            weights: Dictionary mapping source names to weights | 將來源名稱映射到權重的字典
        """
        # Normalize weights to sum to 1 | 權重標準化使總和為1
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.weights = {source: weight / total_weight 
                          for source, weight in weights.items()}
        else:
            self.weights = weights
            
        logger.info(f"Updated weights for {self.combiner_name}: {self.weights}")
    
    def get_current_weights(self) -> Dict[str, float]:
        """
        Get current signal source weights | 獲取當前信號來源權重
        
        Returns:
            Current weights dictionary | 當前權重字典
        """
        return self.weights.copy()
    
    def add_weight_optimization_result(self, signal_source: str, trade_result: Dict[str, Any]):
        """
        Add trading result for weight optimization (to be implemented by subclasses) | 
        添加用於權重優化的交易結果（由子類實現）
        
        Args:
            signal_source: Source of the signal that generated this result | 生成此結果的信號來源
            trade_result: Trading result data | 交易結果數據
        """
        # Base implementation - subclasses should override this
        # 基本實現 - 子類應該覆蓋此方法
        logger.debug(f"Base class received trading result from {signal_source}: {trade_result}")
    
    def optimize_weights_if_needed(self, signal_sources: List[str], 
                                 market_data: Optional[pd.DataFrame] = None) -> bool:
        """
        Optimize weights if needed (to be implemented by subclasses) | 
        如果需要則優化權重（由子類實現）
        
        Args:
            signal_sources: List of available signal sources | 可用信號來源列表
            market_data: Current market data for optimization | 用於優化的當前市場數據
            
        Returns:
            True if weights were updated, False otherwise | 如果權重已更新則為True，否則為False
        """
        # Base implementation - subclasses should override this
        # 基本實現 - 子類應該覆蓋此方法
        return False
    
    def get_signal_statistics(self, signals: List[TradingSignal]) -> Dict[str, Any]:
        """
        Get statistical information about input signals | 獲取輸入信號的統計信息
        
        Args:
            signals: List of trading signals | 交易信號列表
            
        Returns:
            Statistics dictionary | 統計字典
        """
        if not signals:
            return {}
            
        signal_values = [s.get_weighted_signal() for s in signals]
        confidences = [s.confidence for s in signals]
        sources = [s.source for s in signals]
        
        stats = {
            'num_signals': len(signals),
            'signal_mean': np.mean(signal_values),
            'signal_std': np.std(signal_values),
            'signal_range': np.max(signal_values) - np.min(signal_values),
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'sources': list(set(sources)),
            'buy_signals': sum(1 for s in signals if s.signal_type == SignalType.BUY),
            'sell_signals': sum(1 for s in signals if s.signal_type == SignalType.SELL),
            'hold_signals': sum(1 for s in signals if s.signal_type == SignalType.HOLD)
        }
        
        return stats
    
    def validate_signals(self, signals: List[TradingSignal]) -> bool:
        """
        Validate input signals | 驗證輸入信號
        
        Args:
            signals: List of trading signals | 交易信號列表
            
        Returns:
            True if signals are valid | 如果信號有效則為True
        """
        if not signals:
            logger.warning("No signals provided for combination")
            return False
            
        for i, signal in enumerate(signals):
            if not isinstance(signal, TradingSignal):
                logger.error(f"Signal at index {i} is not a TradingSignal instance")
                return False
                
            if not (0.0 <= signal.strength <= 1.0):
                logger.error(f"Signal {i} has invalid strength: {signal.strength}")
                return False
                
            if not (0.0 <= signal.confidence <= 1.0):
                logger.error(f"Signal {i} has invalid confidence: {signal.confidence}")
                return False
        
        return True
    
    def update_performance_stats(self, combined_signal: TradingSignal, 
                               input_signals: List[TradingSignal]):
        """
        Update performance statistics | 更新性能統計
        
        Args:
            combined_signal: The combined signal output | 組合信號輸出
            input_signals: The input signals used | 使用的輸入信號
        """
        self.metadata['signals_processed'] += len(input_signals)
        
        # Store combination history (keep last 1000) | 存儲組合歷史（保留最後1000個）
        combination_record = {
            'timestamp': datetime.now().isoformat(),
            'input_count': len(input_signals),
            'combined_signal_type': combined_signal.signal_type.name,
            'combined_strength': combined_signal.strength,
            'combined_confidence': combined_signal.confidence,
            'input_stats': self.get_signal_statistics(input_signals)
        }
        
        self.combination_history.append(combination_record)
        if len(self.combination_history) > 1000:
            self.combination_history.pop(0)
    
    def get_combiner_info(self) -> Dict[str, Any]:
        """
        Get comprehensive combiner information | 獲取全面的組合器信息
        
        Returns:
            Combiner information dictionary | 組合器信息字典
        """
        return {
            'combiner_name': self.combiner_name,
            'version': self.version,
            'weights': self.weights,
            'metadata': self.metadata,
            'recent_combinations': len(self.combination_history),
            'performance_summary': self._get_performance_summary()
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary from combination history | 從組合歷史獲取性能摘要
        
        Returns:
            Performance summary | 性能摘要
        """
        if not self.combination_history:
            return {}
            
        recent_combinations = self.combination_history[-100:]  # Last 100 combinations
        
        strengths = [combo['combined_strength'] for combo in recent_combinations]
        confidences = [combo['combined_confidence'] for combo in recent_combinations]
        
        signal_types = [combo['combined_signal_type'] for combo in recent_combinations]
        buy_count = signal_types.count('BUY')
        sell_count = signal_types.count('SELL')
        hold_count = signal_types.count('HOLD')
        
        return {
            'avg_strength': np.mean(strengths) if strengths else 0.0,
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'signal_distribution': {
                'buy_percentage': (buy_count / len(recent_combinations)) * 100,
                'sell_percentage': (sell_count / len(recent_combinations)) * 100,
                'hold_percentage': (hold_count / len(recent_combinations)) * 100
            },
            'total_combinations': len(self.combination_history)
        }


class SignalAggregator:
    """
    Signal aggregation utility | 信號聚合工具
    
    Utility class for collecting and managing signals from multiple sources.
    用於收集和管理來自多個來源信號的工具類。
    """
    
    def __init__(self):
        """Initialize signal aggregator | 初始化信號聚合器"""
        self.signals = []
        self.source_counts = {}
        
    def add_signal(self, signal: TradingSignal):
        """
        Add a signal to the aggregator | 添加信號到聚合器
        
        Args:
            signal: Trading signal to add | 要添加的交易信號
        """
        if not isinstance(signal, TradingSignal):
            raise ValueError("Signal must be a TradingSignal instance")
            
        self.signals.append(signal)
        self.source_counts[signal.source] = self.source_counts.get(signal.source, 0) + 1
    
    def get_signals_by_source(self, source: str) -> List[TradingSignal]:
        """
        Get all signals from a specific source | 獲取來自特定來源的所有信號
        
        Args:
            source: Source identifier | 來源識別符
            
        Returns:
            List of signals from the source | 來自該來源的信號列表
        """
        return [signal for signal in self.signals if signal.source == source]
    
    def get_signals_by_type(self, signal_type: SignalType) -> List[TradingSignal]:
        """
        Get all signals of a specific type | 獲取特定類型的所有信號
        
        Args:
            signal_type: Signal type to filter by | 要過濾的信號類型
            
        Returns:
            List of signals of the specified type | 指定類型的信號列表
        """
        return [signal for signal in self.signals if signal.signal_type == signal_type]
    
    def clear(self):
        """Clear all signals | 清除所有信號"""
        self.signals.clear()
        self.source_counts.clear()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of aggregated signals | 獲取聚合信號摘要
        
        Returns:
            Summary dictionary | 摘要字典
        """
        if not self.signals:
            return {'total_signals': 0}
            
        return {
            'total_signals': len(self.signals),
            'source_counts': self.source_counts.copy(),
            'signal_type_counts': {
                'buy': len(self.get_signals_by_type(SignalType.BUY)),
                'sell': len(self.get_signals_by_type(SignalType.SELL)),
                'hold': len(self.get_signals_by_type(SignalType.HOLD))
            },
            'avg_strength': np.mean([s.strength for s in self.signals]),
            'avg_confidence': np.mean([s.confidence for s in self.signals])
        }


# Backward compatibility aliases | 向後兼容別名
SignalCombiner = BaseSignalCombiner  # Main signal combiner class alias