"""
Technical Indicator Signal Fusion | 技術指標信號融合

Converts technical indicator values into trading signals and combines them intelligently.
將技術指標值轉換為交易信號並智能組合。

This module handles traditional technical analysis signals from:
此模組處理來自以下的傳統技術分析信號：
- Moving Averages (SMA, EMA) | 移動平均線
- MACD (Moving Average Convergence Divergence) | MACD
- RSI (Relative Strength Index) | 相對強弱指標
- Bollinger Bands | 布林帶
- ATR (Average True Range) | 平均真實區間
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

from .signal_combiner import BaseSignalCombiner, TradingSignal, SignalType, SignalStrength
from utils.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class TechnicalSignalCombiner(BaseSignalCombiner):
    """
    Technical indicator signal combination implementation | 技術指標信號組合實現
    
    Converts technical indicators into trading signals and combines them using various strategies.
    將技術指標轉換為交易信號並使用各種策略組合它們。
    """
    
    def __init__(self):
        """
        Initialize technical signal combiner | 初始化技術信號組合器
        """
        super().__init__("Technical_Signal_Combiner", "1.0")
        
        self.technical_indicators = TechnicalIndicators()
        
        # Signal generation parameters | 信號生成參數
        self.signal_params = {
            'rsi': {
                'overbought': 70,
                'oversold': 30,
                'strength_multiplier': 1.0
            },
            'macd': {
                'signal_threshold': 0.0,
                'strength_multiplier': 1.0
            },
            'bollinger_bands': {
                'upper_threshold': 0.8,  # How close to upper band for sell
                'lower_threshold': 0.2,  # How close to lower band for buy
                'strength_multiplier': 1.0
            },
            'moving_averages': {
                'price_ma_threshold': 0.002,  # 0.2% threshold for MA crossover
                'strength_multiplier': 1.0
            }
        }
        
        # Default weights for different technical indicators | 不同技術指標的默認權重
        default_weights = {
            'RSI': 0.25,
            'MACD': 0.25,
            'BollingerBands': 0.25,
            'MovingAverages': 0.25
        }
        self.set_weights(default_weights)
        
        logger.info("Initialized Technical Signal Combiner")
    
    def generate_rsi_signal(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate trading signal from RSI indicator | 從RSI指標生成交易信號
        
        Args:
            data: OHLCV DataFrame with RSI calculated | 計算了RSI的OHLCV DataFrame
            
        Returns:
            RSI-based trading signal | 基於RSI的交易信號
        """
        if 'rsi' not in data.columns or data.empty:
            return None
            
        latest_rsi = data['rsi'].iloc[-1]
        
        if pd.isna(latest_rsi):
            return None
        
        # Determine signal type and strength | 確定信號類型和強度
        if latest_rsi >= self.signal_params['rsi']['overbought']:
            # Overbought - SELL signal | 超買 - 賣出信號
            strength = min(1.0, (latest_rsi - 70) / 30 * self.signal_params['rsi']['strength_multiplier'])
            signal_type = SignalType.SELL
            confidence = min(0.9, 0.5 + strength * 0.4)
        elif latest_rsi <= self.signal_params['rsi']['oversold']:
            # Oversold - BUY signal | 超賣 - 買入信號
            strength = min(1.0, (30 - latest_rsi) / 30 * self.signal_params['rsi']['strength_multiplier'])
            signal_type = SignalType.BUY
            confidence = min(0.9, 0.5 + strength * 0.4)
        else:
            # Neutral zone | 中性區域
            signal_type = SignalType.HOLD
            strength = 0.5
            confidence = 0.3
        
        return TradingSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            source="Technical_RSI",
            timestamp=datetime.now(),
            metadata={
                'rsi_value': latest_rsi,
                'overbought_level': self.signal_params['rsi']['overbought'],
                'oversold_level': self.signal_params['rsi']['oversold']
            }
        )
    
    def generate_macd_signal(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate trading signal from MACD indicator | 從MACD指標生成交易信號
        
        Args:
            data: OHLCV DataFrame with MACD calculated | 計算了MACD的OHLCV DataFrame
            
        Returns:
            MACD-based trading signal | 基於MACD的交易信號
        """
        required_cols = ['macd', 'macd_signal', 'macd_histogram']
        if not all(col in data.columns for col in required_cols) or data.empty:
            return None
        
        if len(data) < 2:
            return None
            
        # Get latest and previous MACD values | 獲取最新和之前的MACD值
        latest_macd = data['macd'].iloc[-1]
        latest_signal = data['macd_signal'].iloc[-1]
        latest_histogram = data['macd_histogram'].iloc[-1]
        prev_histogram = data['macd_histogram'].iloc[-2]
        
        if pd.isna(latest_macd) or pd.isna(latest_signal) or pd.isna(latest_histogram):
            return None
        
        # Detect signal crossover | 檢測信號交叉
        signal_crossover = (latest_histogram > 0) != (prev_histogram > 0) if not pd.isna(prev_histogram) else False
        
        # Determine signal type | 確定信號類型
        if latest_macd > latest_signal:
            signal_type = SignalType.BUY
            strength = min(1.0, abs(latest_histogram) * self.signal_params['macd']['strength_multiplier'])
        elif latest_macd < latest_signal:
            signal_type = SignalType.SELL
            strength = min(1.0, abs(latest_histogram) * self.signal_params['macd']['strength_multiplier'])
        else:
            signal_type = SignalType.HOLD
            strength = 0.5
        
        # Higher confidence on crossovers | 交叉時信心更高
        base_confidence = 0.4
        crossover_bonus = 0.3 if signal_crossover else 0.0
        strength_bonus = min(0.3, strength * 0.2)
        confidence = min(0.9, base_confidence + crossover_bonus + strength_bonus)
        
        return TradingSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            source="Technical_MACD",
            timestamp=datetime.now(),
            metadata={
                'macd_value': latest_macd,
                'macd_signal': latest_signal,
                'macd_histogram': latest_histogram,
                'crossover_detected': signal_crossover
            }
        )
    
    def generate_bollinger_bands_signal(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate trading signal from Bollinger Bands | 從布林帶生成交易信號
        
        Args:
            data: OHLCV DataFrame with Bollinger Bands calculated | 計算了布林帶的OHLCV DataFrame
            
        Returns:
            Bollinger Bands-based trading signal | 基於布林帶的交易信號
        """
        required_cols = ['Close', 'bb_upper', 'bb_lower', 'bb_middle']
        if not all(col in data.columns for col in required_cols) or data.empty:
            return None
            
        latest_close = data['Close'].iloc[-1]
        latest_upper = data['bb_upper'].iloc[-1]
        latest_lower = data['bb_lower'].iloc[-1]
        latest_middle = data['bb_middle'].iloc[-1]
        
        if any(pd.isna(val) for val in [latest_close, latest_upper, latest_lower, latest_middle]):
            return None
        
        # Calculate position within bands | 計算在布林帶中的位置
        band_range = latest_upper - latest_lower
        if band_range <= 0:
            return None
            
        position_in_bands = (latest_close - latest_lower) / band_range  # 0 = lower band, 1 = upper band
        
        # Generate signals based on band position | 根據布林帶位置生成信號
        if position_in_bands >= self.signal_params['bollinger_bands']['upper_threshold']:
            # Near upper band - SELL signal | 接近上軌 - 賣出信號
            signal_type = SignalType.SELL
            strength = min(1.0, (position_in_bands - 0.8) / 0.2 * self.signal_params['bollinger_bands']['strength_multiplier'])
            confidence = min(0.8, 0.4 + strength * 0.4)
        elif position_in_bands <= self.signal_params['bollinger_bands']['lower_threshold']:
            # Near lower band - BUY signal | 接近下軌 - 買入信號
            signal_type = SignalType.BUY
            strength = min(1.0, (0.2 - position_in_bands) / 0.2 * self.signal_params['bollinger_bands']['strength_multiplier'])
            confidence = min(0.8, 0.4 + strength * 0.4)
        else:
            # Middle range - HOLD | 中間範圍 - 持有
            signal_type = SignalType.HOLD
            strength = 0.5
            confidence = 0.3
        
        return TradingSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            source="Technical_BollingerBands",
            timestamp=datetime.now(),
            metadata={
                'close_price': latest_close,
                'bb_upper': latest_upper,
                'bb_lower': latest_lower,
                'bb_middle': latest_middle,
                'position_in_bands': position_in_bands,
                'band_width': band_range
            }
        )
    
    def generate_moving_average_signal(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Generate trading signal from Moving Averages | 從移動平均線生成交易信號
        
        Args:
            data: OHLCV DataFrame with moving averages calculated | 計算了移動平均線的OHLCV DataFrame
            
        Returns:
            Moving average-based trading signal | 基於移動平均線的交易信號
        """
        required_cols = ['Close', 'sma_20', 'sma_50']
        if not all(col in data.columns for col in required_cols) or data.empty:
            return None
            
        latest_close = data['Close'].iloc[-1]
        latest_sma_20 = data['sma_20'].iloc[-1]
        latest_sma_50 = data['sma_50'].iloc[-1]
        
        if any(pd.isna(val) for val in [latest_close, latest_sma_20, latest_sma_50]):
            return None
        
        # Calculate price relative to moving averages | 計算價格相對於移動平均線
        price_vs_sma20 = (latest_close - latest_sma_20) / latest_sma_20
        price_vs_sma50 = (latest_close - latest_sma_50) / latest_sma_50
        sma20_vs_sma50 = (latest_sma_20 - latest_sma_50) / latest_sma_50
        
        # Determine trend direction | 確定趋勢方向
        threshold = self.signal_params['moving_averages']['price_ma_threshold']
        
        # Price above both MAs and short MA above long MA = BUY
        if (price_vs_sma20 > threshold and price_vs_sma50 > threshold and 
            sma20_vs_sma50 > threshold / 2):
            signal_type = SignalType.BUY
            strength = min(1.0, (price_vs_sma20 + price_vs_sma50) / 2 / threshold * 
                          self.signal_params['moving_averages']['strength_multiplier'])
            confidence = min(0.8, 0.5 + abs(sma20_vs_sma50) * 20)
        # Price below both MAs and short MA below long MA = SELL  
        elif (price_vs_sma20 < -threshold and price_vs_sma50 < -threshold and 
              sma20_vs_sma50 < -threshold / 2):
            signal_type = SignalType.SELL
            strength = min(1.0, abs(price_vs_sma20 + price_vs_sma50) / 2 / threshold * 
                          self.signal_params['moving_averages']['strength_multiplier'])
            confidence = min(0.8, 0.5 + abs(sma20_vs_sma50) * 20)
        else:
            signal_type = SignalType.HOLD
            strength = 0.5
            confidence = 0.3
        
        return TradingSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            source="Technical_MovingAverages",
            timestamp=datetime.now(),
            metadata={
                'close_price': latest_close,
                'sma_20': latest_sma_20,
                'sma_50': latest_sma_50,
                'price_vs_sma20_pct': price_vs_sma20 * 100,
                'price_vs_sma50_pct': price_vs_sma50 * 100,
                'sma20_vs_sma50_pct': sma20_vs_sma50 * 100
            }
        )
    
    def generate_all_technical_signals(self, data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate all technical indicator signals | 生成所有技術指標信號
        
        Args:
            data: OHLCV DataFrame with all indicators calculated | 計算了所有指標的OHLCV DataFrame
            
        Returns:
            List of technical indicator signals | 技術指標信號列表
        """
        signals = []
        
        # Generate individual signals | 生成個別信號
        signal_generators = [
            ('RSI', self.generate_rsi_signal),
            ('MACD', self.generate_macd_signal),
            ('BollingerBands', self.generate_bollinger_bands_signal),
            ('MovingAverages', self.generate_moving_average_signal)
        ]
        
        for name, generator in signal_generators:
            try:
                signal = generator(data)
                if signal:
                    signals.append(signal)
                    logger.debug(f"Generated {name} signal: {signal}")
                else:
                    logger.debug(f"No {name} signal generated")
            except Exception as e:
                logger.error(f"Error generating {name} signal: {e}")
        
        logger.info(f"Generated {len(signals)} technical indicator signals")
        return signals
    
    def combine_signals(self, signals: List[TradingSignal]) -> TradingSignal:
        """
        Combine multiple technical signals into a single signal | 將多個技術信號組合成單一信號
        
        Args:
            signals: List of technical trading signals | 技術交易信號列表
            
        Returns:
            Combined trading signal | 組合交易信號
        """
        if not self.validate_signals(signals):
            return TradingSignal(
                SignalType.HOLD, 0.5, 0.0, "Technical_Combined_Error",
                metadata={'error': 'Signal validation failed'}
            )
        
        if not signals:
            return TradingSignal(SignalType.HOLD, 0.5, 0.0, "Technical_Combined_Empty")
        
        # Calculate weighted combination | 計算權重組合
        weighted_values = []
        total_weight = 0
        signal_contributions = {}
        
        for signal in signals:
            # Extract indicator type from source | 從來源提取指標類型
            indicator_type = signal.source.replace("Technical_", "")
            weight = self.weights.get(indicator_type, 1.0 / len(signals))
            
            weighted_value = signal.get_weighted_signal() * weight
            weighted_values.append(weighted_value)
            total_weight += weight * signal.confidence
            
            signal_contributions[indicator_type] = {
                'signal_type': signal.signal_type.name,
                'strength': signal.strength,
                'confidence': signal.confidence,
                'weighted_contribution': weighted_value
            }
        
        # Combine weighted values | 組合權重值
        if weighted_values:
            combined_value = sum(weighted_values)
            avg_confidence = total_weight / len(signals) if signals else 0.0
        else:
            combined_value = 0.0
            avg_confidence = 0.0
        
        # Determine combined signal type | 確定組合信號類型
        if combined_value > 0.1:
            signal_type = SignalType.BUY
            strength = min(1.0, abs(combined_value))
        elif combined_value < -0.1:
            signal_type = SignalType.SELL
            strength = min(1.0, abs(combined_value))
        else:
            signal_type = SignalType.HOLD
            strength = 0.5
        
        # Create combined signal | 創建組合信號
        combined_signal = TradingSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=avg_confidence,
            source="Technical_Combined",
            timestamp=datetime.now(),
            metadata={
                'input_signals': len(signals),
                'combined_value': combined_value,
                'signal_contributions': signal_contributions,
                'combination_method': 'weighted_technical'
            }
        )
        
        # Update performance statistics | 更新性能統計
        self.update_performance_stats(combined_signal, signals)
        
        logger.info(f"Combined {len(signals)} technical signals into {signal_type.name} "
                   f"(strength={strength:.3f}, confidence={avg_confidence:.3f})")
        
        return combined_signal
    
    def calculate_confidence(self, signals: List[TradingSignal]) -> float:
        """
        Calculate overall confidence for technical signal combination | 計算技術信號組合的整體信心
        
        Args:
            signals: List of technical trading signals | 技術交易信號列表
            
        Returns:
            Combined confidence score | 組合信心評分
        """
        if not signals:
            return 0.0
        
        # Technical indicators confidence based on signal strength alignment | 技術指標信心基於信號強度對齊
        signal_values = [s.get_weighted_signal() for s in signals]
        confidences = [s.confidence for s in signals]
        
        # Agreement factor: higher when technical signals agree | 一致性因子：技術信號一致時更高
        if len(signal_values) > 1:
            signal_directions = [1 if v > 0 else -1 if v < 0 else 0 for v in signal_values]
            agreement_count = len([d for d in signal_directions if d == signal_directions[0]]) if signal_directions else 0
            agreement_factor = agreement_count / len(signal_directions)
        else:
            agreement_factor = 1.0
        
        # Average individual confidences | 個別信心的平均值
        avg_confidence = np.mean(confidences)
        
        # Combined confidence with agreement weighting | 結合一致性權重的組合信心
        combined_confidence = (avg_confidence * 0.6) + (agreement_factor * 0.4)
        
        return min(1.0, max(0.0, combined_confidence))
    
    def update_signal_parameters(self, parameter_updates: Dict[str, Dict[str, Any]]):
        """
        Update signal generation parameters | 更新信號生成參數
        
        Args:
            parameter_updates: Dictionary of parameter updates | 參數更新字典
        """
        for indicator, params in parameter_updates.items():
            if indicator in self.signal_params:
                self.signal_params[indicator].update(params)
                logger.info(f"Updated {indicator} parameters: {params}")
            else:
                logger.warning(f"Unknown indicator {indicator} in parameter update")
    
    def get_signal_parameters(self) -> Dict[str, Any]:
        """
        Get current signal generation parameters | 獲取當前信號生成參數
        
        Returns:
            Current signal parameters | 當前信號參數
        """
        return {
            'signal_params': self.signal_params.copy(),
            'weights': self.weights.copy(),
            'combiner_info': self.get_combiner_info()
        }