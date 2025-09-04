"""
AI Model Signal Combination | AI模型信號組合

Combines predictions from multiple AI models (XGBoost, Random Forest, LSTM) into trading signals.
將多個AI模型（XGBoost、隨機森林、LSTM）的預測組合成交易信號。

This module integrates AI model outputs with confidence scoring and ensemble methods.
此模組將AI模型輸出與信心評分和集成方法整合。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

from .signal_combiner import BaseSignalCombiner, TradingSignal, SignalType
from .confidence_scorer import AdvancedConfidenceScorer, ConfidenceComponents
from models.base_model import BaseModel
from models.xgboost_model import XGBoostModel
from models.random_forest_model import RandomForestModel

try:
    from models.lstm_model import LSTMModel
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    logging.warning("LSTM model not available - TensorFlow dependency missing")

logger = logging.getLogger(__name__)


class AISignalCombiner(BaseSignalCombiner):
    """
    AI model signal combination implementation | AI模型信號組合實現
    
    Combines predictions from trained AI models into coherent trading signals.
    將訓練好的AI模型預測組合成連貫的交易信號。
    """
    
    def __init__(self, models: Optional[Dict[str, BaseModel]] = None):
        """
        Initialize AI signal combiner | 初始化AI信號組合器
        
        Args:
            models: Dictionary of trained AI models | 訓練好的AI模型字典
        """
        super().__init__("AI_Signal_Combiner", "1.0")
        
        self.models = models or {}
        self.prediction_thresholds = {
            'buy_threshold': 0.6,   # Threshold for BUY signals
            'sell_threshold': 0.4,  # Threshold for SELL signals
            'confidence_threshold': 0.5  # Minimum confidence for signal
        }
        
        # Initialize advanced confidence scorer | 初始化高級信心評分器
        self.confidence_scorer = AdvancedConfidenceScorer(
            lookback_periods=100,
            volatility_window=20,
            agreement_threshold=0.7
        )
        
        # Default weights for different models | 不同模型的默認權重
        default_weights = {
            'XGBoost': 0.4,
            'RandomForest': 0.4,
            'LSTM': 0.2
        }
        self.set_weights(default_weights)
        
        logger.info(f"Initialized AI Signal Combiner with {len(self.models)} models and advanced confidence scoring")
    
    def add_model(self, model_name: str, model: BaseModel):
        """
        Add an AI model to the combiner | 添加AI模型到組合器
        
        Args:
            model_name: Name identifier for the model | 模型名稱標識符
            model: Trained AI model instance | 訓練好的AI模型實例
        """
        if not model.is_trained:
            raise ValueError(f"Model {model_name} must be trained before adding to combiner")
            
        self.models[model_name] = model
        logger.info(f"Added model {model_name} to AI Signal Combiner")
    
    def remove_model(self, model_name: str):
        """
        Remove an AI model from the combiner | 從組合器中移除AI模型
        
        Args:
            model_name: Name of the model to remove | 要移除的模型名稱
        """
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Removed model {model_name} from AI Signal Combiner")
        else:
            logger.warning(f"Model {model_name} not found in combiner")
    
    def predict_all_models(self, features: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Get predictions from all available models | 獲取所有可用模型的預測
        
        Args:
            features: Feature DataFrame for prediction | 用於預測的特徵DataFrame
            
        Returns:
            Dictionary with model predictions and probabilities | 包含模型預測和機率的字典
        """
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Get prediction and probabilities | 獲取預測和機率
                pred = model.predict(features)
                pred_proba = model.predict_proba(features)
                
                # Convert to trading signal format | 轉換為交易信號格式
                predictions[model_name] = {
                    'prediction': pred[0] if len(pred) > 0 else 0,  # Latest prediction
                    'probabilities': pred_proba[0] if len(pred_proba) > 0 else [0.5, 0.5],
                    'confidence': np.max(pred_proba[0]) if len(pred_proba) > 0 else 0.5,
                    'timestamp': datetime.now()
                }
                
            except Exception as e:
                logger.error(f"Error getting prediction from {model_name}: {e}")
                predictions[model_name] = {
                    'prediction': 0,
                    'probabilities': [0.5, 0.5],
                    'confidence': 0.0,
                    'timestamp': datetime.now(),
                    'error': str(e)
                }
        
        return predictions
    
    def convert_ai_predictions_to_signals(self, features: pd.DataFrame) -> List[TradingSignal]:
        """
        Convert AI model predictions to trading signals | 將AI模型預測轉換為交易信號
        
        Args:
            features: Features for prediction | 用於預測的特徵
            
        Returns:
            List of trading signals from AI models | 來自AI模型的交易信號列表
        """
        if features.empty:
            logger.warning("Empty features provided for AI prediction")
            return []
            
        predictions = self.predict_all_models(features)
        signals = []
        
        for model_name, pred_data in predictions.items():
            if 'error' in pred_data:
                logger.warning(f"Skipping {model_name} due to prediction error")
                continue
                
            # Determine signal type based on prediction | 根據預測確定信號類型
            prediction = pred_data['prediction']
            confidence = pred_data['confidence']
            probabilities = pred_data['probabilities']
            
            # Convert binary/multi-class prediction to signal type | 將二元/多類預測轉換為信號類型
            if len(probabilities) >= 2:
                # Assuming binary classification: [down_probability, up_probability]
                up_prob = probabilities[1] if len(probabilities) > 1 else probabilities[0]
                down_prob = probabilities[0] if len(probabilities) > 1 else 1 - probabilities[0]
                
                if up_prob > self.prediction_thresholds['buy_threshold']:
                    signal_type = SignalType.BUY
                    strength = up_prob
                elif down_prob > (1 - self.prediction_thresholds['sell_threshold']):
                    signal_type = SignalType.SELL
                    strength = down_prob
                else:
                    signal_type = SignalType.HOLD
                    strength = max(up_prob, down_prob)
            else:
                # Single value prediction | 單值預測
                if prediction > 0.5:
                    signal_type = SignalType.BUY
                    strength = prediction
                elif prediction < -0.5:
                    signal_type = SignalType.SELL
                    strength = abs(prediction)
                else:
                    signal_type = SignalType.HOLD
                    strength = abs(prediction)
            
            # Apply confidence threshold | 應用信心閾值
            if confidence < self.prediction_thresholds['confidence_threshold']:
                signal_type = SignalType.HOLD
                strength = 0.5
            
            # Create trading signal | 創建交易信號
            signal = TradingSignal(
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                source=f"AI_{model_name}",
                timestamp=pred_data['timestamp'],
                metadata={
                    'model_name': model_name,
                    'raw_prediction': prediction,
                    'probabilities': probabilities,
                    'prediction_method': 'ai_model'
                }
            )
            
            signals.append(signal)
            
        logger.info(f"Generated {len(signals)} AI model signals from {len(predictions)} models")
        return signals
    
    def combine_signals(self, signals: List[TradingSignal]) -> TradingSignal:
        """
        Combine multiple AI signals into a single signal | 將多個AI信號組合成單一信號
        
        Args:
            signals: List of AI trading signals | AI交易信號列表
            
        Returns:
            Combined trading signal | 組合交易信號
        """
        if not self.validate_signals(signals):
            # Return neutral signal if validation fails | 如果驗證失敗則返回中性信號
            return TradingSignal(
                SignalType.HOLD, 0.5, 0.0, "AI_Combined_Error",
                metadata={'error': 'Signal validation failed'}
            )
        
        if not signals:
            return TradingSignal(SignalType.HOLD, 0.5, 0.0, "AI_Combined_Empty")
        
        # Calculate weighted combination | 計算權重組合
        weighted_signals = []
        total_weight = 0
        
        for signal in signals:
            # Extract model name from source | 從來源提取模型名稱
            model_key = signal.source.replace("AI_", "")
            weight = self.weights.get(model_key, 1.0 / len(signals))  # Default equal weight
            
            weighted_value = signal.get_weighted_signal() * weight
            weighted_signals.append(weighted_value)
            total_weight += weight * signal.confidence
        
        # Combine weighted signals | 組合權重信號
        if weighted_signals:
            combined_value = sum(weighted_signals)
            avg_confidence = total_weight / len(signals) if signals else 0.0
        else:
            combined_value = 0.0
            avg_confidence = 0.0
        
        # Determine combined signal type | 確定組合信號類型
        if combined_value > 0.1:
            signal_type = SignalType.BUY
            strength = min(abs(combined_value), 1.0)
        elif combined_value < -0.1:
            signal_type = SignalType.SELL  
            strength = min(abs(combined_value), 1.0)
        else:
            signal_type = SignalType.HOLD
            strength = 0.5
        
        # Create combined signal | 創建組合信號
        combined_signal = TradingSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=avg_confidence,
            source="AI_Combined",
            timestamp=datetime.now(),
            metadata={
                'input_signals': len(signals),
                'combined_value': combined_value,
                'model_sources': [s.source for s in signals],
                'combination_method': 'weighted_average'
            }
        )
        
        # Update performance statistics | 更新性能統計
        self.update_performance_stats(combined_signal, signals)
        
        logger.info(f"Combined {len(signals)} AI signals into {signal_type.name} "
                   f"(strength={strength:.3f}, confidence={avg_confidence:.3f})")
        
        return combined_signal
    
    def calculate_confidence(self, signals: List[TradingSignal], 
                           market_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate overall confidence for AI signal combination | 計算AI信號組合的整體信心
        
        Args:
            signals: List of AI trading signals | AI交易信號列表
            market_data: Recent market data for advanced confidence analysis | 用於高級信心分析的近期市場數據
            
        Returns:
            Combined confidence score | 組合信心評分
        """
        if not signals:
            return 0.0
        
        # Use advanced confidence scoring system | 使用高級信心評分系統
        confidence_components = self.confidence_scorer.calculate_comprehensive_confidence(
            signals=signals,
            market_data=market_data,
            historical_results=self._get_historical_results()
        )
        
        return confidence_components.overall_confidence
    
    def set_prediction_thresholds(self, buy_threshold: float, sell_threshold: float, 
                                confidence_threshold: float):
        """
        Set prediction thresholds for signal generation | 設置信號生成的預測閾值
        
        Args:
            buy_threshold: Threshold for BUY signals | BUY信號閾值
            sell_threshold: Threshold for SELL signals | SELL信號閾值  
            confidence_threshold: Minimum confidence threshold | 最小信心閾值
        """
        self.prediction_thresholds = {
            'buy_threshold': max(0.5, min(1.0, buy_threshold)),
            'sell_threshold': max(0.0, min(0.5, sell_threshold)),
            'confidence_threshold': max(0.0, min(1.0, confidence_threshold))
        }
        
        logger.info(f"Updated prediction thresholds: {self.prediction_thresholds}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get summary of loaded AI models | 獲取加載的AI模型摘要
        
        Returns:
            Summary of models and their status | 模型及其狀態的摘要
        """
        summary = {
            'total_models': len(self.models),
            'model_details': {},
            'prediction_thresholds': self.prediction_thresholds,
            'weights': self.weights
        }
        
        for name, model in self.models.items():
            summary['model_details'][name] = {
                'model_name': model.model_name,
                'version': model.version,
                'is_trained': model.is_trained,
                'training_samples': model.metadata.get('training_samples', 0),
                'performance_metrics': model.metadata.get('performance_metrics', {})
            }
        
        return summary
    
    def _get_historical_results(self) -> List[Dict[str, Any]]:
        """
        Get historical trading results for confidence analysis | 獲取用於信心分析的歷史交易結果
        
        Returns:
            List of historical trading results | 歷史交易結果列表
        """
        # Access historical results from the confidence scorer's signal history
        # 從信心評分器的信號歷史中獲取歷史結果
        return self.confidence_scorer.signal_history
    
    def add_trading_result(self, signal_info: Dict[str, Any], result: Dict[str, Any]):
        """
        Add trading result for performance tracking | 添加交易結果用於績效追蹤
        
        Args:
            signal_info: Information about the executed signal | 執行信號的信息
            result: Trading result data | 交易結果數據
        """
        self.confidence_scorer.add_signal_result(signal_info, result)
        logger.info(f"Added trading result for signal performance tracking: {result.get('profit_loss', 'N/A')}")
    
    def get_advanced_confidence_report(self, signals: List[TradingSignal], 
                                     market_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Get detailed confidence analysis report | 獲取詳細信心分析報告
        
        Args:
            signals: List of trading signals to analyze | 要分析的交易信號列表
            market_data: Recent market data | 近期市場數據
            
        Returns:
            Comprehensive confidence analysis report | 綜合信心分析報告
        """
        confidence_components = self.confidence_scorer.calculate_comprehensive_confidence(
            signals=signals,
            market_data=market_data,
            historical_results=self._get_historical_results()
        )
        
        return self.confidence_scorer.get_confidence_report(confidence_components)
    
    def update_confidence_weights(self, weights: Dict[str, float]):
        """
        Update weights for confidence components | 更新信心組件權重
        
        Args:
            weights: Dictionary mapping component names to weights | 組件名稱到權重的映射字典
        """
        from .confidence_scorer import ConfidenceFactorType
        
        # Map string keys to enum values | 將字符串鍵映射到枚舉值
        enum_weights = {}
        for key, value in weights.items():
            if hasattr(ConfidenceFactorType, key.upper()):
                enum_key = getattr(ConfidenceFactorType, key.upper())
                enum_weights[enum_key] = value
        
        if enum_weights:
            self.confidence_scorer.set_component_weights(enum_weights)
            logger.info(f"Updated confidence component weights: {weights}")
        else:
            logger.warning(f"No valid confidence component weights found in: {weights}")
    
    def get_confidence_scorer_info(self) -> Dict[str, Any]:
        """
        Get information about the confidence scorer | 獲取信心評分器信息
        
        Returns:
            Confidence scorer configuration and statistics | 信心評分器配置和統計信息
        """
        return {
            'lookback_periods': self.confidence_scorer.lookback_periods,
            'volatility_window': self.confidence_scorer.volatility_window,
            'agreement_threshold': self.confidence_scorer.agreement_threshold,
            'component_weights': {k.value: v for k, v in self.confidence_scorer.component_weights.items()},
            'signal_history_count': len(self.confidence_scorer.signal_history),
            'performance_cache_sources': list(self.confidence_scorer.performance_cache.keys())
        }