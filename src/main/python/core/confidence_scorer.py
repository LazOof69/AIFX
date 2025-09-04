"""
Advanced Confidence Scoring System | 高級信心評分系統

Comprehensive confidence evaluation for trading signals based on multiple factors including
model agreement, historical performance, market conditions, and signal quality metrics.
基於多個因子的交易信號綜合信心評估，包括模型一致性、歷史績效、市場條件和信號品質指標。

This module provides sophisticated confidence scoring mechanisms for the AIFX trading system.
此模組為AIFX交易系統提供精細的信心評分機制。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from .signal_combiner import TradingSignal, SignalType

logger = logging.getLogger(__name__)


class ConfidenceFactorType(Enum):
    """
    Types of confidence factors | 信心因子類型
    """
    MODEL_AGREEMENT = "model_agreement"
    HISTORICAL_PERFORMANCE = "historical_performance"
    SIGNAL_STRENGTH = "signal_strength"
    MARKET_VOLATILITY = "market_volatility"
    SIGNAL_FRESHNESS = "signal_freshness"
    CROSS_VALIDATION = "cross_validation"


@dataclass
class ConfidenceComponents:
    """
    Components of confidence score | 信心評分組件
    """
    model_agreement: float = 0.0
    historical_performance: float = 0.0
    signal_strength: float = 0.0
    market_volatility_adjustment: float = 0.0
    signal_freshness: float = 0.0
    cross_validation_score: float = 0.0
    overall_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary | 轉換為字典"""
        return {
            'model_agreement': self.model_agreement,
            'historical_performance': self.historical_performance,
            'signal_strength': self.signal_strength,
            'market_volatility_adjustment': self.market_volatility_adjustment,
            'signal_freshness': self.signal_freshness,
            'cross_validation_score': self.cross_validation_score,
            'overall_confidence': self.overall_confidence
        }


class AdvancedConfidenceScorer:
    """
    Advanced confidence scoring system | 高級信心評分系統
    
    Provides comprehensive confidence evaluation using multiple factors and
    historical performance tracking for trading signal reliability assessment.
    使用多個因子和歷史績效追蹤為交易信號可靠性評估提供綜合信心評估。
    """
    
    def __init__(self, 
                 lookback_periods: int = 100,
                 volatility_window: int = 20,
                 agreement_threshold: float = 0.7):
        """
        Initialize advanced confidence scorer | 初始化高級信心評分器
        
        Args:
            lookback_periods: Number of periods for historical analysis | 歷史分析週期數
            volatility_window: Window for volatility calculation | 波動性計算窗口
            agreement_threshold: Threshold for model agreement | 模型一致性閾值
        """
        self.lookback_periods = lookback_periods
        self.volatility_window = volatility_window
        self.agreement_threshold = agreement_threshold
        
        # Component weights for final confidence calculation | 最終信心計算的組件權重
        self.component_weights = {
            ConfidenceFactorType.MODEL_AGREEMENT: 0.25,
            ConfidenceFactorType.HISTORICAL_PERFORMANCE: 0.20,
            ConfidenceFactorType.SIGNAL_STRENGTH: 0.20,
            ConfidenceFactorType.MARKET_VOLATILITY: 0.15,
            ConfidenceFactorType.SIGNAL_FRESHNESS: 0.10,
            ConfidenceFactorType.CROSS_VALIDATION: 0.10
        }
        
        # Historical performance tracking | 歷史績效追蹤
        self.signal_history: List[Dict[str, Any]] = []
        self.performance_cache: Dict[str, Dict[str, float]] = {}
        
        # Market condition cache | 市場條件緩存
        self.volatility_cache: Dict[str, float] = {}
        self.market_regime_cache: Dict[str, str] = {}
        
        logger.info(f"Initialized AdvancedConfidenceScorer with {lookback_periods} lookback periods")
    
    def calculate_comprehensive_confidence(self, 
                                         signals: List[TradingSignal],
                                         market_data: Optional[pd.DataFrame] = None,
                                         historical_results: Optional[List[Dict]] = None) -> ConfidenceComponents:
        """
        Calculate comprehensive confidence score | 計算綜合信心評分
        
        Args:
            signals: List of trading signals to evaluate | 要評估的交易信號列表
            market_data: Recent market data for volatility analysis | 用於波動性分析的近期市場數據
            historical_results: Historical trading results for performance analysis | 用於績效分析的歷史交易結果
            
        Returns:
            Detailed confidence components | 詳細信心組件
        """
        if not signals:
            logger.warning("No signals provided for confidence calculation")
            return ConfidenceComponents()
        
        components = ConfidenceComponents()
        
        # 1. Model Agreement Score | 模型一致性評分
        components.model_agreement = self._calculate_model_agreement(signals)
        
        # 2. Historical Performance Score | 歷史績效評分
        components.historical_performance = self._calculate_historical_performance(
            signals, historical_results
        )
        
        # 3. Signal Strength Score | 信號強度評分
        components.signal_strength = self._calculate_signal_strength_score(signals)
        
        # 4. Market Volatility Adjustment | 市場波動性調整
        components.market_volatility_adjustment = self._calculate_volatility_adjustment(
            signals, market_data
        )
        
        # 5. Signal Freshness Score | 信號新鮮度評分
        components.signal_freshness = self._calculate_signal_freshness(signals)
        
        # 6. Cross-Validation Score | 交叉驗證評分
        components.cross_validation_score = self._calculate_cross_validation_score(signals)
        
        # 7. Overall Confidence | 整體信心
        components.overall_confidence = self._calculate_weighted_confidence(components)
        
        logger.info(f"Calculated comprehensive confidence: {components.overall_confidence:.3f}")
        return components
    
    def _calculate_model_agreement(self, signals: List[TradingSignal]) -> float:
        """
        Calculate model agreement score | 計算模型一致性評分
        
        Measures how much the signals from different sources agree with each other.
        測量來自不同來源的信號彼此一致的程度。
        """
        if len(signals) < 2:
            return 1.0  # Single signal has perfect "agreement" with itself
        
        # Get signal directions and strengths | 獲取信號方向和強度
        signal_values = []
        for signal in signals:
            if signal.signal_type == SignalType.BUY:
                signal_values.append(signal.strength)
            elif signal.signal_type == SignalType.SELL:
                signal_values.append(-signal.strength)
            else:  # HOLD
                signal_values.append(0.0)
        
        if not signal_values:
            return 0.0
        
        # Calculate agreement based on signal consistency | 基於信號一致性計算一致性
        signal_array = np.array(signal_values)
        
        # Method 1: Standard deviation based agreement | 方法1：基於標準差的一致性
        signal_std = np.std(signal_array)
        std_agreement = max(0.0, 1.0 - signal_std)
        
        # Method 2: Direction consistency | 方法2：方向一致性
        positive_signals = np.sum(signal_array > 0.1)
        negative_signals = np.sum(signal_array < -0.1)
        neutral_signals = len(signal_array) - positive_signals - negative_signals
        
        # Calculate direction dominance | 計算方向主導性
        max_direction = max(positive_signals, negative_signals, neutral_signals)
        direction_agreement = max_direction / len(signal_array)
        
        # Method 3: Correlation-based agreement | 方法3：基於相關性的一致性
        if len(signals) >= 3:
            # Calculate pairwise correlations | 計算成對相關性
            correlations = []
            for i in range(len(signal_values)):
                for j in range(i + 1, len(signal_values)):
                    # Simple correlation proxy | 簡單相關性代理
                    diff = abs(signal_values[i] - signal_values[j])
                    corr = max(0.0, 1.0 - diff)
                    correlations.append(corr)
            correlation_agreement = np.mean(correlations) if correlations else 0.0
        else:
            correlation_agreement = std_agreement
        
        # Combine agreement measures | 組合一致性測量
        final_agreement = (
            std_agreement * 0.4 +
            direction_agreement * 0.4 +
            correlation_agreement * 0.2
        )
        
        return min(1.0, max(0.0, final_agreement))
    
    def _calculate_historical_performance(self, 
                                        signals: List[TradingSignal],
                                        historical_results: Optional[List[Dict]] = None) -> float:
        """
        Calculate historical performance score | 計算歷史績效評分
        
        Evaluates the historical success rate of similar signals.
        評估類似信號的歷史成功率。
        """
        if not historical_results or not signals:
            return 0.5  # Neutral score when no history available
        
        # Aggregate signal sources | 聚合信號來源
        signal_sources = set(signal.source for signal in signals)
        
        # Calculate performance by source | 按來源計算績效
        source_performances = {}
        
        for source in signal_sources:
            source_results = [
                result for result in historical_results
                if result.get('signal_source') == source
            ]
            
            if source_results:
                # Calculate win rate and average return | 計算勝率和平均回報
                wins = sum(1 for r in source_results if r.get('profit_loss', 0) > 0)
                total_trades = len(source_results)
                win_rate = wins / total_trades if total_trades > 0 else 0.5
                
                # Calculate average return | 計算平均回報
                avg_return = np.mean([r.get('profit_loss', 0) for r in source_results])
                
                # Combine win rate and return | 結合勝率和回報
                performance_score = (win_rate * 0.7) + (min(1.0, max(0.0, avg_return + 0.5)) * 0.3)
                source_performances[source] = performance_score
            else:
                source_performances[source] = 0.5  # Neutral for new sources
        
        # Weight by signal confidence | 按信號信心加權
        if source_performances:
            weighted_performance = 0.0
            total_weight = 0.0
            
            for signal in signals:
                source_perf = source_performances.get(signal.source, 0.5)
                weight = signal.confidence
                weighted_performance += source_perf * weight
                total_weight += weight
            
            final_performance = weighted_performance / total_weight if total_weight > 0 else 0.5
        else:
            final_performance = 0.5
        
        return min(1.0, max(0.0, final_performance))
    
    def _calculate_signal_strength_score(self, signals: List[TradingSignal]) -> float:
        """
        Calculate signal strength score | 計算信號強度評分
        
        Evaluates the overall strength and conviction of the signals.
        評估信號的整體強度和確信程度。
        """
        if not signals:
            return 0.0
        
        # Calculate average signal strength | 計算平均信號強度
        strengths = [signal.strength for signal in signals]
        avg_strength = np.mean(strengths)
        
        # Calculate signal conviction (how far from neutral) | 計算信號確信度（與中性的距離）
        neutral_deviations = []
        for signal in signals:
            if signal.signal_type == SignalType.HOLD:
                deviation = 0.0  # HOLD signals have no conviction
            else:
                deviation = abs(signal.strength - 0.5)  # Distance from neutral
            neutral_deviations.append(deviation)
        
        avg_conviction = np.mean(neutral_deviations)
        
        # Calculate consistency of signal strengths | 計算信號強度一致性
        strength_consistency = 1.0 - (np.std(strengths) if len(strengths) > 1 else 0.0)
        
        # Combine metrics | 組合指標
        strength_score = (
            avg_strength * 0.4 +
            avg_conviction * 0.4 +
            strength_consistency * 0.2
        )
        
        return min(1.0, max(0.0, strength_score))
    
    def _calculate_volatility_adjustment(self, 
                                       signals: List[TradingSignal],
                                       market_data: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate volatility adjustment factor | 計算波動性調整因子
        
        Adjusts confidence based on current market volatility conditions.
        基於當前市場波動性條件調整信心。
        """
        if market_data is None or market_data.empty:
            return 0.5  # Neutral adjustment when no market data
        
        try:
            # Calculate recent volatility | 計算近期波動性
            if 'Close' in market_data.columns and len(market_data) >= self.volatility_window:
                returns = market_data['Close'].pct_change().dropna()
                recent_returns = returns.tail(self.volatility_window)
                current_volatility = recent_returns.std()
                
                # Calculate historical volatility for comparison | 計算歷史波動性以供比較
                if len(returns) >= self.volatility_window * 2:
                    historical_vol = returns.std()
                    vol_ratio = current_volatility / historical_vol if historical_vol > 0 else 1.0
                else:
                    vol_ratio = 1.0
                
                # Adjust confidence based on volatility regime | 基於波動性機制調整信心
                if vol_ratio <= 0.8:  # Low volatility
                    volatility_adjustment = 0.8  # Higher confidence in trending markets
                elif vol_ratio <= 1.2:  # Normal volatility
                    volatility_adjustment = 0.6  # Normal confidence
                elif vol_ratio <= 2.0:  # High volatility
                    volatility_adjustment = 0.4  # Lower confidence in choppy markets
                else:  # Extreme volatility
                    volatility_adjustment = 0.2  # Very low confidence
                
            else:
                volatility_adjustment = 0.5  # Neutral when insufficient data
            
        except Exception as e:
            logger.warning(f"Error calculating volatility adjustment: {e}")
            volatility_adjustment = 0.5
        
        return volatility_adjustment
    
    def _calculate_signal_freshness(self, signals: List[TradingSignal]) -> float:
        """
        Calculate signal freshness score | 計算信號新鮮度評分
        
        Evaluates how recent the signals are (fresher signals generally more reliable).
        評估信號的新近程度（更新的信號通常更可靠）。
        """
        if not signals:
            return 0.0
        
        current_time = datetime.now()
        freshness_scores = []
        
        for signal in signals:
            # Calculate age in minutes | 計算年齡（分鐘）
            signal_age = (current_time - signal.timestamp).total_seconds() / 60.0
            
            # Score based on age (fresher = higher score) | 基於年齡評分（更新 = 更高分）
            if signal_age <= 5:  # Very fresh (0-5 minutes)
                freshness = 1.0
            elif signal_age <= 15:  # Fresh (5-15 minutes)
                freshness = 0.8
            elif signal_age <= 60:  # Moderately fresh (15-60 minutes)
                freshness = 0.6
            elif signal_age <= 240:  # Aging (1-4 hours)
                freshness = 0.4
            elif signal_age <= 1440:  # Old (4-24 hours)
                freshness = 0.2
            else:  # Very old (>24 hours)
                freshness = 0.1
            
            freshness_scores.append(freshness)
        
        return np.mean(freshness_scores)
    
    def _calculate_cross_validation_score(self, signals: List[TradingSignal]) -> float:
        """
        Calculate cross-validation score | 計算交叉驗證評分
        
        Evaluates signal quality through cross-validation of different timeframes/methods.
        通過不同時間框架/方法的交叉驗證評估信號品質。
        """
        if not signals:
            return 0.0
        
        # This is a simplified cross-validation score | 這是一個簡化的交叉驗證評分
        # In a real implementation, this would involve actual cross-validation
        # 在實際實現中，這將涉及實際的交叉驗證
        
        # For now, use signal diversity as a proxy | 目前使用信號多樣性作為代理
        unique_sources = len(set(signal.source for signal in signals))
        max_diversity = min(len(signals), 5)  # Cap at 5 for normalization
        
        diversity_score = unique_sources / max_diversity if max_diversity > 0 else 0.0
        
        # Also consider metadata quality | 還要考慮元數據品質
        metadata_scores = []
        for signal in signals:
            metadata_quality = 0.0
            if signal.metadata:
                # Score based on metadata richness | 基於元數據豐富度評分
                metadata_quality = min(1.0, len(signal.metadata) / 5.0)
            metadata_scores.append(metadata_quality)
        
        avg_metadata_quality = np.mean(metadata_scores) if metadata_scores else 0.0
        
        # Combine diversity and metadata quality | 結合多樣性和元數據品質
        cross_validation_score = (diversity_score * 0.7) + (avg_metadata_quality * 0.3)
        
        return min(1.0, max(0.0, cross_validation_score))
    
    def _calculate_weighted_confidence(self, components: ConfidenceComponents) -> float:
        """
        Calculate final weighted confidence score | 計算最終權重信心評分
        
        Combines all confidence components using configured weights.
        使用配置權重組合所有信心組件。
        """
        weighted_sum = 0.0
        
        component_values = {
            ConfidenceFactorType.MODEL_AGREEMENT: components.model_agreement,
            ConfidenceFactorType.HISTORICAL_PERFORMANCE: components.historical_performance,
            ConfidenceFactorType.SIGNAL_STRENGTH: components.signal_strength,
            ConfidenceFactorType.MARKET_VOLATILITY: components.market_volatility_adjustment,
            ConfidenceFactorType.SIGNAL_FRESHNESS: components.signal_freshness,
            ConfidenceFactorType.CROSS_VALIDATION: components.cross_validation_score
        }
        
        for factor_type, value in component_values.items():
            weight = self.component_weights.get(factor_type, 0.0)
            weighted_sum += value * weight
        
        return min(1.0, max(0.0, weighted_sum))
    
    def set_component_weights(self, weights: Dict[ConfidenceFactorType, float]):
        """
        Set custom weights for confidence components | 設置信心組件的自定義權重
        
        Args:
            weights: Dictionary mapping factor types to weights | 將因子類型映射到權重的字典
        """
        # Normalize weights to sum to 1 | 標準化權重使總和為1
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.component_weights = {
                factor: weight / total_weight
                for factor, weight in weights.items()
            }
            logger.info(f"Updated confidence component weights: {self.component_weights}")
        else:
            logger.warning("Invalid weights provided - weights must sum to positive value")
    
    def add_signal_result(self, 
                         signal_info: Dict[str, Any], 
                         result: Dict[str, Any]):
        """
        Add trading result for signal performance tracking | 添加交易結果用於信號績效追蹤
        
        Args:
            signal_info: Information about the signal that was acted upon | 被執行信號的信息
            result: Trading result (profit/loss, duration, etc.) | 交易結果（盈虧、持續時間等）
        """
        combined_record = {
            **signal_info,
            **result,
            'timestamp': datetime.now().isoformat()
        }
        
        self.signal_history.append(combined_record)
        
        # Keep only recent history | 只保留近期歷史
        if len(self.signal_history) > self.lookback_periods:
            self.signal_history = self.signal_history[-self.lookback_periods:]
        
        # Update performance cache | 更新績效緩存
        self._update_performance_cache()
    
    def _update_performance_cache(self):
        """Update performance cache based on signal history | 基於信號歷史更新績效緩存"""
        if not self.signal_history:
            return
        
        # Group by signal source | 按信號來源分組
        source_results = {}
        for record in self.signal_history:
            source = record.get('signal_source', 'unknown')
            if source not in source_results:
                source_results[source] = []
            source_results[source].append(record)
        
        # Calculate performance metrics for each source | 為每個來源計算績效指標
        for source, results in source_results.items():
            if results:
                profits = [r.get('profit_loss', 0) for r in results]
                wins = sum(1 for p in profits if p > 0)
                total_trades = len(profits)
                
                self.performance_cache[source] = {
                    'win_rate': wins / total_trades if total_trades > 0 else 0.0,
                    'avg_return': np.mean(profits),
                    'total_trades': total_trades,
                    'recent_performance': np.mean(profits[-10:]) if len(profits) >= 10 else np.mean(profits)
                }
    
    def get_confidence_report(self, components: ConfidenceComponents) -> Dict[str, Any]:
        """
        Generate detailed confidence report | 生成詳細信心報告
        
        Args:
            components: Confidence components to report on | 要報告的信心組件
            
        Returns:
            Detailed confidence analysis report | 詳細信心分析報告
        """
        return {
            'overall_confidence': components.overall_confidence,
            'confidence_level': self._get_confidence_level(components.overall_confidence),
            'component_scores': components.to_dict(),
            'component_weights': {k.value: v for k, v in self.component_weights.items()},
            'recommendations': self._get_confidence_recommendations(components),
            'risk_assessment': self._assess_confidence_risk(components.overall_confidence)
        }
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get qualitative confidence level | 獲取定性信心水平"""
        if confidence >= 0.8:
            return "Very High | 非常高"
        elif confidence >= 0.7:
            return "High | 高"
        elif confidence >= 0.5:
            return "Moderate | 中等"
        elif confidence >= 0.3:
            return "Low | 低"
        else:
            return "Very Low | 非常低"
    
    def _get_confidence_recommendations(self, components: ConfidenceComponents) -> List[str]:
        """Get recommendations based on confidence components | 基於信心組件獲取建議"""
        recommendations = []
        
        if components.model_agreement < 0.5:
            recommendations.append("Low model agreement - consider waiting for more consensus")
            recommendations.append("模型一致性低 - 考慮等待更多共識")
        
        if components.historical_performance < 0.4:
            recommendations.append("Poor historical performance - reduce position size")
            recommendations.append("歷史績效差 - 減少倉位大小")
        
        if components.signal_strength < 0.5:
            recommendations.append("Weak signal strength - consider additional confirmation")
            recommendations.append("信號強度弱 - 考慮額外確認")
        
        if components.market_volatility_adjustment < 0.4:
            recommendations.append("High market volatility - use tighter risk management")
            recommendations.append("市場波動性高 - 使用更嚴格的風險管理")
        
        if components.signal_freshness < 0.3:
            recommendations.append("Stale signals detected - refresh data sources")
            recommendations.append("檢測到陳舊信號 - 刷新數據源")
        
        return recommendations
    
    def _assess_confidence_risk(self, confidence: float) -> str:
        """Assess risk level based on confidence | 基於信心評估風險水平"""
        if confidence >= 0.8:
            return "Low Risk | 低風險"
        elif confidence >= 0.6:
            return "Moderate Risk | 中等風險"
        elif confidence >= 0.4:
            return "High Risk | 高風險"
        else:
            return "Very High Risk | 非常高風險"