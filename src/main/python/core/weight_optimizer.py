"""
Dynamic Signal Weight Optimization | 動態信號權重優化

Advanced weight optimization system for trading signal combination based on historical 
performance, adaptive learning, and real-time market conditions.
基於歷史績效、自適應學習和實時市場條件的交易信號組合高級權重優化系統。

This module provides sophisticated weight optimization mechanisms for the AIFX trading system.
此模組為AIFX交易系統提供精細的權重優化機制。
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
import warnings

from .signal_combiner import TradingSignal, SignalType

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """
    Weight optimization methods | 權重優化方法
    """
    PERFORMANCE_BASED = "performance_based"
    RISK_ADJUSTED = "risk_adjusted"
    SHARPE_OPTIMIZED = "sharpe_optimized"
    KELLY_CRITERION = "kelly_criterion"
    ADAPTIVE_LEARNING = "adaptive_learning"
    ENSEMBLE_VOTING = "ensemble_voting"


class AdaptationStrategy(Enum):
    """
    Adaptation strategies for dynamic weighting | 動態權重的適應策略
    """
    EXPONENTIAL_DECAY = "exponential_decay"
    SLIDING_WINDOW = "sliding_window"
    MOMENTUM_BASED = "momentum_based"
    REGIME_AWARE = "regime_aware"


@dataclass
class WeightOptimizationConfig:
    """
    Configuration for weight optimization | 權重優化配置
    """
    optimization_method: OptimizationMethod = OptimizationMethod.PERFORMANCE_BASED
    adaptation_strategy: AdaptationStrategy = AdaptationStrategy.SLIDING_WINDOW
    lookback_window: int = 50
    min_trades_threshold: int = 10
    rebalance_frequency: int = 10  # Rebalance every N signals
    learning_rate: float = 0.01
    momentum_factor: float = 0.9
    regularization_factor: float = 0.01
    min_weight: float = 0.05
    max_weight: float = 0.8
    enable_regime_detection: bool = True
    risk_aversion: float = 0.5  # 0 = risk seeking, 1 = risk averse


@dataclass
class SourcePerformanceMetrics:
    """
    Performance metrics for signal sources | 信號來源的績效指標
    """
    source_name: str
    total_trades: int = 0
    winning_trades: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    std_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    recent_performance: float = 0.0
    confidence_score: float = 0.0
    last_updated: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary | 轉換為字典"""
        return {
            'source_name': self.source_name,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.win_rate,
            'avg_return': self.avg_return,
            'std_return': self.std_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'profit_factor': self.profit_factor,
            'recent_performance': self.recent_performance,
            'confidence_score': self.confidence_score,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }


class DynamicWeightOptimizer:
    """
    Dynamic weight optimization system | 動態權重優化系統
    
    Provides sophisticated weight optimization using historical performance analysis,
    adaptive learning algorithms, and real-time market condition adjustments.
    使用歷史績效分析、自適應學習算法和實時市場條件調整提供精細的權重優化。
    """
    
    def __init__(self, config: Optional[WeightOptimizationConfig] = None):
        """
        Initialize dynamic weight optimizer | 初始化動態權重優化器
        
        Args:
            config: Optimization configuration | 優化配置
        """
        self.config = config or WeightOptimizationConfig()
        
        # Performance tracking | 績效追蹤
        self.source_metrics: Dict[str, SourcePerformanceMetrics] = {}
        self.historical_weights: List[Dict[str, float]] = []
        self.trade_history: List[Dict[str, Any]] = []
        
        # Optimization state | 優化狀態
        self.current_weights: Dict[str, float] = {}
        self.momentum_weights: Dict[str, float] = {}
        self.last_rebalance: Optional[datetime] = None
        self.trades_since_rebalance: int = 0
        
        # Market regime detection | 市場機制檢測
        self.regime_detector = MarketRegimeDetector() if self.config.enable_regime_detection else None
        
        # Adaptive learning state | 自適應學習狀態
        self.learning_history: List[Dict[str, Any]] = []
        self.gradient_cache: Dict[str, float] = {}
        
        logger.info(f"Initialized DynamicWeightOptimizer with method: {self.config.optimization_method.value}")
    
    def update_performance(self, source: str, trade_result: Dict[str, Any]):
        """
        Update performance metrics for a signal source | 更新信號來源的績效指標
        
        Args:
            source: Signal source identifier | 信號來源標識符
            trade_result: Trading result data | 交易結果數據
        """
        if source not in self.source_metrics:
            self.source_metrics[source] = SourcePerformanceMetrics(source_name=source)
        
        metrics = self.source_metrics[source]
        
        # Update basic metrics | 更新基本指標
        profit_loss = trade_result.get('profit_loss', 0.0)
        metrics.total_trades += 1
        if profit_loss > 0:
            metrics.winning_trades += 1
        
        # Track trade history for this source | 追蹤此來源的交易歷史
        trade_record = {
            'source': source,
            'timestamp': datetime.now(),
            'profit_loss': profit_loss,
            **trade_result
        }
        self.trade_history.append(trade_record)
        
        # Calculate updated metrics | 計算更新的指標
        self._recalculate_source_metrics(source)
        
        # Check if rebalancing is needed | 檢查是否需要重新平衡
        self.trades_since_rebalance += 1
        if self.trades_since_rebalance >= self.config.rebalance_frequency:
            self._trigger_rebalancing()
        
        logger.debug(f"Updated performance for {source}: Win rate={metrics.win_rate:.3f}, "
                    f"Avg return={metrics.avg_return:.4f}")
    
    def _recalculate_source_metrics(self, source: str):
        """
        Recalculate comprehensive metrics for a source | 重新計算來源的綜合指標
        
        Args:
            source: Signal source identifier | 信號來源標識符
        """
        metrics = self.source_metrics[source]
        
        # Get relevant trade history | 獲取相關交易歷史
        source_trades = [
            trade for trade in self.trade_history 
            if trade['source'] == source
        ]
        
        if not source_trades:
            return
        
        # Get lookback window of trades | 獲取回望窗口的交易
        recent_trades = source_trades[-self.config.lookback_window:]
        profit_losses = [trade['profit_loss'] for trade in recent_trades]
        
        # Basic metrics | 基本指標
        metrics.total_trades = len(source_trades)
        metrics.winning_trades = sum(1 for pl in profit_losses if pl > 0)
        metrics.win_rate = metrics.winning_trades / len(profit_losses) if profit_losses else 0.0
        metrics.avg_return = np.mean(profit_losses) if profit_losses else 0.0
        metrics.std_return = np.std(profit_losses) if len(profit_losses) > 1 else 0.0
        
        # Risk-adjusted metrics | 風險調整指標
        if metrics.std_return > 0:
            metrics.sharpe_ratio = metrics.avg_return / metrics.std_return
        else:
            metrics.sharpe_ratio = 0.0
        
        # Maximum drawdown calculation | 最大回撤計算
        if len(profit_losses) > 0:
            cumulative_returns = np.cumsum(profit_losses)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = running_max - cumulative_returns
            metrics.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Profit factor calculation | 獲利因子計算
        gross_profit = sum(pl for pl in profit_losses if pl > 0)
        gross_loss = abs(sum(pl for pl in profit_losses if pl < 0))
        if gross_loss > 0:
            metrics.profit_factor = gross_profit / gross_loss
        else:
            metrics.profit_factor = float('inf') if gross_profit > 0 else 0.0
        
        # Recent performance (last 10 trades) | 近期績效（最後10筆交易）
        recent_subset = profit_losses[-10:] if len(profit_losses) >= 10 else profit_losses
        metrics.recent_performance = np.mean(recent_subset) if recent_subset else 0.0
        
        # Confidence score based on trade count and consistency | 基於交易次數和一致性的信心評分
        trade_count_factor = min(1.0, len(profit_losses) / self.config.min_trades_threshold)
        consistency_factor = max(0.0, 1.0 - (metrics.std_return / (abs(metrics.avg_return) + 1e-6)))
        metrics.confidence_score = (trade_count_factor * 0.6) + (consistency_factor * 0.4)
        
        metrics.last_updated = datetime.now()
    
    def optimize_weights(self, signal_sources: List[str], 
                        market_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Optimize weights for signal sources | 優化信號來源權重
        
        Args:
            signal_sources: List of signal source identifiers | 信號來源標識符列表
            market_data: Current market data for regime detection | 用於機制檢測的當前市場數據
            
        Returns:
            Optimized weights dictionary | 優化權重字典
        """
        if not signal_sources:
            return {}
        
        # Initialize missing sources | 初始化缺少的來源
        for source in signal_sources:
            if source not in self.source_metrics:
                self.source_metrics[source] = SourcePerformanceMetrics(source_name=source)
        
        # Detect market regime if enabled | 如果啟用則檢測市場機制
        current_regime = None
        if self.regime_detector and market_data is not None:
            current_regime = self.regime_detector.detect_regime(market_data)
        
        # Apply optimization method | 應用優化方法
        if self.config.optimization_method == OptimizationMethod.PERFORMANCE_BASED:
            weights = self._optimize_performance_based(signal_sources)
        elif self.config.optimization_method == OptimizationMethod.RISK_ADJUSTED:
            weights = self._optimize_risk_adjusted(signal_sources)
        elif self.config.optimization_method == OptimizationMethod.SHARPE_OPTIMIZED:
            weights = self._optimize_sharpe_ratio(signal_sources)
        elif self.config.optimization_method == OptimizationMethod.KELLY_CRITERION:
            weights = self._optimize_kelly_criterion(signal_sources)
        elif self.config.optimization_method == OptimizationMethod.ADAPTIVE_LEARNING:
            weights = self._optimize_adaptive_learning(signal_sources)
        else:  # ENSEMBLE_VOTING
            weights = self._optimize_ensemble_voting(signal_sources)
        
        # Apply adaptation strategy | 應用適應策略
        weights = self._apply_adaptation_strategy(weights, signal_sources)
        
        # Apply constraints and normalization | 應用約束和標準化
        weights = self._apply_weight_constraints(weights)
        weights = self._normalize_weights(weights)
        
        # Update current weights and history | 更新當前權重和歷史
        self.current_weights = weights.copy()
        self.historical_weights.append({
            'timestamp': datetime.now().isoformat(),
            'weights': weights.copy(),
            'method': self.config.optimization_method.value,
            'regime': current_regime
        })
        
        # Keep history manageable | 保持歷史可管理
        if len(self.historical_weights) > 1000:
            self.historical_weights = self.historical_weights[-500:]
        
        logger.info(f"Optimized weights using {self.config.optimization_method.value}: {weights}")
        return weights
    
    def _optimize_performance_based(self, signal_sources: List[str]) -> Dict[str, float]:
        """Performance-based weight optimization | 基於績效的權重優化"""
        weights = {}
        
        for source in signal_sources:
            metrics = self.source_metrics.get(source)
            if not metrics or metrics.total_trades < self.config.min_trades_threshold:
                weights[source] = 1.0 / len(signal_sources)  # Equal weight for new sources
                continue
            
            # Combine multiple performance factors | 結合多個績效因子
            win_rate_score = metrics.win_rate
            return_score = max(0.0, metrics.avg_return + 0.5)  # Shift to positive
            consistency_score = metrics.confidence_score
            recent_score = max(0.0, metrics.recent_performance + 0.5)
            
            # Weighted combination of factors | 因子的權重組合
            performance_score = (
                win_rate_score * 0.3 +
                return_score * 0.3 +
                consistency_score * 0.2 +
                recent_score * 0.2
            )
            
            weights[source] = performance_score
        
        return weights
    
    def _optimize_risk_adjusted(self, signal_sources: List[str]) -> Dict[str, float]:
        """Risk-adjusted weight optimization | 風險調整權重優化"""
        weights = {}
        risk_aversion = self.config.risk_aversion
        
        for source in signal_sources:
            metrics = self.source_metrics.get(source)
            if not metrics or metrics.total_trades < self.config.min_trades_threshold:
                weights[source] = 1.0 / len(signal_sources)
                continue
            
            # Risk-adjusted score | 風險調整評分
            expected_return = metrics.avg_return
            risk_penalty = risk_aversion * (metrics.std_return ** 2)
            risk_adjusted_score = expected_return - risk_penalty
            
            # Add drawdown penalty | 添加回撤懲罰
            drawdown_penalty = risk_aversion * metrics.max_drawdown
            final_score = risk_adjusted_score - drawdown_penalty
            
            weights[source] = max(0.01, final_score + 1.0)  # Ensure positive weights
        
        return weights
    
    def _optimize_sharpe_ratio(self, signal_sources: List[str]) -> Dict[str, float]:
        """Sharpe ratio optimization | 夏普比率優化"""
        weights = {}
        
        for source in signal_sources:
            metrics = self.source_metrics.get(source)
            if not metrics or metrics.total_trades < self.config.min_trades_threshold:
                weights[source] = 1.0 / len(signal_sources)
                continue
            
            # Use Sharpe ratio as primary factor | 使用夏普比率作為主要因子
            sharpe_score = max(0.1, metrics.sharpe_ratio + 1.0)  # Shift to ensure positive
            
            # Adjust for recent performance | 調整近期績效
            recent_adjustment = 1.0 + (0.2 * metrics.recent_performance)
            adjusted_sharpe = sharpe_score * recent_adjustment
            
            weights[source] = adjusted_sharpe
        
        return weights
    
    def _optimize_kelly_criterion(self, signal_sources: List[str]) -> Dict[str, float]:
        """Kelly Criterion-based optimization | 基於凱利公式的優化"""
        weights = {}
        
        for source in signal_sources:
            metrics = self.source_metrics.get(source)
            if not metrics or metrics.total_trades < self.config.min_trades_threshold:
                weights[source] = 1.0 / len(signal_sources)
                continue
            
            # Kelly Criterion calculation | 凱利公式計算
            win_rate = metrics.win_rate
            if win_rate > 0 and win_rate < 1:
                # Simplified Kelly formula | 簡化凱利公式
                avg_win = abs(metrics.avg_return) if metrics.avg_return > 0 else 0.01
                avg_loss = abs(metrics.avg_return) if metrics.avg_return < 0 else 0.01
                
                kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_fraction = max(0.05, min(0.5, kelly_fraction))  # Constrain Kelly
            else:
                kelly_fraction = 1.0 / len(signal_sources)
            
            weights[source] = kelly_fraction
        
        return weights
    
    def _optimize_adaptive_learning(self, signal_sources: List[str]) -> Dict[str, float]:
        """Adaptive learning-based optimization | 基於自適應學習的優化"""
        weights = {}
        learning_rate = self.config.learning_rate
        
        for source in signal_sources:
            # Get previous weight | 獲取前一個權重
            prev_weight = self.current_weights.get(source, 1.0 / len(signal_sources))
            
            metrics = self.source_metrics.get(source)
            if not metrics or metrics.total_trades < self.config.min_trades_threshold:
                weights[source] = prev_weight
                continue
            
            # Calculate performance gradient | 計算績效梯度
            performance_signal = metrics.recent_performance
            gradient = performance_signal  # Simplified gradient
            
            # Apply momentum | 應用動量
            momentum = self.momentum_weights.get(source, 0.0)
            momentum = self.config.momentum_factor * momentum + (1 - self.config.momentum_factor) * gradient
            self.momentum_weights[source] = momentum
            
            # Update weight with learning rule | 使用學習規則更新權重
            new_weight = prev_weight + learning_rate * momentum
            new_weight = max(self.config.min_weight, min(self.config.max_weight, new_weight))
            
            weights[source] = new_weight
        
        return weights
    
    def _optimize_ensemble_voting(self, signal_sources: List[str]) -> Dict[str, float]:
        """Ensemble voting optimization | 集成投票優化"""
        # Combine multiple optimization methods | 結合多種優化方法
        methods = [
            self._optimize_performance_based(signal_sources),
            self._optimize_risk_adjusted(signal_sources),
            self._optimize_sharpe_ratio(signal_sources)
        ]
        
        # Voting weights for each method | 每種方法的投票權重
        method_weights = [0.4, 0.3, 0.3]
        
        # Combine results | 結合結果
        weights = {}
        for source in signal_sources:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for method_result, method_weight in zip(methods, method_weights):
                source_weight = method_result.get(source, 0.0)
                weighted_sum += source_weight * method_weight
                total_weight += method_weight
            
            weights[source] = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return weights
    
    def _apply_adaptation_strategy(self, weights: Dict[str, float], 
                                 signal_sources: List[str]) -> Dict[str, float]:
        """Apply adaptation strategy to weights | 對權重應用適應策略"""
        if self.config.adaptation_strategy == AdaptationStrategy.EXPONENTIAL_DECAY:
            return self._apply_exponential_decay(weights, signal_sources)
        elif self.config.adaptation_strategy == AdaptationStrategy.MOMENTUM_BASED:
            return self._apply_momentum_adaptation(weights, signal_sources)
        elif self.config.adaptation_strategy == AdaptationStrategy.REGIME_AWARE:
            return self._apply_regime_awareness(weights, signal_sources)
        else:  # SLIDING_WINDOW (default)
            return weights  # Already applied in metric calculation
    
    def _apply_exponential_decay(self, weights: Dict[str, float], 
                               signal_sources: List[str]) -> Dict[str, float]:
        """Apply exponential decay to historical performance | 對歷史績效應用指數衰減"""
        decay_factor = 0.95  # Decay recent performance more slowly
        
        for source in signal_sources:
            metrics = self.source_metrics.get(source)
            if metrics and metrics.total_trades > 0:
                # Apply exponential decay to recent performance weighting
                decay_weight = decay_factor ** (datetime.now() - (metrics.last_updated or datetime.now())).days
                weights[source] *= (1.0 + 0.2 * metrics.recent_performance * decay_weight)
        
        return weights
    
    def _apply_momentum_adaptation(self, weights: Dict[str, float], 
                                 signal_sources: List[str]) -> Dict[str, float]:
        """Apply momentum-based adaptation | 應用基於動量的適應"""
        momentum_strength = 0.3
        
        for source in signal_sources:
            if source in self.current_weights:
                momentum = weights[source] - self.current_weights[source]
                weights[source] += momentum_strength * momentum
        
        return weights
    
    def _apply_regime_awareness(self, weights: Dict[str, float], 
                              signal_sources: List[str]) -> Dict[str, float]:
        """Apply regime-aware weight adjustments | 應用機制感知權重調整"""
        # This is a simplified regime awareness - in practice would be more sophisticated
        # 這是一個簡化的機制感知 - 在實際中會更復雜
        
        # Check recent market volatility trend from metrics
        # 從指標檢查近期市場波動趨勢
        high_volatility_sources = []
        stable_sources = []
        
        for source in signal_sources:
            metrics = self.source_metrics.get(source)
            if metrics and metrics.std_return > 0.02:  # High volatility threshold
                high_volatility_sources.append(source)
            else:
                stable_sources.append(source)
        
        # Adjust weights based on detected regime
        # 基於檢測到的機制調整權重
        regime_adjustment = 0.1
        for source in high_volatility_sources:
            weights[source] *= (1.0 - regime_adjustment)  # Reduce volatile sources
        
        for source in stable_sources:
            weights[source] *= (1.0 + regime_adjustment)  # Boost stable sources
        
        return weights
    
    def _apply_weight_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply weight constraints | 應用權重約束"""
        constrained_weights = {}
        
        for source, weight in weights.items():
            # Apply min/max constraints
            constrained_weight = max(self.config.min_weight, 
                                   min(self.config.max_weight, weight))
            constrained_weights[source] = constrained_weight
        
        return constrained_weights
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1 | 標準化權重使總和為1"""
        total_weight = sum(weights.values())
        if total_weight <= 0:
            # Equal weights fallback | 等權重回退
            return {source: 1.0 / len(weights) for source in weights.keys()}
        
        return {source: weight / total_weight for source, weight in weights.items()}
    
    def _trigger_rebalancing(self):
        """Trigger weight rebalancing | 觸發權重重新平衡"""
        self.last_rebalance = datetime.now()
        self.trades_since_rebalance = 0
        logger.info("Triggered weight rebalancing")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Get comprehensive optimization report | 獲取綜合優化報告
        
        Returns:
            Detailed optimization analysis report | 詳細優化分析報告
        """
        source_summaries = {}
        for source, metrics in self.source_metrics.items():
            source_summaries[source] = metrics.to_dict()
        
        return {
            'current_weights': self.current_weights,
            'optimization_config': {
                'method': self.config.optimization_method.value,
                'adaptation_strategy': self.config.adaptation_strategy.value,
                'lookback_window': self.config.lookback_window,
                'rebalance_frequency': self.config.rebalance_frequency
            },
            'source_performance': source_summaries,
            'recent_rebalancing': {
                'last_rebalance': self.last_rebalance.isoformat() if self.last_rebalance else None,
                'trades_since_rebalance': self.trades_since_rebalance,
                'total_weight_updates': len(self.historical_weights)
            },
            'learning_metrics': {
                'momentum_weights': self.momentum_weights,
                'gradient_cache': self.gradient_cache
            }
        }
    
    def update_configuration(self, new_config: WeightOptimizationConfig):
        """Update optimization configuration | 更新優化配置"""
        self.config = new_config
        logger.info(f"Updated optimization configuration: {new_config.optimization_method.value}")


class MarketRegimeDetector:
    """
    Simple market regime detector | 簡單市場機制檢測器
    
    Detects basic market regimes like trending vs ranging markets.
    檢測基本市場機制，如趨勢市場vs震盪市場。
    """
    
    def __init__(self, volatility_threshold: float = 0.02, trend_threshold: float = 0.05):
        """
        Initialize regime detector | 初始化機制檢測器
        
        Args:
            volatility_threshold: Threshold for high volatility detection | 高波動性檢測閾值
            trend_threshold: Threshold for trend detection | 趨勢檢測閾值
        """
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
    
    def detect_regime(self, market_data: pd.DataFrame, window: int = 20) -> str:
        """
        Detect current market regime | 檢測當前市場機制
        
        Args:
            market_data: Market price data | 市場價格數據
            window: Analysis window size | 分析窗口大小
            
        Returns:
            Market regime identifier | 市場機制標識符
        """
        try:
            if market_data.empty or 'Close' not in market_data.columns:
                return 'unknown'
            
            recent_data = market_data.tail(window)
            returns = recent_data['Close'].pct_change().dropna()
            
            if len(returns) < 5:
                return 'unknown'
            
            # Calculate volatility | 計算波動性
            volatility = returns.std()
            
            # Calculate trend strength | 計算趨勢強度
            price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
            
            # Classify regime | 分類機制
            if volatility > self.volatility_threshold:
                if abs(price_change) > self.trend_threshold:
                    return 'high_volatility_trending'
                else:
                    return 'high_volatility_ranging'
            else:
                if abs(price_change) > self.trend_threshold:
                    return 'low_volatility_trending'
                else:
                    return 'low_volatility_ranging'
        
        except Exception as e:
            logger.warning(f"Error in regime detection: {e}")
            return 'unknown'