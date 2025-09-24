#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Diversification Engine | 投資組合多元化引擎
=================================================

Advanced portfolio diversification system with dynamic allocation, correlation analysis,
and intelligent position management for multi-currency forex trading.

具有動態配置、相關性分析和智能倉位管理的高級投資組合多元化系統，適用於多貨幣外匯交易。

Features | 功能:
- Dynamic asset allocation based on market conditions
- Real-time correlation analysis and adjustment
- Risk parity and volatility targeting
- Currency exposure balancing
- Momentum and mean-reversion strategies
- Portfolio rebalancing algorithms

Author: AIFX Development Team
Date: 2025-01-15
Version: 2.0.0 - Live Trading Enhancement
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)

class AllocationStrategy(Enum):
    """Portfolio allocation strategies | 投資組合配置策略"""
    EQUAL_WEIGHT = "EQUAL_WEIGHT"
    RISK_PARITY = "RISK_PARITY"
    VOLATILITY_TARGET = "VOLATILITY_TARGET"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    CORRELATION_AWARE = "CORRELATION_AWARE"
    DYNAMIC_ADAPTIVE = "DYNAMIC_ADAPTIVE"

class RebalanceFrequency(Enum):
    """Portfolio rebalancing frequency | 投資組合再平衡頻率"""
    INTRADAY = "INTRADAY"      # Every 4 hours
    DAILY = "DAILY"            # Once per day
    WEEKLY = "WEEKLY"          # Once per week
    MONTHLY = "MONTHLY"        # Once per month
    THRESHOLD = "THRESHOLD"     # Based on deviation thresholds

@dataclass
class CurrencyAllocation:
    """Currency allocation configuration | 貨幣配置配置"""
    currency_pair: str
    target_weight: float
    current_weight: float
    min_weight: float
    max_weight: float
    volatility: float
    correlation_score: float
    momentum_score: float
    mean_reversion_score: float
    last_rebalance: datetime = field(default_factory=datetime.now)

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics | 投資組合績效指標"""
    timestamp: datetime = field(default_factory=datetime.now)
    total_value: float = 0.0
    total_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0

    # Diversification metrics
    effective_number_of_assets: float = 0.0
    diversification_ratio: float = 0.0
    concentration_index: float = 0.0

    # Risk metrics
    portfolio_var_95: float = 0.0
    expected_shortfall: float = 0.0
    correlation_average: float = 0.0

@dataclass
class RebalanceSignal:
    """Portfolio rebalance signal | 投資組合再平衡信號"""
    currency_pair: str
    current_allocation: float
    target_allocation: float
    recommended_action: str  # BUY, SELL, HOLD
    size_adjustment: float
    reason: str
    urgency: str  # LOW, MEDIUM, HIGH, CRITICAL

class PortfolioDiversificationEngine:
    """
    Advanced Portfolio Diversification Engine | 高級投資組合多元化引擎

    Sophisticated portfolio management system with dynamic allocation strategies,
    real-time correlation analysis, and intelligent rebalancing for forex trading.

    具有動態配置策略、實時相關性分析和智能再平衡的復雜投資組合管理系統。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize portfolio diversification engine
        初始化投資組合多元化引擎

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Portfolio configuration
        self.portfolio_value = 10000.0  # Starting portfolio value
        self.target_volatility = 0.15   # 15% target volatility
        self.max_position_size = 0.25   # 25% maximum position size
        self.min_position_size = 0.05   # 5% minimum position size
        self.rebalance_threshold = 0.05  # 5% deviation threshold

        # Strategy configuration
        self.allocation_strategy = AllocationStrategy.DYNAMIC_ADAPTIVE
        self.rebalance_frequency = RebalanceFrequency.THRESHOLD

        # Currency pairs and their configurations
        self.currency_pairs = {
            'USD/JPY': CurrencyAllocation(
                currency_pair='USD/JPY',
                target_weight=0.30,
                current_weight=0.00,
                min_weight=0.05,
                max_weight=0.40,
                volatility=0.013,
                correlation_score=0.0,
                momentum_score=0.0,
                mean_reversion_score=0.0
            ),
            'EUR/USD': CurrencyAllocation(
                currency_pair='EUR/USD',
                target_weight=0.25,
                current_weight=0.00,
                min_weight=0.05,
                max_weight=0.35,
                volatility=0.012,
                correlation_score=0.0,
                momentum_score=0.0,
                mean_reversion_score=0.0
            ),
            'GBP/USD': CurrencyAllocation(
                currency_pair='GBP/USD',
                target_weight=0.25,
                current_weight=0.00,
                min_weight=0.05,
                max_weight=0.35,
                volatility=0.015,
                correlation_score=0.0,
                momentum_score=0.0,
                mean_reversion_score=0.0
            ),
            'AUD/USD': CurrencyAllocation(
                currency_pair='AUD/USD',
                target_weight=0.20,
                current_weight=0.00,
                min_weight=0.05,
                max_weight=0.30,
                volatility=0.016,
                correlation_score=0.0,
                momentum_score=0.0,
                mean_reversion_score=0.0
            )
        }

        # Market data and analytics
        self.price_history: Dict[str, List[Dict]] = {}
        self.correlation_matrix = np.eye(len(self.currency_pairs))
        self.returns_history: Dict[str, List[float]] = {}

        # Performance tracking
        self.portfolio_metrics_history: List[PortfolioMetrics] = []
        self.rebalance_history: List[Dict] = []

        # Current positions
        self.current_positions: Dict[str, Dict] = {}

        logger.info("Portfolio Diversification Engine initialized | 投資組合多元化引擎已初始化")

    async def initialize_portfolio(self, initial_allocation: Optional[Dict[str, float]] = None):
        """
        Initialize portfolio with starting allocation
        使用初始配置初始化投資組合

        Args:
            initial_allocation: Custom initial allocation weights
        """
        try:
            logger.info("🎯 Initializing portfolio allocation...")

            if initial_allocation:
                # Use custom allocation
                total_weight = sum(initial_allocation.values())
                if abs(total_weight - 1.0) > 0.01:  # Allow 1% tolerance
                    logger.warning(f"⚠️ Initial allocation weights sum to {total_weight:.2f}, normalizing...")
                    for pair in initial_allocation:
                        initial_allocation[pair] /= total_weight

                for pair, weight in initial_allocation.items():
                    if pair in self.currency_pairs:
                        self.currency_pairs[pair].target_weight = weight

            # Calculate initial metrics
            await self._update_portfolio_metrics()

            logger.info("✅ Portfolio initialized with target allocation:")
            for pair, allocation in self.currency_pairs.items():
                logger.info(f"   {pair}: {allocation.target_weight:.1%}")

        except Exception as e:
            logger.error(f"Portfolio initialization error: {e}")

    async def analyze_diversification_opportunities(self) -> Dict[str, Any]:
        """
        Analyze current portfolio diversification and identify opportunities
        分析當前投資組合多元化並識別機會

        Returns:
            Dict with diversification analysis and recommendations
        """
        try:
            logger.info("📊 Analyzing diversification opportunities...")

            # Update correlations
            await self._update_correlation_matrix()

            # Calculate diversification metrics
            diversification_metrics = await self._calculate_diversification_metrics()

            # Identify optimization opportunities
            opportunities = await self._identify_diversification_opportunities()

            # Generate rebalance signals
            rebalance_signals = await self._generate_rebalance_signals()

            analysis = {
                'timestamp': datetime.now().isoformat(),
                'current_allocation': {
                    pair: allocation.current_weight
                    for pair, allocation in self.currency_pairs.items()
                },
                'target_allocation': {
                    pair: allocation.target_weight
                    for pair, allocation in self.currency_pairs.items()
                },
                'diversification_metrics': diversification_metrics,
                'correlation_analysis': {
                    'average_correlation': float(np.mean(self.correlation_matrix[np.triu_indices(len(self.correlation_matrix), k=1)])),
                    'max_correlation': float(np.max(self.correlation_matrix[np.triu_indices(len(self.correlation_matrix), k=1)])),
                    'min_correlation': float(np.min(self.correlation_matrix[np.triu_indices(len(self.correlation_matrix), k=1)])),
                    'correlation_matrix': self.correlation_matrix.tolist()
                },
                'optimization_opportunities': opportunities,
                'rebalance_signals': rebalance_signals,
                'allocation_strategy': self.allocation_strategy.value,
                'recommendations': await self._generate_allocation_recommendations()
            }

            return analysis

        except Exception as e:
            logger.error(f"Diversification analysis error: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    async def optimize_allocation(self, strategy: AllocationStrategy = None) -> Dict[str, float]:
        """
        Optimize portfolio allocation using specified strategy
        使用指定策略優化投資組合配置

        Args:
            strategy: Allocation strategy to use

        Returns:
            Dict with optimized allocation weights
        """
        try:
            strategy = strategy or self.allocation_strategy
            logger.info(f"🎯 Optimizing allocation using {strategy.value} strategy...")

            if strategy == AllocationStrategy.EQUAL_WEIGHT:
                optimized_allocation = await self._optimize_equal_weight()
            elif strategy == AllocationStrategy.RISK_PARITY:
                optimized_allocation = await self._optimize_risk_parity()
            elif strategy == AllocationStrategy.VOLATILITY_TARGET:
                optimized_allocation = await self._optimize_volatility_target()
            elif strategy == AllocationStrategy.MOMENTUM:
                optimized_allocation = await self._optimize_momentum()
            elif strategy == AllocationStrategy.MEAN_REVERSION:
                optimized_allocation = await self._optimize_mean_reversion()
            elif strategy == AllocationStrategy.CORRELATION_AWARE:
                optimized_allocation = await self._optimize_correlation_aware()
            elif strategy == AllocationStrategy.DYNAMIC_ADAPTIVE:
                optimized_allocation = await self._optimize_dynamic_adaptive()
            else:
                optimized_allocation = await self._optimize_equal_weight()

            # Update target weights
            for pair, weight in optimized_allocation.items():
                if pair in self.currency_pairs:
                    self.currency_pairs[pair].target_weight = weight

            logger.info("✅ Allocation optimized:")
            for pair, weight in optimized_allocation.items():
                logger.info(f"   {pair}: {weight:.1%}")

            return optimized_allocation

        except Exception as e:
            logger.error(f"Allocation optimization error: {e}")
            return {pair: allocation.target_weight for pair, allocation in self.currency_pairs.items()}

    async def _optimize_equal_weight(self) -> Dict[str, float]:
        """Equal weight allocation | 等權重配置"""
        num_pairs = len(self.currency_pairs)
        equal_weight = 1.0 / num_pairs

        return {pair: equal_weight for pair in self.currency_pairs.keys()}

    async def _optimize_risk_parity(self) -> Dict[str, float]:
        """Risk parity allocation (equal risk contribution) | 風險平價配置（等風險貢獻）"""
        try:
            # Calculate inverse volatility weights
            volatilities = np.array([allocation.volatility for allocation in self.currency_pairs.values()])
            inverse_vol = 1.0 / volatilities
            weights = inverse_vol / np.sum(inverse_vol)

            allocation = {}
            for i, pair in enumerate(self.currency_pairs.keys()):
                allocation[pair] = float(weights[i])

            return allocation

        except Exception as e:
            logger.error(f"Risk parity optimization error: {e}")
            return await self._optimize_equal_weight()

    async def _optimize_volatility_target(self) -> Dict[str, float]:
        """Volatility targeting allocation | 波動率目標配置"""
        try:
            # Target portfolio volatility
            target_vol = self.target_volatility

            # Calculate weights to achieve target volatility
            volatilities = np.array([allocation.volatility for allocation in self.currency_pairs.values()])

            # Simplified volatility targeting (assumes low correlation)
            weights = np.sqrt(target_vol) / np.sqrt(volatilities)
            weights = weights / np.sum(weights)  # Normalize

            allocation = {}
            for i, pair in enumerate(self.currency_pairs.keys()):
                allocation[pair] = float(weights[i])

            return allocation

        except Exception as e:
            logger.error(f"Volatility targeting optimization error: {e}")
            return await self._optimize_equal_weight()

    async def _optimize_momentum(self) -> Dict[str, float]:
        """Momentum-based allocation | 基於動量的配置"""
        try:
            # Calculate momentum scores
            momentum_scores = []
            for pair, allocation in self.currency_pairs.items():
                momentum = await self._calculate_momentum_score(pair)
                allocation.momentum_score = momentum
                momentum_scores.append(max(0, momentum))  # Only positive momentum

            # Normalize momentum scores to weights
            total_momentum = sum(momentum_scores)
            if total_momentum <= 0:
                return await self._optimize_equal_weight()

            allocation = {}
            for i, pair in enumerate(self.currency_pairs.keys()):
                allocation[pair] = momentum_scores[i] / total_momentum

            return allocation

        except Exception as e:
            logger.error(f"Momentum optimization error: {e}")
            return await self._optimize_equal_weight()

    async def _optimize_mean_reversion(self) -> Dict[str, float]:
        """Mean reversion allocation | 均值回歸配置"""
        try:
            # Calculate mean reversion scores
            reversion_scores = []
            for pair, allocation in self.currency_pairs.items():
                reversion = await self._calculate_mean_reversion_score(pair)
                allocation.mean_reversion_score = reversion
                reversion_scores.append(max(0, reversion))  # Only positive reversion signals

            # Normalize reversion scores to weights
            total_reversion = sum(reversion_scores)
            if total_reversion <= 0:
                return await self._optimize_equal_weight()

            allocation = {}
            for i, pair in enumerate(self.currency_pairs.keys()):
                allocation[pair] = reversion_scores[i] / total_reversion

            return allocation

        except Exception as e:
            logger.error(f"Mean reversion optimization error: {e}")
            return await self._optimize_equal_weight()

    async def _optimize_correlation_aware(self) -> Dict[str, float]:
        """Correlation-aware allocation | 相關性感知配置"""
        try:
            # Start with equal weights
            n = len(self.currency_pairs)
            weights = np.ones(n) / n

            # Adjust weights based on correlation matrix
            # Reduce weights for highly correlated pairs
            avg_correlations = np.mean(np.abs(self.correlation_matrix), axis=1)

            # Inverse correlation weighting
            correlation_weights = 1.0 / (1.0 + avg_correlations)
            correlation_weights = correlation_weights / np.sum(correlation_weights)

            allocation = {}
            for i, pair in enumerate(self.currency_pairs.keys()):
                allocation[pair] = float(correlation_weights[i])

            return allocation

        except Exception as e:
            logger.error(f"Correlation-aware optimization error: {e}")
            return await self._optimize_equal_weight()

    async def _optimize_dynamic_adaptive(self) -> Dict[str, float]:
        """Dynamic adaptive allocation combining multiple strategies | 結合多種策略的動態自適應配置"""
        try:
            # Get allocations from different strategies
            equal_weight = await self._optimize_equal_weight()
            risk_parity = await self._optimize_risk_parity()
            momentum = await self._optimize_momentum()
            mean_reversion = await self._optimize_mean_reversion()
            correlation_aware = await self._optimize_correlation_aware()

            # Dynamic weighting based on market conditions
            market_volatility = await self._calculate_market_volatility()
            momentum_strength = await self._calculate_momentum_strength()
            correlation_level = np.mean(np.abs(self.correlation_matrix[np.triu_indices(len(self.correlation_matrix), k=1)]))

            # Strategy weights based on market conditions
            if market_volatility > 0.02:  # High volatility - prefer risk parity
                strategy_weights = {'risk_parity': 0.4, 'equal_weight': 0.3, 'correlation_aware': 0.3}
            elif momentum_strength > 0.6:  # Strong momentum - prefer momentum
                strategy_weights = {'momentum': 0.5, 'risk_parity': 0.3, 'equal_weight': 0.2}
            elif correlation_level > 0.6:  # High correlation - prefer correlation-aware
                strategy_weights = {'correlation_aware': 0.5, 'risk_parity': 0.3, 'equal_weight': 0.2}
            else:  # Balanced conditions
                strategy_weights = {'equal_weight': 0.3, 'risk_parity': 0.3, 'momentum': 0.2, 'mean_reversion': 0.2}

            # Combine allocations
            combined_allocation = {}
            for pair in self.currency_pairs.keys():
                weight = 0.0
                if 'equal_weight' in strategy_weights:
                    weight += equal_weight[pair] * strategy_weights['equal_weight']
                if 'risk_parity' in strategy_weights:
                    weight += risk_parity[pair] * strategy_weights['risk_parity']
                if 'momentum' in strategy_weights:
                    weight += momentum[pair] * strategy_weights['momentum']
                if 'mean_reversion' in strategy_weights:
                    weight += mean_reversion[pair] * strategy_weights['mean_reversion']
                if 'correlation_aware' in strategy_weights:
                    weight += correlation_aware[pair] * strategy_weights['correlation_aware']

                combined_allocation[pair] = weight

            # Ensure weights sum to 1.0
            total_weight = sum(combined_allocation.values())
            for pair in combined_allocation:
                combined_allocation[pair] /= total_weight

            return combined_allocation

        except Exception as e:
            logger.error(f"Dynamic adaptive optimization error: {e}")
            return await self._optimize_equal_weight()

    async def generate_rebalance_signals(self) -> List[RebalanceSignal]:
        """
        Generate portfolio rebalance signals
        生成投資組合再平衡信號

        Returns:
            List of rebalance signals
        """
        try:
            signals = []

            for pair, allocation in self.currency_pairs.items():
                deviation = allocation.current_weight - allocation.target_weight
                abs_deviation = abs(deviation)

                if abs_deviation > self.rebalance_threshold:
                    # Determine action
                    if deviation > 0:
                        action = "SELL"  # Overweight - sell
                        size_adjustment = -abs_deviation
                    else:
                        action = "BUY"   # Underweight - buy
                        size_adjustment = abs_deviation

                    # Determine urgency
                    if abs_deviation > 0.15:  # 15%
                        urgency = "CRITICAL"
                    elif abs_deviation > 0.10:  # 10%
                        urgency = "HIGH"
                    elif abs_deviation > 0.05:  # 5%
                        urgency = "MEDIUM"
                    else:
                        urgency = "LOW"

                    signal = RebalanceSignal(
                        currency_pair=pair,
                        current_allocation=allocation.current_weight,
                        target_allocation=allocation.target_weight,
                        recommended_action=action,
                        size_adjustment=size_adjustment,
                        reason=f"Allocation deviation: {deviation:+.1%}",
                        urgency=urgency
                    )

                    signals.append(signal)

            # Sort by urgency and deviation size
            urgency_priority = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
            signals.sort(key=lambda s: (urgency_priority[s.urgency], abs(s.size_adjustment)), reverse=True)

            return signals

        except Exception as e:
            logger.error(f"Rebalance signal generation error: {e}")
            return []

    async def execute_rebalancing(self, signals: List[RebalanceSignal] = None) -> Dict[str, Any]:
        """
        Execute portfolio rebalancing based on signals
        基於信號執行投資組合再平衡

        Args:
            signals: Rebalance signals to execute

        Returns:
            Dict with rebalancing results
        """
        try:
            if signals is None:
                signals = await self.generate_rebalance_signals()

            if not signals:
                logger.info("📊 No rebalancing needed - portfolio is within target allocation")
                return {'rebalanced': False, 'reason': 'No signals generated'}

            logger.info(f"🔄 Executing portfolio rebalancing with {len(signals)} signals...")

            executed_trades = []
            total_adjustments = 0.0

            for signal in signals:
                # Calculate position size adjustment
                current_value = self.portfolio_value * signal.current_allocation
                target_value = self.portfolio_value * signal.target_allocation
                value_adjustment = target_value - current_value

                # Convert to position size (assuming $100,000 per lot)
                lot_adjustment = abs(value_adjustment) / 100000

                if lot_adjustment >= 0.01:  # Minimum 0.01 lots
                    trade_info = {
                        'currency_pair': signal.currency_pair,
                        'action': signal.recommended_action,
                        'size': round(lot_adjustment, 2),
                        'value_adjustment': value_adjustment,
                        'reason': signal.reason,
                        'urgency': signal.urgency,
                        'timestamp': datetime.now().isoformat()
                    }

                    executed_trades.append(trade_info)
                    total_adjustments += abs(value_adjustment)

                    logger.info(f"   🎯 {signal.currency_pair}: {signal.recommended_action} "
                              f"{lot_adjustment:.2f} lots (${value_adjustment:+,.2f})")

            # Update allocation weights (simulated execution)
            for trade in executed_trades:
                pair = trade['currency_pair']
                if pair in self.currency_pairs:
                    # Update current weight towards target
                    current_weight = self.currency_pairs[pair].current_weight
                    target_weight = self.currency_pairs[pair].target_weight
                    adjustment_factor = 0.8  # 80% of the way to target

                    new_weight = current_weight + (target_weight - current_weight) * adjustment_factor
                    self.currency_pairs[pair].current_weight = new_weight
                    self.currency_pairs[pair].last_rebalance = datetime.now()

            # Record rebalancing event
            rebalance_event = {
                'timestamp': datetime.now().isoformat(),
                'strategy': self.allocation_strategy.value,
                'signals_processed': len(signals),
                'trades_executed': len(executed_trades),
                'total_value_adjusted': total_adjustments,
                'executed_trades': executed_trades
            }

            self.rebalance_history.append(rebalance_event)

            # Keep only last 100 rebalance events
            if len(self.rebalance_history) > 100:
                self.rebalance_history = self.rebalance_history[-100:]

            logger.info(f"✅ Rebalancing completed: {len(executed_trades)} trades, "
                       f"${total_adjustments:,.2f} total adjustment")

            return {
                'rebalanced': True,
                'signals_processed': len(signals),
                'trades_executed': len(executed_trades),
                'total_value_adjusted': total_adjustments,
                'executed_trades': executed_trades,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Rebalancing execution error: {e}")
            return {'rebalanced': False, 'error': str(e)}

    async def _update_correlation_matrix(self):
        """Update correlation matrix between currency pairs | 更新貨幣對之間的相關性矩陣"""
        try:
            pairs = list(self.currency_pairs.keys())
            n = len(pairs)
            correlation_matrix = np.eye(n)

            # Calculate correlations based on returns history
            for i in range(n):
                for j in range(i + 1, n):
                    pair1, pair2 = pairs[i], pairs[j]
                    correlation = await self._calculate_pair_correlation(pair1, pair2)
                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation  # Symmetric matrix

            self.correlation_matrix = correlation_matrix

            # Update correlation scores for each currency
            for i, (pair, allocation) in enumerate(self.currency_pairs.items()):
                avg_correlation = np.mean(np.abs(correlation_matrix[i]))
                allocation.correlation_score = avg_correlation

        except Exception as e:
            logger.error(f"Correlation matrix update error: {e}")

    async def _calculate_pair_correlation(self, pair1: str, pair2: str, lookback_days: int = 30) -> float:
        """Calculate correlation between two currency pairs | 計算兩個貨幣對之間的相關性"""
        try:
            # Simplified correlation based on currency sharing
            currencies1 = set(pair1.split('/'))
            currencies2 = set(pair2.split('/'))

            shared_currencies = currencies1.intersection(currencies2)

            if len(shared_currencies) == 2:  # Same pair
                return 1.0
            elif len(shared_currencies) == 1:  # One shared currency
                return 0.6  # Moderate correlation
            else:  # No shared currencies
                # Use known correlations for major pairs
                known_correlations = {
                    ('EUR/USD', 'GBP/USD'): 0.7,
                    ('USD/JPY', 'EUR/JPY'): 0.8,
                    ('AUD/USD', 'EUR/USD'): 0.6,
                    ('USD/CHF', 'EUR/USD'): -0.8
                }

                pair_tuple = tuple(sorted([pair1, pair2]))
                return known_correlations.get(pair_tuple, 0.2)  # Default low correlation

        except Exception as e:
            logger.error(f"Pair correlation calculation error: {e}")
            return 0.0

    async def _calculate_diversification_metrics(self) -> Dict[str, float]:
        """Calculate portfolio diversification metrics | 計算投資組合多元化指標"""
        try:
            # Current weights
            weights = np.array([allocation.current_weight for allocation in self.currency_pairs.values()])

            # Effective number of assets (inverse of Herfindahl index)
            herfindahl_index = np.sum(weights ** 2)
            effective_assets = 1.0 / herfindahl_index if herfindahl_index > 0 else 1.0

            # Diversification ratio
            portfolio_volatility = await self._calculate_portfolio_volatility(weights)
            individual_volatilities = np.array([allocation.volatility for allocation in self.currency_pairs.values()])
            weighted_avg_volatility = np.sum(weights * individual_volatilities)

            diversification_ratio = weighted_avg_volatility / max(portfolio_volatility, 0.001)

            # Concentration index (max weight)
            concentration_index = np.max(weights)

            return {
                'effective_number_of_assets': float(effective_assets),
                'diversification_ratio': float(diversification_ratio),
                'concentration_index': float(concentration_index),
                'herfindahl_index': float(herfindahl_index),
                'portfolio_volatility': float(portfolio_volatility),
                'weighted_avg_volatility': float(weighted_avg_volatility)
            }

        except Exception as e:
            logger.error(f"Diversification metrics calculation error: {e}")
            return {}

    async def _calculate_portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility using correlation matrix | 使用相關性矩陣計算投資組合波動率"""
        try:
            volatilities = np.array([allocation.volatility for allocation in self.currency_pairs.values()])

            # Covariance matrix
            cov_matrix = np.outer(volatilities, volatilities) * self.correlation_matrix

            # Portfolio variance
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

            # Portfolio volatility
            portfolio_volatility = np.sqrt(portfolio_variance)

            return float(portfolio_volatility)

        except Exception as e:
            logger.error(f"Portfolio volatility calculation error: {e}")
            return 0.015  # Default 1.5% volatility

    async def _identify_diversification_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities to improve diversification | 識別改善多元化的機會"""
        try:
            opportunities = []

            # Check for high correlations
            high_correlations = np.where(self.correlation_matrix > 0.8)
            pairs = list(self.currency_pairs.keys())

            for i, j in zip(high_correlations[0], high_correlations[1]):
                if i < j:  # Avoid duplicates
                    pair1, pair2 = pairs[i], pairs[j]
                    correlation = self.correlation_matrix[i, j]

                    opportunities.append({
                        'type': 'HIGH_CORRELATION',
                        'description': f'High correlation between {pair1} and {pair2}',
                        'correlation': float(correlation),
                        'recommendation': f'Consider reducing allocation to one of {pair1} or {pair2}',
                        'priority': 'HIGH' if correlation > 0.9 else 'MEDIUM'
                    })

            # Check for concentration risk
            for pair, allocation in self.currency_pairs.items():
                if allocation.current_weight > 0.4:  # 40% concentration
                    opportunities.append({
                        'type': 'CONCENTRATION_RISK',
                        'description': f'High concentration in {pair}',
                        'current_weight': allocation.current_weight,
                        'recommendation': f'Consider reducing {pair} allocation below 35%',
                        'priority': 'HIGH'
                    })

            # Check for under-diversification
            effective_assets = (await self._calculate_diversification_metrics()).get('effective_number_of_assets', 0)
            if effective_assets < 2.5:  # Low diversification
                opportunities.append({
                    'type': 'UNDER_DIVERSIFICATION',
                    'description': 'Portfolio lacks sufficient diversification',
                    'effective_assets': effective_assets,
                    'recommendation': 'Consider more balanced allocation across currency pairs',
                    'priority': 'MEDIUM'
                })

            return opportunities

        except Exception as e:
            logger.error(f"Diversification opportunities identification error: {e}")
            return []

    async def _generate_rebalance_signals(self) -> List[Dict[str, Any]]:
        """Generate rebalance signals in dictionary format | 以字典格式生成再平衡信號"""
        signals = await self.generate_rebalance_signals()

        return [
            {
                'currency_pair': signal.currency_pair,
                'current_allocation': signal.current_allocation,
                'target_allocation': signal.target_allocation,
                'recommended_action': signal.recommended_action,
                'size_adjustment': signal.size_adjustment,
                'reason': signal.reason,
                'urgency': signal.urgency
            }
            for signal in signals
        ]

    async def _generate_allocation_recommendations(self) -> List[Dict[str, Any]]:
        """Generate allocation recommendations | 生成配置建議"""
        try:
            recommendations = []

            # Analyze each currency pair
            for pair, allocation in self.currency_pairs.items():
                recommendation = {
                    'currency_pair': pair,
                    'current_weight': allocation.current_weight,
                    'target_weight': allocation.target_weight,
                    'recommendations': []
                }

                # Weight recommendations
                deviation = allocation.current_weight - allocation.target_weight
                if abs(deviation) > 0.05:  # 5% threshold
                    if deviation > 0:
                        recommendation['recommendations'].append(
                            f"Consider reducing {pair} allocation by {abs(deviation):.1%}"
                        )
                    else:
                        recommendation['recommendations'].append(
                            f"Consider increasing {pair} allocation by {abs(deviation):.1%}"
                        )

                # Volatility recommendations
                if allocation.volatility > 0.02:  # High volatility
                    recommendation['recommendations'].append(
                        f"{pair} has high volatility ({allocation.volatility:.1%}) - consider smaller allocation"
                    )

                # Correlation recommendations
                if allocation.correlation_score > 0.7:  # High average correlation
                    recommendation['recommendations'].append(
                        f"{pair} has high correlation with other positions - monitor concentration risk"
                    )

                recommendations.append(recommendation)

            return recommendations

        except Exception as e:
            logger.error(f"Allocation recommendations generation error: {e}")
            return []

    async def _calculate_momentum_score(self, pair: str, lookback_days: int = 20) -> float:
        """Calculate momentum score for currency pair | 計算貨幣對的動量評分"""
        try:
            # Simplified momentum calculation
            # In production, this would use actual price data

            # Generate synthetic momentum based on pair characteristics
            momentum_scores = {
                'USD/JPY': 0.3,   # Moderate momentum
                'EUR/USD': 0.1,   # Low momentum
                'GBP/USD': 0.5,   # Higher momentum
                'AUD/USD': 0.4    # Moderate-high momentum
            }

            return momentum_scores.get(pair, 0.2)

        except Exception as e:
            logger.error(f"Momentum score calculation error: {e}")
            return 0.0

    async def _calculate_mean_reversion_score(self, pair: str, lookback_days: int = 20) -> float:
        """Calculate mean reversion score for currency pair | 計算貨幣對的均值回歸評分"""
        try:
            # Simplified mean reversion calculation
            # In production, this would analyze actual price deviations from mean

            # Generate synthetic mean reversion scores
            reversion_scores = {
                'USD/JPY': 0.4,   # Moderate mean reversion
                'EUR/USD': 0.6,   # Higher mean reversion
                'GBP/USD': 0.2,   # Lower mean reversion
                'AUD/USD': 0.3    # Moderate mean reversion
            }

            return reversion_scores.get(pair, 0.3)

        except Exception as e:
            logger.error(f"Mean reversion score calculation error: {e}")
            return 0.0

    async def _calculate_market_volatility(self) -> float:
        """Calculate overall market volatility | 計算整體市場波動率"""
        try:
            volatilities = [allocation.volatility for allocation in self.currency_pairs.values()]
            return np.mean(volatilities)

        except Exception as e:
            logger.error(f"Market volatility calculation error: {e}")
            return 0.015

    async def _calculate_momentum_strength(self) -> float:
        """Calculate overall momentum strength | 計算整體動量強度"""
        try:
            momentum_scores = []
            for pair in self.currency_pairs.keys():
                momentum = await self._calculate_momentum_score(pair)
                momentum_scores.append(momentum)

            return np.mean(momentum_scores)

        except Exception as e:
            logger.error(f"Momentum strength calculation error: {e}")
            return 0.3

    async def _update_portfolio_metrics(self):
        """Update portfolio performance metrics | 更新投資組合績效指標"""
        try:
            # Calculate current portfolio metrics
            weights = np.array([allocation.current_weight for allocation in self.currency_pairs.values()])
            portfolio_volatility = await self._calculate_portfolio_volatility(weights)
            diversification_metrics = await self._calculate_diversification_metrics()

            # Create metrics object
            metrics = PortfolioMetrics(
                timestamp=datetime.now(),
                total_value=self.portfolio_value,
                volatility=portfolio_volatility,
                effective_number_of_assets=diversification_metrics.get('effective_number_of_assets', 0),
                diversification_ratio=diversification_metrics.get('diversification_ratio', 0),
                concentration_index=diversification_metrics.get('concentration_index', 0),
                correlation_average=float(np.mean(np.abs(self.correlation_matrix[np.triu_indices(len(self.correlation_matrix), k=1)])))
            )

            # Add to history
            self.portfolio_metrics_history.append(metrics)

            # Keep only last 1000 metrics
            if len(self.portfolio_metrics_history) > 1000:
                self.portfolio_metrics_history = self.portfolio_metrics_history[-1000:]

        except Exception as e:
            logger.error(f"Portfolio metrics update error: {e}")

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary | 獲取全面的投資組合摘要"""
        try:
            current_allocation = {
                pair: allocation.current_weight
                for pair, allocation in self.currency_pairs.items()
            }

            target_allocation = {
                pair: allocation.target_weight
                for pair, allocation in self.currency_pairs.items()
            }

            latest_metrics = self.portfolio_metrics_history[-1] if self.portfolio_metrics_history else None

            return {
                'timestamp': datetime.now().isoformat(),
                'portfolio_value': self.portfolio_value,
                'allocation_strategy': self.allocation_strategy.value,
                'rebalance_frequency': self.rebalance_frequency.value,
                'current_allocation': current_allocation,
                'target_allocation': target_allocation,
                'allocation_deviations': {
                    pair: allocation.current_weight - allocation.target_weight
                    for pair, allocation in self.currency_pairs.items()
                },
                'diversification_metrics': {
                    'effective_assets': latest_metrics.effective_number_of_assets if latest_metrics else 0,
                    'diversification_ratio': latest_metrics.diversification_ratio if latest_metrics else 0,
                    'concentration_index': latest_metrics.concentration_index if latest_metrics else 0,
                    'correlation_average': latest_metrics.correlation_average if latest_metrics else 0
                },
                'portfolio_metrics': {
                    'volatility': latest_metrics.volatility if latest_metrics else 0,
                    'total_value': latest_metrics.total_value if latest_metrics else self.portfolio_value
                },
                'rebalance_history_count': len(self.rebalance_history),
                'last_rebalance': self.rebalance_history[-1]['timestamp'] if self.rebalance_history else None
            }

        except Exception as e:
            logger.error(f"Portfolio summary error: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}


# Factory function
def create_diversification_engine(config: Optional[Dict[str, Any]] = None) -> PortfolioDiversificationEngine:
    """
    Create portfolio diversification engine
    創建投資組合多元化引擎

    Args:
        config: Configuration dictionary

    Returns:
        PortfolioDiversificationEngine: Configured engine instance
    """
    return PortfolioDiversificationEngine(config)


# Example usage
async def main():
    """Example usage of Portfolio Diversification Engine | 投資組合多元化引擎使用示例"""

    engine = create_diversification_engine()

    try:
        # Initialize portfolio
        await engine.initialize_portfolio()

        # Analyze diversification opportunities
        analysis = await engine.analyze_diversification_opportunities()
        print("📊 Diversification Analysis:")
        print(f"   Average Correlation: {analysis['correlation_analysis']['average_correlation']:.2f}")
        print(f"   Optimization Opportunities: {len(analysis['optimization_opportunities'])}")

        # Optimize allocation using different strategies
        print("\n🎯 Testing Allocation Strategies:")

        strategies = [
            AllocationStrategy.EQUAL_WEIGHT,
            AllocationStrategy.RISK_PARITY,
            AllocationStrategy.MOMENTUM,
            AllocationStrategy.DYNAMIC_ADAPTIVE
        ]

        for strategy in strategies:
            allocation = await engine.optimize_allocation(strategy)
            print(f"\n{strategy.value}:")
            for pair, weight in allocation.items():
                print(f"   {pair}: {weight:.1%}")

        # Generate and execute rebalancing signals
        print("\n🔄 Rebalancing Analysis:")
        signals = await engine.generate_rebalance_signals()
        print(f"   Rebalance signals: {len(signals)}")

        for signal in signals[:3]:  # Show first 3 signals
            print(f"   {signal.currency_pair}: {signal.recommended_action} "
                  f"({signal.current_allocation:.1%} → {signal.target_allocation:.1%})")

        # Get portfolio summary
        summary = engine.get_portfolio_summary()
        print(f"\n📈 Portfolio Summary:")
        print(f"   Strategy: {summary['allocation_strategy']}")
        print(f"   Effective Assets: {summary['diversification_metrics']['effective_assets']:.1f}")
        print(f"   Diversification Ratio: {summary['diversification_metrics']['diversification_ratio']:.2f}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())