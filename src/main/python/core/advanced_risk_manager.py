#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Risk Management System | 高級風險管理系統
=====================================================

Enhanced risk management with portfolio-level controls, correlation analysis,
dynamic position sizing, and advanced stop loss algorithms.

具有投資組合級控制、相關性分析、動態倉位大小和高級止損算法的增強風險管理。

Features | 功能:
- Multi-currency portfolio risk analysis
- Dynamic correlation-based position sizing
- Advanced stop loss algorithms (ATR, volatility-based)
- Real-time drawdown monitoring
- Currency exposure limits
- Emergency risk controls

Author: AIFX Development Team
Date: 2025-01-15
Version: 2.0.0 - Live Trading Enhancement
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)

class RiskLevel(Enum):
    """Risk level enumeration | 風險級別枚舉"""
    CONSERVATIVE = "CONSERVATIVE"
    MODERATE = "MODERATE"
    AGGRESSIVE = "AGGRESSIVE"
    EMERGENCY = "EMERGENCY"

class StopLossType(Enum):
    """Stop loss type enumeration | 止損類型枚舉"""
    FIXED_PERCENT = "FIXED_PERCENT"
    ATR_BASED = "ATR_BASED"
    VOLATILITY_BASED = "VOLATILITY_BASED"
    TRAILING = "TRAILING"
    TIME_BASED = "TIME_BASED"

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics | 綜合風險指標"""
    timestamp: datetime = field(default_factory=datetime.now)
    portfolio_value: float = 0.0
    total_exposure: float = 0.0
    available_margin: float = 0.0
    margin_utilization: float = 0.0

    # Portfolio risk metrics
    portfolio_var_95: float = 0.0  # 95% Value at Risk
    portfolio_var_99: float = 0.0  # 99% Value at Risk
    expected_shortfall: float = 0.0
    maximum_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0

    # Currency exposure
    currency_exposures: Dict[str, float] = field(default_factory=dict)

    # Position metrics
    total_positions: int = 0
    winning_positions: int = 0
    losing_positions: int = 0
    win_rate: float = 0.0

    # Risk limits
    risk_limits_breached: List[str] = field(default_factory=list)

    # Correlation metrics
    average_correlation: float = 0.0
    max_correlation: float = 0.0
    correlation_risk_score: float = 0.0

@dataclass
class PositionRisk:
    """Individual position risk analysis | 個別倉位風險分析"""
    position_id: str
    symbol: str
    direction: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float

    # Risk metrics
    position_var: float = 0.0
    correlation_exposure: float = 0.0
    time_decay_risk: float = 0.0
    volatility_risk: float = 0.0

    # Stop loss levels
    current_stop_loss: float = 0.0
    suggested_stop_loss: float = 0.0
    trailing_stop_distance: float = 0.0

    # Risk scores (0-100)
    overall_risk_score: float = 0.0
    liquidity_risk_score: float = 0.0
    correlation_risk_score: float = 0.0
    volatility_risk_score: float = 0.0

@dataclass
class RiskLimits:
    """Risk management limits configuration | 風險管理限制配置"""
    # Portfolio limits
    max_portfolio_risk: float = 0.10  # 10% of portfolio
    max_daily_loss: float = 0.05      # 5% daily loss limit
    max_drawdown: float = 0.15        # 15% maximum drawdown

    # Position limits
    max_position_size: float = 0.05   # 5% per position
    max_currency_exposure: float = 0.20  # 20% per currency
    max_correlations: float = 0.70    # 70% correlation limit

    # Stop loss limits
    min_stop_distance: float = 0.005  # 0.5% minimum stop distance
    max_stop_distance: float = 0.030  # 3% maximum stop distance

    # Time limits
    max_position_duration: int = 24   # 24 hours maximum

    # Emergency limits
    emergency_stop_loss: float = 0.08  # 8% emergency stop
    margin_call_level: float = 0.80    # 80% margin utilization

class AdvancedRiskManager:
    """
    Advanced Risk Management System | 高級風險管理系統

    Comprehensive risk management system with portfolio-level controls,
    dynamic position sizing, and sophisticated stop loss algorithms.

    具有投資組合級控制、動態倉位大小和復雜止損算法的綜合風險管理系統。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize advanced risk manager
        初始化高級風險管理器

        Args:
            config: Risk management configuration
        """
        self.config = config or {}
        self.risk_level = RiskLevel.MODERATE
        self.risk_limits = RiskLimits()

        # Portfolio state
        self.portfolio_value = 10000.0
        self.available_margin = 10000.0
        self.positions: Dict[str, PositionRisk] = {}

        # Risk tracking
        self.daily_pnl = 0.0
        self.peak_portfolio_value = 10000.0
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0

        # Price history for calculations
        self.price_history: Dict[str, List[Dict]] = {}
        self.correlation_matrix = np.eye(4)  # For 4 major pairs
        self.volatility_cache: Dict[str, float] = {}

        # Risk alerts
        self.active_alerts: List[Dict] = []
        self.emergency_mode = False

        # Performance tracking
        self.risk_metrics_history: List[RiskMetrics] = []

        logger.info("Advanced Risk Manager initialized | 高級風險管理器已初始化")

    def set_risk_level(self, risk_level: RiskLevel):
        """Set overall risk level | 設置總體風險級別"""
        self.risk_level = risk_level

        # Adjust limits based on risk level
        if risk_level == RiskLevel.CONSERVATIVE:
            self.risk_limits.max_portfolio_risk = 0.05  # 5%
            self.risk_limits.max_position_size = 0.02   # 2%
            self.risk_limits.max_daily_loss = 0.02      # 2%
        elif risk_level == RiskLevel.MODERATE:
            self.risk_limits.max_portfolio_risk = 0.10  # 10%
            self.risk_limits.max_position_size = 0.05   # 5%
            self.risk_limits.max_daily_loss = 0.05      # 5%
        elif risk_level == RiskLevel.AGGRESSIVE:
            self.risk_limits.max_portfolio_risk = 0.20  # 20%
            self.risk_limits.max_position_size = 0.10   # 10%
            self.risk_limits.max_daily_loss = 0.08      # 8%
        elif risk_level == RiskLevel.EMERGENCY:
            self.risk_limits.max_portfolio_risk = 0.02  # 2%
            self.risk_limits.max_position_size = 0.01   # 1%
            self.risk_limits.max_daily_loss = 0.01      # 1%

        logger.info(f"Risk level set to: {risk_level.value}")

    async def validate_new_position(self,
                                  symbol: str,
                                  direction: str,
                                  size: float,
                                  entry_price: float,
                                  confidence: float = 0.5) -> Tuple[bool, str, float]:
        """
        Validate new position against risk limits | 根據風險限制驗證新倉位

        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            size: Position size
            entry_price: Entry price
            confidence: Signal confidence (0-1)

        Returns:
            Tuple[bool, str, float]: (approved, reason, suggested_size)
        """
        try:
            logger.info(f"🔍 Validating new position: {symbol} {direction} {size}")

            # Check emergency mode
            if self.emergency_mode:
                return False, "Emergency mode active - no new positions allowed", 0.0

            # Check daily loss limit
            if self.daily_pnl < -self.risk_limits.max_daily_loss * self.portfolio_value:
                return False, f"Daily loss limit exceeded: {self.daily_pnl:.2f}", 0.0

            # Check maximum drawdown
            if self.current_drawdown > self.risk_limits.max_drawdown:
                return False, f"Maximum drawdown exceeded: {self.current_drawdown:.2%}", 0.0

            # Check margin utilization
            margin_usage = (self.portfolio_value - self.available_margin) / self.portfolio_value
            if margin_usage > self.risk_limits.margin_call_level:
                return False, f"Margin utilization too high: {margin_usage:.2%}", 0.0

            # Calculate position value
            position_value = size * entry_price * 100000  # Standard lot
            position_risk = position_value / self.portfolio_value

            # Check position size limit
            if position_risk > self.risk_limits.max_position_size:
                suggested_size = (self.risk_limits.max_position_size * self.portfolio_value) / (entry_price * 100000)
                return False, f"Position too large: {position_risk:.2%} > {self.risk_limits.max_position_size:.2%}", suggested_size

            # Check currency exposure
            currency = symbol.split('/')[0] if direction == 'BUY' else symbol.split('/')[1]
            current_exposure = self._calculate_currency_exposure(currency)
            new_exposure = (current_exposure + position_value) / self.portfolio_value

            if new_exposure > self.risk_limits.max_currency_exposure:
                max_additional = self.risk_limits.max_currency_exposure * self.portfolio_value - current_exposure
                suggested_size = max_additional / (entry_price * 100000)
                return False, f"Currency exposure limit: {new_exposure:.2%}", max(0, suggested_size)

            # Check correlation risk
            correlation_risk = await self._calculate_correlation_risk(symbol, direction, size)
            if correlation_risk > self.risk_limits.max_correlations:
                suggested_size = size * 0.5  # Reduce size by 50%
                return False, f"High correlation risk: {correlation_risk:.2f}", suggested_size

            # Check portfolio heat (number of positions)
            portfolio_heat = len(self.positions) / 10.0  # Assume max 10 positions
            if portfolio_heat > 0.8:  # 80% portfolio heat
                return False, "Too many open positions", 0.0

            # Adjust size based on confidence
            confidence_adjusted_size = size * (0.5 + confidence * 0.5)

            # Adjust size based on volatility
            volatility = await self._get_symbol_volatility(symbol)
            volatility_adjustment = max(0.5, min(1.5, 1.0 / (volatility * 100)))
            final_size = confidence_adjusted_size * volatility_adjustment

            # Ensure minimum position size
            if final_size < 0.01:  # Minimum 0.01 lots
                return False, "Calculated position size too small", 0.0

            logger.info(f"✅ Position validated: Original={size:.2f}, Final={final_size:.2f}")
            return True, "Position approved", round(final_size, 2)

        except Exception as e:
            logger.error(f"Position validation error: {e}")
            return False, f"Validation error: {str(e)}", 0.0

    async def calculate_dynamic_stop_loss(self,
                                        symbol: str,
                                        direction: str,
                                        entry_price: float,
                                        position_size: float,
                                        stop_type: StopLossType = StopLossType.ATR_BASED) -> float:
        """
        Calculate dynamic stop loss based on multiple factors | 基於多種因素計算動態止損

        Args:
            symbol: Trading symbol
            direction: BUY or SELL
            entry_price: Entry price
            position_size: Position size
            stop_type: Type of stop loss calculation

        Returns:
            float: Calculated stop loss level
        """
        try:
            logger.info(f"📊 Calculating {stop_type.value} stop loss for {symbol}")

            if stop_type == StopLossType.FIXED_PERCENT:
                # Fixed percentage stop loss
                percent = 0.02  # 2%
                if direction == 'BUY':
                    return entry_price * (1 - percent)
                else:
                    return entry_price * (1 + percent)

            elif stop_type == StopLossType.ATR_BASED:
                # ATR-based stop loss
                atr = await self._calculate_atr(symbol)
                multiplier = 2.0  # 2x ATR

                if direction == 'BUY':
                    return entry_price - (atr * multiplier)
                else:
                    return entry_price + (atr * multiplier)

            elif stop_type == StopLossType.VOLATILITY_BASED:
                # Volatility-based stop loss
                volatility = await self._get_symbol_volatility(symbol)
                volatility_multiplier = max(1.5, min(3.0, volatility * 150))

                distance = entry_price * 0.01 * volatility_multiplier  # 1% base * volatility

                if direction == 'BUY':
                    return entry_price - distance
                else:
                    return entry_price + distance

            elif stop_type == StopLossType.TRAILING:
                # Trailing stop loss
                atr = await self._calculate_atr(symbol)
                trailing_distance = atr * 1.5  # 1.5x ATR trailing

                if direction == 'BUY':
                    return entry_price - trailing_distance
                else:
                    return entry_price + trailing_distance

            elif stop_type == StopLossType.TIME_BASED:
                # Time-based stop loss (tighter stops for longer-held positions)
                base_percent = 0.015  # 1.5% base
                if direction == 'BUY':
                    return entry_price * (1 - base_percent)
                else:
                    return entry_price * (1 + base_percent)

            else:
                # Default to fixed percentage
                return await self.calculate_dynamic_stop_loss(symbol, direction, entry_price, position_size, StopLossType.FIXED_PERCENT)

        except Exception as e:
            logger.error(f"Stop loss calculation error: {e}")
            # Fallback to 2% fixed stop
            if direction == 'BUY':
                return entry_price * 0.98
            else:
                return entry_price * 1.02

    async def update_trailing_stops(self):
        """Update trailing stop losses for all positions | 更新所有倉位的移動止損"""
        try:
            for position_id, position in self.positions.items():
                if position.trailing_stop_distance > 0:
                    new_stop = await self._calculate_trailing_stop(position)

                    if new_stop != position.current_stop_loss:
                        logger.info(f"📈 Updating trailing stop for {position.symbol}: "
                                  f"{position.current_stop_loss:.5f} → {new_stop:.5f}")

                        # Update position stop loss
                        position.current_stop_loss = new_stop
                        position.suggested_stop_loss = new_stop

        except Exception as e:
            logger.error(f"Trailing stop update error: {e}")

    async def _calculate_trailing_stop(self, position: PositionRisk) -> float:
        """Calculate trailing stop for a position | 計算倉位的移動止損"""
        try:
            if position.direction == 'BUY':
                # For long positions, trail stop up as price increases
                potential_stop = position.current_price - position.trailing_stop_distance
                return max(position.current_stop_loss, potential_stop)
            else:
                # For short positions, trail stop down as price decreases
                potential_stop = position.current_price + position.trailing_stop_distance
                return min(position.current_stop_loss, potential_stop)

        except Exception as e:
            logger.error(f"Trailing stop calculation error: {e}")
            return position.current_stop_loss

    async def analyze_portfolio_risk(self) -> RiskMetrics:
        """
        Comprehensive portfolio risk analysis | 全面的投資組合風險分析

        Returns:
            RiskMetrics: Complete risk analysis
        """
        try:
            # Calculate basic metrics
            total_positions = len(self.positions)
            total_exposure = sum(abs(pos.size * pos.current_price * 100000) for pos in self.positions.values())
            margin_utilization = (self.portfolio_value - self.available_margin) / self.portfolio_value

            # Calculate PnL metrics
            total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            winning_positions = sum(1 for pos in self.positions.values() if pos.unrealized_pnl > 0)
            losing_positions = total_positions - winning_positions
            win_rate = (winning_positions / max(1, total_positions)) * 100

            # Calculate VaR (Value at Risk)
            pnl_values = [pos.unrealized_pnl for pos in self.positions.values()]
            var_95 = np.percentile(pnl_values, 5) if pnl_values else 0.0
            var_99 = np.percentile(pnl_values, 1) if pnl_values else 0.0

            # Calculate Expected Shortfall (average of worst 5% outcomes)
            worst_5_percent = [pnl for pnl in pnl_values if pnl <= var_95]
            expected_shortfall = np.mean(worst_5_percent) if worst_5_percent else 0.0

            # Calculate Sharpe ratio
            if len(self.risk_metrics_history) >= 30:
                returns = [metrics.portfolio_value / 10000.0 - 1 for metrics in self.risk_metrics_history[-30:]]
                if np.std(returns) > 0:
                    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
                else:
                    sharpe_ratio = 0.0
            else:
                sharpe_ratio = 0.0

            # Calculate Sortino ratio (using downside deviation)
            if len(self.risk_metrics_history) >= 30:
                returns = [metrics.portfolio_value / 10000.0 - 1 for metrics in self.risk_metrics_history[-30:]]
                downside_returns = [r for r in returns if r < 0]
                if downside_returns and np.std(downside_returns) > 0:
                    sortino_ratio = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
                else:
                    sortino_ratio = 0.0
            else:
                sortino_ratio = 0.0

            # Calculate currency exposures
            currency_exposures = {}
            for position in self.positions.values():
                currencies = position.symbol.split('/')
                base_currency, quote_currency = currencies[0], currencies[1]

                exposure = position.size * position.current_price * 100000

                if position.direction == 'BUY':
                    currency_exposures[base_currency] = currency_exposures.get(base_currency, 0) + exposure
                    currency_exposures[quote_currency] = currency_exposures.get(quote_currency, 0) - exposure
                else:
                    currency_exposures[base_currency] = currency_exposures.get(base_currency, 0) - exposure
                    currency_exposures[quote_currency] = currency_exposures.get(quote_currency, 0) + exposure

            # Calculate correlation metrics
            correlations = []
            positions_list = list(self.positions.values())
            for i in range(len(positions_list)):
                for j in range(i + 1, len(positions_list)):
                    correlation = await self._calculate_position_correlation(positions_list[i], positions_list[j])
                    correlations.append(correlation)

            avg_correlation = np.mean(correlations) if correlations else 0.0
            max_correlation = max(correlations) if correlations else 0.0
            correlation_risk_score = max_correlation * 100  # Convert to 0-100 score

            # Check for breached risk limits
            risk_limits_breached = []

            if margin_utilization > self.risk_limits.margin_call_level:
                risk_limits_breached.append(f"MARGIN_UTILIZATION_{margin_utilization:.2%}")

            if abs(self.daily_pnl) > self.risk_limits.max_daily_loss * self.portfolio_value:
                risk_limits_breached.append(f"DAILY_LOSS_{self.daily_pnl:.2f}")

            if self.current_drawdown > self.risk_limits.max_drawdown:
                risk_limits_breached.append(f"MAX_DRAWDOWN_{self.current_drawdown:.2%}")

            for currency, exposure in currency_exposures.items():
                exposure_percent = abs(exposure) / self.portfolio_value
                if exposure_percent > self.risk_limits.max_currency_exposure:
                    risk_limits_breached.append(f"CURRENCY_EXPOSURE_{currency}_{exposure_percent:.2%}")

            # Create risk metrics object
            risk_metrics = RiskMetrics(
                timestamp=datetime.now(),
                portfolio_value=self.portfolio_value + total_pnl,
                total_exposure=total_exposure,
                available_margin=self.available_margin,
                margin_utilization=margin_utilization,
                portfolio_var_95=var_95,
                portfolio_var_99=var_99,
                expected_shortfall=expected_shortfall,
                maximum_drawdown=self.max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                currency_exposures=currency_exposures,
                total_positions=total_positions,
                winning_positions=winning_positions,
                losing_positions=losing_positions,
                win_rate=win_rate,
                risk_limits_breached=risk_limits_breached,
                average_correlation=avg_correlation,
                max_correlation=max_correlation,
                correlation_risk_score=correlation_risk_score
            )

            # Store in history
            self.risk_metrics_history.append(risk_metrics)

            # Keep only last 1000 metrics
            if len(self.risk_metrics_history) > 1000:
                self.risk_metrics_history = self.risk_metrics_history[-1000:]

            # Trigger alerts if necessary
            await self._check_risk_alerts(risk_metrics)

            return risk_metrics

        except Exception as e:
            logger.error(f"Portfolio risk analysis error: {e}")
            return RiskMetrics()

    async def _check_risk_alerts(self, risk_metrics: RiskMetrics):
        """Check for risk alerts and trigger actions | 檢查風險警報並觸發行動"""
        try:
            # High-priority alerts
            if risk_metrics.margin_utilization > 0.90:  # 90% margin utilization
                await self._trigger_alert("CRITICAL_MARGIN_UTILIZATION", risk_metrics.margin_utilization)

            if risk_metrics.maximum_drawdown > 0.12:  # 12% drawdown
                await self._trigger_alert("HIGH_DRAWDOWN", risk_metrics.maximum_drawdown)

            if len(risk_metrics.risk_limits_breached) > 0:
                await self._trigger_alert("RISK_LIMITS_BREACHED", risk_metrics.risk_limits_breached)

            if risk_metrics.correlation_risk_score > 80:  # High correlation
                await self._trigger_alert("HIGH_CORRELATION_RISK", risk_metrics.correlation_risk_score)

            # Medium-priority alerts
            if risk_metrics.win_rate < 30 and risk_metrics.total_positions > 5:  # Low win rate
                await self._trigger_alert("LOW_WIN_RATE", risk_metrics.win_rate)

            if abs(risk_metrics.expected_shortfall) > 1000:  # Large expected shortfall
                await self._trigger_alert("HIGH_EXPECTED_SHORTFALL", risk_metrics.expected_shortfall)

        except Exception as e:
            logger.error(f"Risk alert check error: {e}")

    async def _trigger_alert(self, alert_type: str, alert_data: Any):
        """Trigger risk management alert | 觸發風險管理警報"""
        try:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': alert_type,
                'data': alert_data,
                'severity': 'HIGH' if 'CRITICAL' in alert_type else 'MEDIUM'
            }

            self.active_alerts.append(alert)

            # Keep only last 100 alerts
            if len(self.active_alerts) > 100:
                self.active_alerts = self.active_alerts[-100:]

            logger.warning(f"🚨 Risk Alert: {alert_type} - {alert_data}")

            # Trigger emergency mode for critical alerts
            if 'CRITICAL' in alert_type or alert_type == 'RISK_LIMITS_BREACHED':
                await self._activate_emergency_mode()

        except Exception as e:
            logger.error(f"Alert trigger error: {e}")

    async def _activate_emergency_mode(self):
        """Activate emergency risk management mode | 啟動緊急風險管理模式"""
        try:
            if not self.emergency_mode:
                self.emergency_mode = True
                logger.critical("🛑 EMERGENCY MODE ACTIVATED")

                # Switch to emergency risk limits
                self.set_risk_level(RiskLevel.EMERGENCY)

                # Consider closing worst positions
                await self._emergency_position_review()

        except Exception as e:
            logger.error(f"Emergency mode activation error: {e}")

    async def _emergency_position_review(self):
        """Review positions during emergency mode | 在緊急模式下審查倉位"""
        try:
            if not self.positions:
                return

            # Sort positions by risk score (highest first)
            positions_by_risk = sorted(
                self.positions.values(),
                key=lambda p: p.overall_risk_score,
                reverse=True
            )

            # Consider closing highest risk positions
            high_risk_positions = [p for p in positions_by_risk if p.overall_risk_score > 70]

            if high_risk_positions:
                logger.warning(f"🔍 Emergency review: {len(high_risk_positions)} high-risk positions identified")

                for position in high_risk_positions[:3]:  # Review top 3 highest risk
                    logger.warning(f"   High risk: {position.symbol} {position.direction} "
                                 f"Risk Score: {position.overall_risk_score:.1f}")

        except Exception as e:
            logger.error(f"Emergency position review error: {e}")

    def add_position(self, position_data: Dict[str, Any]) -> str:
        """
        Add new position to risk tracking | 將新倉位添加到風險跟踪

        Args:
            position_data: Position information

        Returns:
            str: Position ID
        """
        try:
            position_id = position_data.get('position_id', f"POS_{len(self.positions)}")

            position_risk = PositionRisk(
                position_id=position_id,
                symbol=position_data.get('symbol', ''),
                direction=position_data.get('direction', ''),
                size=position_data.get('size', 0.0),
                entry_price=position_data.get('entry_price', 0.0),
                current_price=position_data.get('current_price', position_data.get('entry_price', 0.0)),
                unrealized_pnl=position_data.get('unrealized_pnl', 0.0)
            )

            # Calculate initial risk metrics
            position_risk.overall_risk_score = await self._calculate_position_risk_score(position_risk)

            self.positions[position_id] = position_risk

            logger.info(f"📊 Position added to risk tracking: {position_id}")

            return position_id

        except Exception as e:
            logger.error(f"Add position error: {e}")
            return ""

    def remove_position(self, position_id: str):
        """Remove position from risk tracking | 從風險跟踪中移除倉位"""
        try:
            if position_id in self.positions:
                position = self.positions[position_id]

                # Update daily P&L
                self.daily_pnl += position.unrealized_pnl

                # Remove position
                del self.positions[position_id]

                logger.info(f"📊 Position removed from risk tracking: {position_id}")

        except Exception as e:
            logger.error(f"Remove position error: {e}")

    async def update_position(self, position_id: str, current_price: float, unrealized_pnl: float):
        """Update position with current market data | 使用當前市場數據更新倉位"""
        try:
            if position_id in self.positions:
                position = self.positions[position_id]
                position.current_price = current_price
                position.unrealized_pnl = unrealized_pnl

                # Recalculate risk metrics
                position.overall_risk_score = await self._calculate_position_risk_score(position)

        except Exception as e:
            logger.error(f"Update position error: {e}")

    # Helper methods for calculations

    async def _calculate_position_risk_score(self, position: PositionRisk) -> float:
        """Calculate overall risk score for position | 計算倉位的整體風險評分"""
        try:
            # Base risk from position size
            position_value = position.size * position.current_price * 100000
            size_risk = (position_value / self.portfolio_value) * 100  # Convert to percentage

            # Volatility risk
            volatility = await self._get_symbol_volatility(position.symbol)
            volatility_risk = min(100, volatility * 1000)  # Scale volatility to 0-100

            # Time decay risk (positions held longer are riskier)
            # This would need position duration calculation in a real implementation
            time_risk = 20  # Placeholder

            # Unrealized P&L risk (large losses increase risk)
            pnl_risk = max(0, min(100, abs(position.unrealized_pnl) / 100))

            # Combine risks (weighted average)
            overall_risk = (
                size_risk * 0.3 +
                volatility_risk * 0.3 +
                time_risk * 0.2 +
                pnl_risk * 0.2
            )

            return min(100, overall_risk)

        except Exception as e:
            logger.error(f"Position risk score calculation error: {e}")
            return 50.0  # Medium risk default

    def _calculate_currency_exposure(self, currency: str) -> float:
        """Calculate current exposure to a currency | 計算對某貨幣的當前敞口"""
        exposure = 0.0

        for position in self.positions.values():
            currencies = position.symbol.split('/')
            base_currency, quote_currency = currencies[0], currencies[1]

            position_value = position.size * position.current_price * 100000

            if currency == base_currency:
                if position.direction == 'BUY':
                    exposure += position_value
                else:
                    exposure -= position_value
            elif currency == quote_currency:
                if position.direction == 'BUY':
                    exposure -= position_value
                else:
                    exposure += position_value

        return abs(exposure)

    async def _calculate_correlation_risk(self, symbol: str, direction: str, size: float) -> float:
        """Calculate correlation risk of new position | 計算新倉位的相關性風險"""
        try:
            if not self.positions:
                return 0.0

            max_correlation = 0.0

            for position in self.positions.values():
                correlation = await self._calculate_symbol_correlation(symbol, position.symbol)

                # Increase correlation if same direction
                if direction == position.direction:
                    correlation *= 1.2  # 20% increase for same direction

                max_correlation = max(max_correlation, abs(correlation))

            return max_correlation

        except Exception as e:
            logger.error(f"Correlation risk calculation error: {e}")
            return 0.0

    async def _calculate_position_correlation(self, pos1: PositionRisk, pos2: PositionRisk) -> float:
        """Calculate correlation between two positions | 計算兩個倉位之間的相關性"""
        try:
            base_correlation = await self._calculate_symbol_correlation(pos1.symbol, pos2.symbol)

            # Adjust for direction
            if pos1.direction == pos2.direction:
                return base_correlation
            else:
                return -base_correlation

        except Exception as e:
            logger.error(f"Position correlation calculation error: {e}")
            return 0.0

    async def _calculate_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two currency pairs | 計算兩個貨幣對之間的相關性"""
        try:
            # Simplified correlation based on shared currencies
            currencies1 = set(symbol1.split('/'))
            currencies2 = set(symbol2.split('/'))

            shared_currencies = currencies1.intersection(currencies2)

            if len(shared_currencies) == 2:  # Same pair
                return 1.0
            elif len(shared_currencies) == 1:  # One shared currency
                return 0.6
            else:  # No shared currencies
                # Use known correlations for major pairs
                correlations = {
                    ('EUR/USD', 'GBP/USD'): 0.7,
                    ('USD/JPY', 'EUR/JPY'): 0.8,
                    ('AUD/USD', 'NZD/USD'): 0.9,
                    ('USD/CHF', 'EUR/USD'): -0.8
                }

                pair = tuple(sorted([symbol1, symbol2]))
                return correlations.get(pair, 0.2)  # Default low correlation

        except Exception as e:
            logger.error(f"Symbol correlation calculation error: {e}")
            return 0.0

    async def _get_symbol_volatility(self, symbol: str) -> float:
        """Get volatility for symbol | 獲取品種的波動率"""
        try:
            if symbol in self.volatility_cache:
                return self.volatility_cache[symbol]

            # Default volatilities for major pairs
            default_volatilities = {
                'EUR/USD': 0.012,
                'GBP/USD': 0.015,
                'USD/JPY': 0.013,
                'AUD/USD': 0.016,
                'USD/CHF': 0.011,
                'EUR/JPY': 0.014,
                'GBP/JPY': 0.018
            }

            volatility = default_volatilities.get(symbol, 0.015)
            self.volatility_cache[symbol] = volatility

            return volatility

        except Exception as e:
            logger.error(f"Volatility calculation error: {e}")
            return 0.015

    async def _calculate_atr(self, symbol: str, period: int = 14) -> float:
        """Calculate Average True Range | 計算平均真實範圍"""
        try:
            # Simplified ATR calculation
            # In a real implementation, this would use price history
            base_atr = await self._get_symbol_volatility(symbol)
            return base_atr * 0.8  # Convert volatility to ATR approximation

        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return 0.001  # 10 pips default

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary | 獲取全面的風險總結"""
        try:
            if not self.risk_metrics_history:
                return {'error': 'No risk metrics available'}

            latest_metrics = self.risk_metrics_history[-1]

            return {
                'timestamp': latest_metrics.timestamp.isoformat(),
                'risk_level': self.risk_level.value,
                'emergency_mode': self.emergency_mode,
                'portfolio_summary': {
                    'value': latest_metrics.portfolio_value,
                    'total_exposure': latest_metrics.total_exposure,
                    'margin_utilization': f"{latest_metrics.margin_utilization:.2%}",
                    'daily_pnl': self.daily_pnl,
                    'max_drawdown': f"{latest_metrics.maximum_drawdown:.2%}"
                },
                'position_summary': {
                    'total_positions': latest_metrics.total_positions,
                    'winning_positions': latest_metrics.winning_positions,
                    'win_rate': f"{latest_metrics.win_rate:.1f}%"
                },
                'risk_metrics': {
                    'var_95': latest_metrics.portfolio_var_95,
                    'expected_shortfall': latest_metrics.expected_shortfall,
                    'sharpe_ratio': f"{latest_metrics.sharpe_ratio:.2f}",
                    'correlation_risk': f"{latest_metrics.correlation_risk_score:.1f}"
                },
                'currency_exposures': {
                    currency: f"${exposure:,.2f}"
                    for currency, exposure in latest_metrics.currency_exposures.items()
                },
                'active_alerts': len(self.active_alerts),
                'risk_limits_breached': latest_metrics.risk_limits_breached
            }

        except Exception as e:
            logger.error(f"Risk summary error: {e}")
            return {'error': str(e)}


# Factory function
def create_risk_manager(config: Optional[Dict[str, Any]] = None) -> AdvancedRiskManager:
    """
    Create advanced risk manager instance
    創建高級風險管理器實例

    Args:
        config: Risk management configuration

    Returns:
        AdvancedRiskManager: Configured risk manager
    """
    return AdvancedRiskManager(config)


# Example usage
async def main():
    """Example usage of Advanced Risk Manager | 高級風險管理器使用示例"""

    risk_manager = create_risk_manager()

    try:
        # Set risk level
        risk_manager.set_risk_level(RiskLevel.MODERATE)

        # Validate a new position
        approved, reason, suggested_size = await risk_manager.validate_new_position(
            symbol="USD/JPY",
            direction="BUY",
            size=1.0,
            entry_price=150.25,
            confidence=0.75
        )

        print(f"Position validation: {approved}")
        print(f"Reason: {reason}")
        print(f"Suggested size: {suggested_size}")

        if approved:
            # Calculate dynamic stop loss
            stop_loss = await risk_manager.calculate_dynamic_stop_loss(
                symbol="USD/JPY",
                direction="BUY",
                entry_price=150.25,
                position_size=suggested_size,
                stop_type=StopLossType.ATR_BASED
            )

            print(f"Calculated stop loss: {stop_loss:.5f}")

            # Add position to tracking
            position_data = {
                'position_id': 'TEST_001',
                'symbol': 'USD/JPY',
                'direction': 'BUY',
                'size': suggested_size,
                'entry_price': 150.25,
                'current_price': 150.25,
                'unrealized_pnl': 0.0
            }

            risk_manager.add_position(position_data)

            # Analyze portfolio risk
            risk_metrics = await risk_manager.analyze_portfolio_risk()
            print(f"Portfolio risk analysis completed")
            print(f"Total exposure: ${risk_metrics.total_exposure:,.2f}")

            # Get risk summary
            risk_summary = risk_manager.get_risk_summary()
            print("Risk Summary:", json.dumps(risk_summary, indent=2))

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())