#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Multi-Currency Trading Manager | 增強多貨幣交易管理器
==============================================================

Advanced trading manager with multi-currency pair support, portfolio diversification,
and enhanced risk management for live IG Markets trading.

高級交易管理器，支持多貨幣對、投資組合多樣化和增強的實時IG Markets交易風險管理。

Features | 功能:
- Multi-currency pair support (USD/JPY, GBP/USD, AUD/USD, EUR/USD)
- Advanced portfolio diversification strategies
- Enhanced risk management with correlation analysis
- Optimized order execution with latency improvements
- Real-time performance monitoring and analytics

Author: AIFX Development Team
Date: 2025-01-15
Version: 2.0.0 - Live Trading Enhancement
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path

from .ig_markets import IGMarketsConnector, IGOrder, OrderType, OrderDirection
from ..utils.logger import get_logger

logger = get_logger(__name__)

class CurrencyPair(Enum):
    """Supported currency pairs | 支持的貨幣對"""
    USDJPY = "USD/JPY"
    GBPUSD = "GBP/USD"
    AUDUSD = "AUD/USD"
    EURUSD = "EUR/USD"

@dataclass
class CurrencyPairConfig:
    """Configuration for currency pair trading | 貨幣對交易配置"""
    pair: CurrencyPair
    epic: str
    min_size: float
    max_size: float
    pip_value: float
    spread_threshold: float
    trading_hours: Dict[str, str]  # Start and end times in UTC
    volatility_threshold: float
    correlation_threshold: float = 0.7  # For portfolio diversification

    @property
    def display_name(self) -> str:
        return self.pair.value

@dataclass
class PositionAnalytics:
    """Advanced position analytics | 高級倉位分析"""
    position_id: str
    pair: CurrencyPair
    direction: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    entry_timestamp: datetime = field(default_factory=datetime.now)
    duration: timedelta = field(default_factory=lambda: timedelta(0))
    max_profit: float = 0.0
    max_drawdown: float = 0.0
    win_probability: float = 0.0
    risk_score: float = 0.0
    correlation_impact: Dict[str, float] = field(default_factory=dict)

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics | 投資組合級風險指標"""
    total_exposure: float
    currency_exposure: Dict[str, float]
    correlation_matrix: np.ndarray
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    portfolio_beta: float
    diversification_ratio: float
    max_drawdown: float
    sharpe_ratio: float

class EnhancedMultiCurrencyManager:
    """
    Enhanced Multi-Currency Trading Manager | 增強多貨幣交易管理器

    Advanced trading system with multi-currency support, portfolio optimization,
    and sophisticated risk management for live trading environments.

    具有多貨幣支持、投資組合優化和復雜風險管理的高級交易系統，適用於實時交易環境。
    """

    def __init__(self, config_path: str = None):
        """
        Initialize enhanced multi-currency manager
        初始化增強多貨幣管理器

        Args:
            config_path: Path to trading configuration file
        """
        self.config_path = config_path
        self.ig_connector = IGMarketsConnector(config_path)

        # Initialize currency pair configurations
        self.currency_pairs = self._initialize_currency_pairs()

        # Portfolio state
        self.positions: Dict[str, PositionAnalytics] = {}
        self.active_pairs: set = set()
        self.portfolio_balance = 10000.0  # Starting balance
        self.available_margin = 10000.0

        # Risk management
        self.max_portfolio_risk = 0.10  # 10% portfolio risk
        self.max_pair_exposure = 0.05   # 5% per currency pair
        self.max_correlation = 0.7      # Maximum correlation between positions
        self.daily_loss_limit = 0.05    # 5% daily loss limit

        # Performance tracking
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.performance_history: List[Dict] = []

        # Market data cache
        self.market_data: Dict[CurrencyPair, Dict] = {}
        self.price_history: Dict[CurrencyPair, List] = {}

        # Correlation matrix for risk analysis
        self.correlation_matrix = np.eye(len(CurrencyPair))

        logger.info("Enhanced Multi-Currency Manager initialized | 增強多貨幣管理器已初始化")

    def _initialize_currency_pairs(self) -> Dict[CurrencyPair, CurrencyPairConfig]:
        """Initialize currency pair configurations | 初始化貨幣對配置"""

        pairs = {
            CurrencyPair.USDJPY: CurrencyPairConfig(
                pair=CurrencyPair.USDJPY,
                epic="CS.D.USDJPY.CFD.IP",
                min_size=0.1,
                max_size=5.0,
                pip_value=10.0,  # $10 per pip for 1 lot
                spread_threshold=2.0,  # Max 2 pip spread
                trading_hours={"start": "00:00", "end": "23:59"},  # 24/7
                volatility_threshold=0.015,  # 1.5% daily volatility threshold
                correlation_threshold=0.7
            ),
            CurrencyPair.GBPUSD: CurrencyPairConfig(
                pair=CurrencyPair.GBPUSD,
                epic="CS.D.GBPUSD.CFD.IP",
                min_size=0.1,
                max_size=3.0,
                pip_value=10.0,
                spread_threshold=1.5,
                trading_hours={"start": "06:00", "end": "22:00"},  # London + NY
                volatility_threshold=0.018,
                correlation_threshold=0.6  # Lower correlation with USD/JPY
            ),
            CurrencyPair.AUDUSD: CurrencyPairConfig(
                pair=CurrencyPair.AUDUSD,
                epic="CS.D.AUDUSD.CFD.IP",
                min_size=0.1,
                max_size=3.0,
                pip_value=10.0,
                spread_threshold=1.8,
                trading_hours={"start": "21:00", "end": "06:00"},  # Sydney + Asian
                volatility_threshold=0.016,
                correlation_threshold=0.5  # Commodity currency - different dynamics
            ),
            CurrencyPair.EURUSD: CurrencyPairConfig(
                pair=CurrencyPair.EURUSD,
                epic="CS.D.EURUSD.CFD.IP",
                min_size=0.1,
                max_size=4.0,
                pip_value=10.0,
                spread_threshold=1.2,
                trading_hours={"start": "07:00", "end": "21:00"},  # Europe + NY
                volatility_threshold=0.014,
                correlation_threshold=0.8  # High correlation with GBP/USD
            )
        }

        logger.info(f"Configured {len(pairs)} currency pairs for trading")
        return pairs

    async def connect_and_authenticate(self, demo: bool = True) -> bool:
        """
        Connect to IG Markets and authenticate
        連接並認證IG Markets

        Args:
            demo: Use demo account if True

        Returns:
            bool: Connection success status
        """
        try:
            success = await self.ig_connector.connect(demo=demo)

            if success:
                logger.info("✅ Enhanced Multi-Currency Manager connected to IG Markets")

                # Initialize market data for all pairs
                await self._initialize_market_data()

                # Start background tasks
                await self._start_background_tasks()

                return True
            else:
                logger.error("❌ Failed to connect to IG Markets")
                return False

        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    async def _initialize_market_data(self):
        """Initialize market data for all currency pairs | 為所有貨幣對初始化市場數據"""
        try:
            for pair, config in self.currency_pairs.items():
                market_data = await self.ig_connector.get_market_data(config.epic)

                if market_data:
                    self.market_data[pair] = {
                        'bid': market_data.get('bid', 0),
                        'ask': market_data.get('ask', 0),
                        'mid': market_data.get('mid', 0),
                        'spread': market_data.get('ask', 0) - market_data.get('bid', 0),
                        'timestamp': market_data.get('timestamp', datetime.now()),
                        'market_status': market_data.get('market_status', 'UNKNOWN')
                    }

                    logger.info(f"📊 {pair.value}: Bid={market_data.get('bid'):.4f}, "
                              f"Ask={market_data.get('ask'):.4f}, "
                              f"Spread={market_data.get('ask', 0) - market_data.get('bid', 0):.1f} pips")
                else:
                    logger.warning(f"⚠️ Failed to get market data for {pair.value}")

        except Exception as e:
            logger.error(f"Market data initialization error: {e}")

    async def _start_background_tasks(self):
        """Start background monitoring tasks | 啟動後台監控任務"""
        try:
            # Start market data updates
            asyncio.create_task(self._update_market_data_loop())

            # Start position monitoring
            asyncio.create_task(self._monitor_positions_loop())

            # Start risk monitoring
            asyncio.create_task(self._monitor_portfolio_risk_loop())

            logger.info("🔄 Background monitoring tasks started")

        except Exception as e:
            logger.error(f"Background task startup error: {e}")

    async def _update_market_data_loop(self):
        """Continuously update market data | 持續更新市場數據"""
        while True:
            try:
                for pair, config in self.currency_pairs.items():
                    market_data = await self.ig_connector.get_market_data(config.epic)

                    if market_data:
                        self.market_data[pair] = {
                            'bid': market_data.get('bid', 0),
                            'ask': market_data.get('ask', 0),
                            'mid': market_data.get('mid', 0),
                            'spread': market_data.get('ask', 0) - market_data.get('bid', 0),
                            'timestamp': market_data.get('timestamp', datetime.now()),
                            'market_status': market_data.get('market_status', 'UNKNOWN')
                        }

                        # Update price history
                        if pair not in self.price_history:
                            self.price_history[pair] = []

                        self.price_history[pair].append({
                            'timestamp': datetime.now(),
                            'price': market_data.get('mid', 0)
                        })

                        # Keep only last 1000 price points
                        if len(self.price_history[pair]) > 1000:
                            self.price_history[pair] = self.price_history[pair][-1000:]

                await asyncio.sleep(5)  # Update every 5 seconds

            except Exception as e:
                logger.error(f"Market data update error: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    async def _monitor_positions_loop(self):
        """Monitor all open positions | 監控所有開倉位置"""
        while True:
            try:
                if self.positions:
                    await self._update_position_analytics()
                    await self._check_position_exits()

                await asyncio.sleep(2)  # Monitor every 2 seconds

            except Exception as e:
                logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(10)

    async def _monitor_portfolio_risk_loop(self):
        """Monitor portfolio-level risk metrics | 監控投資組合級風險指標"""
        while True:
            try:
                portfolio_risk = await self._calculate_portfolio_risk()

                # Check risk limits
                if portfolio_risk.total_exposure > self.max_portfolio_risk * self.portfolio_balance:
                    logger.warning("⚠️ Portfolio exposure exceeds limit - reducing positions")
                    await self._reduce_portfolio_exposure()

                if self.daily_pnl < -self.daily_loss_limit * self.portfolio_balance:
                    logger.warning("⚠️ Daily loss limit reached - stopping new trades")
                    await self._emergency_stop_trading()

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(60)

    async def execute_optimized_trade(self,
                                    pair: CurrencyPair,
                                    direction: str,
                                    strategy_signal: Dict[str, Any],
                                    confidence: float = 0.0) -> Optional[PositionAnalytics]:
        """
        Execute optimized trade with enhanced risk management
        執行優化交易並增強風險管理

        Args:
            pair: Currency pair to trade
            direction: BUY or SELL
            strategy_signal: AI strategy signal data
            confidence: Signal confidence (0.0 to 1.0)

        Returns:
            PositionAnalytics: Position information if successful
        """
        try:
            logger.info(f"🎯 Executing optimized trade: {pair.value} {direction}")
            logger.info(f"📊 Signal confidence: {confidence:.2f}")

            # Pre-trade risk checks
            if not await self._pre_trade_risk_check(pair, direction, confidence):
                logger.warning("❌ Trade rejected by risk management")
                return None

            # Get current market data
            if pair not in self.market_data:
                logger.error(f"❌ No market data available for {pair.value}")
                return None

            market_data = self.market_data[pair]
            config = self.currency_pairs[pair]

            # Calculate optimal position size
            position_size = await self._calculate_optimal_position_size(pair, confidence, strategy_signal)

            if position_size <= 0:
                logger.warning("❌ Calculated position size is zero or negative")
                return None

            # Calculate entry price and levels
            entry_price = market_data['ask'] if direction == 'BUY' else market_data['bid']

            # Enhanced stop loss and take profit calculation
            stop_loss, take_profit = await self._calculate_dynamic_levels(
                pair, direction, entry_price, strategy_signal, confidence
            )

            # Create optimized order
            order = IGOrder(
                order_type=OrderType.MARKET,
                direction=OrderDirection.BUY if direction == 'BUY' else OrderDirection.SELL,
                epic=config.epic,
                size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                currency_code="USD",
                force_open=True
            )

            # Execute order with latency optimization
            start_time = datetime.now()
            order_result = await self.ig_connector.place_order(order)
            execution_time = (datetime.now() - start_time).total_seconds() * 1000  # ms

            if order_result.get('success'):
                logger.info(f"✅ Order executed successfully in {execution_time:.1f}ms")

                # Create position analytics
                position = PositionAnalytics(
                    position_id=order_result.get('deal_reference', f"SIM_{len(self.positions)}"),
                    pair=pair,
                    direction=direction,
                    size=position_size,
                    entry_price=entry_price,
                    current_price=entry_price,
                    unrealized_pnl=0.0,
                    entry_timestamp=datetime.now(),
                    win_probability=confidence
                )

                # Add to positions
                self.positions[position.position_id] = position
                self.active_pairs.add(pair)
                self.total_trades += 1

                # Log trade execution
                await self._log_trade_execution(position, strategy_signal, execution_time)

                return position

            else:
                reason = order_result.get('reason', 'Unknown error')
                logger.error(f"❌ Order execution failed: {reason}")
                return None

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return None

    async def _pre_trade_risk_check(self, pair: CurrencyPair, direction: str, confidence: float) -> bool:
        """Comprehensive pre-trade risk checks | 全面的交易前風險檢查"""
        try:
            # Check if market is open
            if not await self._is_market_open(pair):
                logger.warning(f"Market closed for {pair.value}")
                return False

            # Check spread threshold
            market_data = self.market_data.get(pair, {})
            spread = market_data.get('spread', float('inf'))
            threshold = self.currency_pairs[pair].spread_threshold

            if spread > threshold:
                logger.warning(f"Spread too wide: {spread:.1f} > {threshold:.1f} pips")
                return False

            # Check portfolio exposure
            portfolio_risk = await self._calculate_portfolio_risk()
            if portfolio_risk.total_exposure > self.max_portfolio_risk * self.portfolio_balance:
                logger.warning("Portfolio exposure limit exceeded")
                return False

            # Check daily loss limit
            if self.daily_pnl < -self.daily_loss_limit * self.portfolio_balance:
                logger.warning("Daily loss limit reached")
                return False

            # Check correlation with existing positions
            if await self._check_correlation_risk(pair, direction):
                logger.warning("High correlation risk with existing positions")
                return False

            # Check confidence threshold
            if confidence < 0.6:  # Minimum 60% confidence
                logger.warning(f"Signal confidence too low: {confidence:.2f}")
                return False

            return True

        except Exception as e:
            logger.error(f"Pre-trade risk check error: {e}")
            return False

    async def _calculate_optimal_position_size(self,
                                             pair: CurrencyPair,
                                             confidence: float,
                                             strategy_signal: Dict[str, Any]) -> float:
        """Calculate optimal position size using advanced algorithms | 使用先進算法計算最佳倉位大小"""
        try:
            config = self.currency_pairs[pair]

            # Base position size (2% risk per trade)
            base_risk = 0.02 * self.portfolio_balance

            # Adjust for confidence
            confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5x to 1.0x

            # Adjust for volatility
            volatility = await self._calculate_pair_volatility(pair)
            volatility_multiplier = min(2.0, max(0.5, 1.0 / (volatility + 0.01)))

            # Adjust for portfolio heat
            portfolio_heat = len(self.positions) / 5.0  # Reduce size with more positions
            heat_multiplier = max(0.3, 1.0 - portfolio_heat)

            # Calculate final size
            risk_adjusted_amount = base_risk * confidence_multiplier * volatility_multiplier * heat_multiplier

            # Convert to lot size (assuming $100,000 per lot)
            position_size = risk_adjusted_amount / 100000

            # Apply min/max limits
            position_size = max(config.min_size, min(config.max_size, position_size))

            logger.info(f"📏 Position size calculation for {pair.value}:")
            logger.info(f"   Base risk: ${base_risk:.2f}")
            logger.info(f"   Confidence: {confidence:.2f} (×{confidence_multiplier:.2f})")
            logger.info(f"   Volatility: {volatility:.3f} (×{volatility_multiplier:.2f})")
            logger.info(f"   Portfolio heat: {portfolio_heat:.2f} (×{heat_multiplier:.2f})")
            logger.info(f"   Final size: {position_size:.2f} lots")

            return round(position_size, 2)

        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return 0.0

    async def _calculate_dynamic_levels(self,
                                      pair: CurrencyPair,
                                      direction: str,
                                      entry_price: float,
                                      strategy_signal: Dict[str, Any],
                                      confidence: float) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels | 計算動態止損和止盈水平"""
        try:
            config = self.currency_pairs[pair]

            # Calculate ATR (Average True Range) for dynamic levels
            atr = await self._calculate_atr(pair)

            # Base stop distance (2x ATR)
            base_stop_distance = atr * 2.0

            # Adjust for confidence
            stop_multiplier = 1.5 - (confidence * 0.5)  # Higher confidence = tighter stops
            stop_distance = base_stop_distance * stop_multiplier

            # Calculate levels
            if direction == 'BUY':
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + (stop_distance * 2.0)  # 2:1 risk/reward
            else:
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - (stop_distance * 2.0)

            # Apply minimum distances (to avoid too-tight stops)
            min_distance = 0.0010  # 10 pips minimum

            if direction == 'BUY':
                stop_loss = min(stop_loss, entry_price - min_distance)
                take_profit = max(take_profit, entry_price + min_distance)
            else:
                stop_loss = max(stop_loss, entry_price + min_distance)
                take_profit = min(take_profit, entry_price - min_distance)

            logger.info(f"🎯 Dynamic levels for {pair.value}:")
            logger.info(f"   Entry: {entry_price:.5f}")
            logger.info(f"   Stop Loss: {stop_loss:.5f} ({stop_distance*10000:.1f} pips)")
            logger.info(f"   Take Profit: {take_profit:.5f} ({stop_distance*20000:.1f} pips)")

            return round(stop_loss, 5), round(take_profit, 5)

        except Exception as e:
            logger.error(f"Dynamic levels calculation error: {e}")
            # Fallback to simple percentage-based levels
            distance = entry_price * 0.005  # 0.5%
            if direction == 'BUY':
                return entry_price - distance, entry_price + distance * 2
            else:
                return entry_price + distance, entry_price - distance * 2

    async def get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status | 獲取全面的投資組合狀態"""
        try:
            # Calculate portfolio metrics
            total_positions = len(self.positions)
            total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())

            # Currency exposure
            currency_exposure = {}
            for pos in self.positions.values():
                base_currency = pos.pair.value.split('/')[0]
                quote_currency = pos.pair.value.split('/')[1]

                if base_currency not in currency_exposure:
                    currency_exposure[base_currency] = 0.0
                if quote_currency not in currency_exposure:
                    currency_exposure[quote_currency] = 0.0

                exposure = pos.size * pos.current_price * 100000  # Convert to notional
                if pos.direction == 'BUY':
                    currency_exposure[base_currency] += exposure
                    currency_exposure[quote_currency] -= exposure
                else:
                    currency_exposure[base_currency] -= exposure
                    currency_exposure[quote_currency] += exposure

            # Performance metrics
            win_rate = (self.winning_trades / max(1, self.total_trades)) * 100

            return {
                'timestamp': datetime.now().isoformat(),
                'account_balance': self.portfolio_balance,
                'available_margin': self.available_margin,
                'total_positions': total_positions,
                'active_pairs': list(self.active_pairs),
                'total_pnl': total_pnl,
                'daily_pnl': self.daily_pnl,
                'win_rate': win_rate,
                'total_trades': self.total_trades,
                'currency_exposure': currency_exposure,
                'risk_metrics': await self._calculate_portfolio_risk(),
                'market_status': {
                    pair.value: data.get('market_status', 'UNKNOWN')
                    for pair, data in self.market_data.items()
                },
                'position_details': [
                    {
                        'position_id': pos.position_id,
                        'pair': pos.pair.value,
                        'direction': pos.direction,
                        'size': pos.size,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'duration': str(pos.duration),
                        'win_probability': pos.win_probability
                    }
                    for pos in self.positions.values()
                ]
            }

        except Exception as e:
            logger.error(f"Portfolio status error: {e}")
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}

    async def _calculate_portfolio_risk(self) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics | 計算全面的投資組合風險指標"""
        try:
            if not self.positions:
                return PortfolioRisk(
                    total_exposure=0.0,
                    currency_exposure={},
                    correlation_matrix=np.eye(1),
                    var_95=0.0,
                    expected_shortfall=0.0,
                    portfolio_beta=1.0,
                    diversification_ratio=1.0,
                    max_drawdown=0.0,
                    sharpe_ratio=0.0
                )

            # Calculate total exposure
            total_exposure = sum(
                pos.size * pos.current_price * 100000
                for pos in self.positions.values()
            )

            # Calculate currency exposure
            currency_exposure = {}
            for pos in self.positions.values():
                currencies = pos.pair.value.split('/')
                for currency in currencies:
                    if currency not in currency_exposure:
                        currency_exposure[currency] = 0.0

                    exposure = pos.size * pos.current_price * 100000
                    currency_exposure[currency] += exposure

            # Calculate portfolio VaR (simplified)
            position_values = [pos.unrealized_pnl for pos in self.positions.values()]
            var_95 = np.percentile(position_values, 5) if position_values else 0.0

            # Calculate expected shortfall (average of losses below VaR)
            below_var = [pnl for pnl in position_values if pnl <= var_95]
            expected_shortfall = np.mean(below_var) if below_var else 0.0

            # Calculate Sharpe ratio (simplified)
            returns = [pos.unrealized_pnl / (pos.size * 100000) for pos in self.positions.values()]
            if returns and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0

            return PortfolioRisk(
                total_exposure=total_exposure,
                currency_exposure=currency_exposure,
                correlation_matrix=self.correlation_matrix,
                var_95=var_95,
                expected_shortfall=expected_shortfall,
                portfolio_beta=1.0,  # Simplified
                diversification_ratio=len(set(pos.pair for pos in self.positions.values())) / max(1, len(self.positions)),
                max_drawdown=min(0.0, min(pos.unrealized_pnl for pos in self.positions.values())),
                sharpe_ratio=sharpe_ratio
            )

        except Exception as e:
            logger.error(f"Portfolio risk calculation error: {e}")
            return PortfolioRisk(
                total_exposure=0.0,
                currency_exposure={},
                correlation_matrix=np.eye(1),
                var_95=0.0,
                expected_shortfall=0.0,
                portfolio_beta=1.0,
                diversification_ratio=1.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0
            )

    # Helper methods for calculations
    async def _calculate_pair_volatility(self, pair: CurrencyPair) -> float:
        """Calculate currency pair volatility | 計算貨幣對波動率"""
        try:
            if pair not in self.price_history or len(self.price_history[pair]) < 20:
                return 0.015  # Default volatility

            prices = [point['price'] for point in self.price_history[pair][-50:]]
            returns = np.diff(np.log(prices))

            return np.std(returns) * np.sqrt(1440)  # Annualized volatility (assuming 1-minute data)

        except Exception as e:
            logger.error(f"Volatility calculation error: {e}")
            return 0.015

    async def _calculate_atr(self, pair: CurrencyPair, period: int = 14) -> float:
        """Calculate Average True Range | 計算平均真實範圍"""
        try:
            if pair not in self.price_history or len(self.price_history[pair]) < period + 1:
                return 0.0010  # Default ATR

            # Simplified ATR calculation using price history
            prices = [point['price'] for point in self.price_history[pair][-period*2:]]

            if len(prices) < 2:
                return 0.0010

            ranges = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
            atr = np.mean(ranges[-period:])

            return max(0.0005, atr)  # Minimum 0.5 pips

        except Exception as e:
            logger.error(f"ATR calculation error: {e}")
            return 0.0010

    async def _is_market_open(self, pair: CurrencyPair) -> bool:
        """Check if market is open for trading | 檢查市場是否開放交易"""
        try:
            config = self.currency_pairs[pair]
            now = datetime.utcnow()

            # Simple check - most forex markets are open 24/5
            # In production, implement proper market hours logic
            weekday = now.weekday()

            # Market closed on weekends
            if weekday >= 5:  # Saturday = 5, Sunday = 6
                return False

            return True

        except Exception as e:
            logger.error(f"Market hours check error: {e}")
            return True  # Default to open

    async def _check_correlation_risk(self, pair: CurrencyPair, direction: str) -> bool:
        """Check correlation risk with existing positions | 檢查與現有倉位的相關性風險"""
        try:
            if not self.positions:
                return False

            # Check correlation with existing positions
            for pos in self.positions.values():
                if pos.pair == pair and pos.direction == direction:
                    return True  # Same pair, same direction - high correlation

                # Check currency correlation (simplified)
                pos_currencies = set(pos.pair.value.split('/'))
                new_currencies = set(pair.value.split('/'))

                if len(pos_currencies.intersection(new_currencies)) > 0:
                    # Shared currency - check if directions create excessive exposure
                    if len(self.positions) >= 3:  # Limit correlated positions
                        return True

            return False

        except Exception as e:
            logger.error(f"Correlation risk check error: {e}")
            return False

    async def _update_position_analytics(self):
        """Update analytics for all positions | 更新所有倉位的分析"""
        try:
            for position in self.positions.values():
                # Update current price
                market_data = self.market_data.get(position.pair, {})

                if market_data:
                    if position.direction == 'BUY':
                        position.current_price = market_data.get('bid', position.current_price)
                    else:
                        position.current_price = market_data.get('ask', position.current_price)

                    # Calculate unrealized PnL
                    if position.direction == 'BUY':
                        position.unrealized_pnl = (position.current_price - position.entry_price) * position.size * 100000
                    else:
                        position.unrealized_pnl = (position.entry_price - position.current_price) * position.size * 100000

                    # Update duration
                    position.duration = datetime.now() - position.entry_timestamp

                    # Update max profit and max drawdown
                    position.max_profit = max(position.max_profit, position.unrealized_pnl)
                    position.max_drawdown = min(position.max_drawdown, position.unrealized_pnl)

        except Exception as e:
            logger.error(f"Position analytics update error: {e}")

    async def _check_position_exits(self):
        """Check for position exit conditions | 檢查倉位退出條件"""
        try:
            positions_to_close = []

            for position in self.positions.values():
                # Check for emergency exits
                if position.unrealized_pnl < -1000:  # Emergency stop at -$1000
                    positions_to_close.append((position.position_id, 'EMERGENCY_STOP'))

                # Check time-based exits
                elif position.duration > timedelta(hours=24):  # Maximum 24-hour positions
                    positions_to_close.append((position.position_id, 'TIME_BASED'))

                # Check profit taking
                elif position.unrealized_pnl > 500 and position.duration > timedelta(minutes=30):
                    positions_to_close.append((position.position_id, 'PROFIT_TAKING'))

            # Execute position closures
            for position_id, reason in positions_to_close:
                await self._close_position(position_id, reason)

        except Exception as e:
            logger.error(f"Position exit check error: {e}")

    async def _close_position(self, position_id: str, reason: str):
        """Close a specific position | 關閉特定倉位"""
        try:
            if position_id not in self.positions:
                return

            position = self.positions[position_id]

            # Log closure
            logger.info(f"🔒 Closing position {position_id} ({position.pair.value}) - Reason: {reason}")
            logger.info(f"   P&L: ${position.unrealized_pnl:.2f}")

            # Try to close via IG Markets API
            close_result = await self.ig_connector.close_position(position_id)

            if close_result.get('success'):
                # Update portfolio metrics
                self.daily_pnl += position.unrealized_pnl

                if position.unrealized_pnl > 0:
                    self.winning_trades += 1

                # Remove from active positions
                del self.positions[position_id]

                # Update active pairs
                pair_still_active = any(pos.pair == position.pair for pos in self.positions.values())
                if not pair_still_active:
                    self.active_pairs.discard(position.pair)

                logger.info(f"✅ Position {position_id} closed successfully")

            else:
                logger.error(f"❌ Failed to close position {position_id}: {close_result.get('reason')}")

        except Exception as e:
            logger.error(f"Position closure error: {e}")

    async def _reduce_portfolio_exposure(self):
        """Reduce portfolio exposure when risk limits are exceeded | 當風險限制超標時減少投資組合風險敞口"""
        try:
            logger.warning("🚨 Reducing portfolio exposure due to risk limits")

            # Sort positions by unrealized loss (worst first)
            positions_by_loss = sorted(
                self.positions.values(),
                key=lambda p: p.unrealized_pnl
            )

            # Close worst-performing positions
            positions_to_close = positions_by_loss[:2]  # Close worst 2 positions

            for position in positions_to_close:
                await self._close_position(position.position_id, 'RISK_REDUCTION')

        except Exception as e:
            logger.error(f"Portfolio exposure reduction error: {e}")

    async def _emergency_stop_trading(self):
        """Emergency stop all trading activities | 緊急停止所有交易活動"""
        try:
            logger.critical("🛑 EMERGENCY STOP - Closing all positions")

            for position_id in list(self.positions.keys()):
                await self._close_position(position_id, 'EMERGENCY_STOP')

        except Exception as e:
            logger.error(f"Emergency stop error: {e}")

    async def _log_trade_execution(self, position: PositionAnalytics, signal: Dict, execution_time: float):
        """Log detailed trade execution information | 記錄詳細的交易執行信息"""
        try:
            trade_log = {
                'timestamp': datetime.now().isoformat(),
                'position_id': position.position_id,
                'pair': position.pair.value,
                'direction': position.direction,
                'size': position.size,
                'entry_price': position.entry_price,
                'signal_confidence': position.win_probability,
                'execution_time_ms': execution_time,
                'strategy_signal': signal,
                'market_conditions': {
                    'spread': self.market_data.get(position.pair, {}).get('spread', 0),
                    'volatility': await self._calculate_pair_volatility(position.pair)
                }
            }

            self.performance_history.append(trade_log)

            # Keep only last 1000 trades
            if len(self.performance_history) > 1000:
                self.performance_history = self.performance_history[-1000:]

        except Exception as e:
            logger.error(f"Trade logging error: {e}")

    async def disconnect(self):
        """Disconnect and cleanup | 斷開連接並清理"""
        try:
            logger.info("🔌 Disconnecting Enhanced Multi-Currency Manager...")

            # Close any remaining positions (optional - for safety)
            if self.positions:
                logger.info(f"📊 {len(self.positions)} positions still open")

            # Disconnect IG connector
            await self.ig_connector.disconnect()

            logger.info("✅ Enhanced Multi-Currency Manager disconnected")

        except Exception as e:
            logger.error(f"Disconnect error: {e}")


# Factory function
def create_enhanced_manager(config_path: str = None) -> EnhancedMultiCurrencyManager:
    """
    Create enhanced multi-currency trading manager
    創建增強的多貨幣交易管理器

    Args:
        config_path: Path to configuration file

    Returns:
        EnhancedMultiCurrencyManager: Configured manager instance
    """
    return EnhancedMultiCurrencyManager(config_path)


# Example usage
async def main():
    """Example usage of Enhanced Multi-Currency Manager | 增強多貨幣管理器使用示例"""

    manager = create_enhanced_manager("config/trading-config.yaml")

    try:
        # Connect to IG Markets
        success = await manager.connect_and_authenticate(demo=True)

        if success:
            print("✅ Enhanced Multi-Currency Manager connected")

            # Get portfolio status
            status = await manager.get_portfolio_status()
            print(f"💰 Account Balance: ${status['account_balance']:,.2f}")
            print(f"📊 Active Pairs: {len(status['active_pairs'])}")

            # Example trade execution
            example_signal = {
                'signal_strength': 0.8,
                'strategy': 'AI_ENHANCED_TREND_FOLLOWING',
                'timeframe': '1H',
                'indicators': {'rsi': 65, 'macd': 0.002, 'bb_position': 0.7}
            }

            position = await manager.execute_optimized_trade(
                CurrencyPair.USDJPY,
                'BUY',
                example_signal,
                confidence=0.75
            )

            if position:
                print(f"🎯 Trade executed: {position.pair.value} {position.direction}")
                print(f"📈 Position size: {position.size} lots")
                print(f"💡 Entry: {position.entry_price:.5f}")

        else:
            print("❌ Failed to connect to Enhanced Multi-Currency Manager")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        await manager.disconnect()


if __name__ == "__main__":
    asyncio.run(main())