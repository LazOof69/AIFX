"""
Risk Management System | 風險管理系統

Comprehensive risk management for the AIFX trading system including position sizing,
stop-loss, take-profit, and portfolio-level risk controls.
AIFX交易系統的綜合風險管理，包括倉位大小、止損、止盈和投資組合級風險控制。

This module provides sophisticated risk management mechanisms for forex trading strategies.
此模組為外匯交易策略提供精細的風險管理機制。
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
from abc import ABC, abstractmethod

from .signal_combiner import TradingSignal, SignalType

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration | 風險等級枚舉"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class PositionSizeMethod(Enum):
    """Position sizing methods | 倉位大小方法"""
    FIXED_PERCENTAGE = "fixed_percentage"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    ATR_BASED = "atr_based"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


@dataclass
class RiskParameters:
    """
    Risk management parameters | 風險管理參數
    """
    # Position sizing | 倉位大小
    max_position_size: float = 0.02  # Maximum 2% per trade
    min_position_size: float = 0.005  # Minimum 0.5% per trade
    position_sizing_method: PositionSizeMethod = PositionSizeMethod.FIXED_PERCENTAGE
    
    # Portfolio limits | 投資組合限制
    max_portfolio_risk: float = 0.10  # Maximum 10% total portfolio risk
    max_open_positions: int = 5  # Maximum simultaneous positions
    max_daily_trades: int = 10  # Maximum trades per day
    
    # Stop-loss and take-profit | 止損和止盈
    stop_loss_atr_multiplier: float = 2.0  # ATR multiplier for stop loss
    take_profit_atr_multiplier: float = 3.0  # ATR multiplier for take profit
    trailing_stop_enabled: bool = True
    trailing_stop_atr_multiplier: float = 1.5
    
    # Risk-reward ratios | 風險收益比
    min_risk_reward_ratio: float = 1.5  # Minimum 1.5:1 risk-reward
    max_risk_per_trade: float = 0.02  # Maximum 2% risk per trade
    
    # Drawdown protection | 回撤保護
    max_drawdown_limit: float = 0.15  # Stop trading if 15% drawdown
    daily_loss_limit: float = 0.05  # Stop trading if 5% daily loss
    
    # Confidence-based adjustments | 信心基礎調整
    min_signal_confidence: float = 0.6  # Minimum confidence for trade
    confidence_scaling_factor: float = 1.0  # Scale position by confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary | 轉換為字典"""
        return {
            'max_position_size': self.max_position_size,
            'min_position_size': self.min_position_size,
            'position_sizing_method': self.position_sizing_method.value,
            'max_portfolio_risk': self.max_portfolio_risk,
            'max_open_positions': self.max_open_positions,
            'max_daily_trades': self.max_daily_trades,
            'stop_loss_atr_multiplier': self.stop_loss_atr_multiplier,
            'take_profit_atr_multiplier': self.take_profit_atr_multiplier,
            'trailing_stop_enabled': self.trailing_stop_enabled,
            'trailing_stop_atr_multiplier': self.trailing_stop_atr_multiplier,
            'min_risk_reward_ratio': self.min_risk_reward_ratio,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_drawdown_limit': self.max_drawdown_limit,
            'daily_loss_limit': self.daily_loss_limit,
            'min_signal_confidence': self.min_signal_confidence,
            'confidence_scaling_factor': self.confidence_scaling_factor
        }


@dataclass
class Position:
    """
    Trading position representation | 交易倉位表示
    """
    symbol: str
    side: str  # 'long' or 'short'
    size: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    entry_time: Optional[datetime] = None
    signal_confidence: float = 0.0
    unrealized_pnl: float = 0.0
    
    def update_current_price(self, price: float):
        """Update current price and unrealized PnL | 更新當前價格和未實現盈虧"""
        self.current_price = price
        if self.side == 'long':
            self.unrealized_pnl = (price - self.entry_price) * self.size
        else:  # short
            self.unrealized_pnl = (self.entry_price - price) * self.size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary | 轉換為字典"""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'size': self.size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'signal_confidence': self.signal_confidence,
            'unrealized_pnl': self.unrealized_pnl
        }


@dataclass
class RiskMetrics:
    """
    Risk metrics calculation results | 風險指標計算結果
    """
    portfolio_risk: float = 0.0
    position_size: float = 0.0
    stop_loss_level: float = 0.0
    take_profit_level: float = 0.0
    risk_reward_ratio: float = 0.0
    max_loss_amount: float = 0.0
    confidence_adjustment: float = 1.0
    risk_approval: bool = False
    risk_warnings: List[str] = None
    
    def __post_init__(self):
        if self.risk_warnings is None:
            self.risk_warnings = []


class BaseRiskManager(ABC):
    """
    Abstract base class for risk management | 風險管理的抽象基類
    """
    
    def __init__(self, risk_parameters: RiskParameters):
        """
        Initialize base risk manager | 初始化基礎風險管理器
        
        Args:
            risk_parameters: Risk management parameters | 風險管理參數
        """
        self.risk_parameters = risk_parameters
        self.open_positions: Dict[str, Position] = {}
        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0
        self.trade_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal, market_data: pd.DataFrame, 
                              account_balance: float) -> float:
        """Calculate position size for a signal | 計算信號的倉位大小"""
        pass
    
    @abstractmethod
    def calculate_stop_loss(self, signal: TradingSignal, entry_price: float, 
                           market_data: pd.DataFrame) -> float:
        """Calculate stop loss level | 計算止損水平"""
        pass
    
    @abstractmethod
    def calculate_take_profit(self, signal: TradingSignal, entry_price: float, 
                            market_data: pd.DataFrame) -> float:
        """Calculate take profit level | 計算止盈水平"""
        pass


class AdvancedRiskManager(BaseRiskManager):
    """
    Advanced risk management system | 高級風險管理系統
    
    Implements sophisticated risk management with multiple position sizing methods,
    dynamic stop-loss/take-profit, and comprehensive portfolio risk controls.
    實現精細的風險管理，包括多種倉位大小方法、動態止損/止盈和綜合投資組合風險控制。
    """
    
    def __init__(self, risk_parameters: RiskParameters, account_balance: float = 100000.0):
        """
        Initialize advanced risk manager | 初始化高級風險管理器
        
        Args:
            risk_parameters: Risk management parameters | 風險管理參數
            account_balance: Initial account balance | 初始賬戶餘額
        """
        super().__init__(risk_parameters)
        self.initial_balance = account_balance
        self.current_balance = account_balance
        self.equity_curve = [account_balance]
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        # Performance tracking | 績效追蹤
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        self.profit_factor = 0.0
        
        # Risk monitoring | 風險監控
        self.risk_violations = []
        self.emergency_stop = False
        
        logger.info(f"Initialized AdvancedRiskManager with balance: ${account_balance:,.2f}")
    
    def evaluate_trade_risk(self, signal: TradingSignal, market_data: pd.DataFrame, 
                          current_price: float) -> RiskMetrics:
        """
        Comprehensive trade risk evaluation | 綜合交易風險評估
        
        Args:
            signal: Trading signal to evaluate | 要評估的交易信號
            market_data: Recent market data | 近期市場數據
            current_price: Current market price | 當前市場價格
            
        Returns:
            Risk metrics and approval decision | 風險指標和批准決定
        """
        risk_metrics = RiskMetrics()
        
        try:
            # 1. Check basic constraints | 檢查基本約束
            if not self._check_basic_constraints(signal, risk_metrics):
                return risk_metrics
            
            # 2. Calculate position size | 計算倉位大小
            position_size = self.calculate_position_size(signal, market_data, self.current_balance)
            risk_metrics.position_size = position_size
            
            # 3. Calculate stop loss and take profit | 計算止損和止盈
            stop_loss = self.calculate_stop_loss(signal, current_price, market_data)
            take_profit = self.calculate_take_profit(signal, current_price, market_data)
            
            risk_metrics.stop_loss_level = stop_loss
            risk_metrics.take_profit_level = take_profit
            
            # 4. Calculate risk-reward ratio | 計算風險收益比
            risk_amount = abs(current_price - stop_loss) * position_size
            reward_amount = abs(take_profit - current_price) * position_size
            
            if risk_amount > 0:
                risk_metrics.risk_reward_ratio = reward_amount / risk_amount
            else:
                risk_metrics.risk_reward_ratio = 0.0
            
            # 5. Calculate portfolio risk | 計算投資組合風險
            risk_metrics.portfolio_risk = self._calculate_portfolio_risk()
            
            # 6. Apply confidence adjustments | 應用信心調整
            risk_metrics.confidence_adjustment = self._calculate_confidence_adjustment(signal)
            
            # 7. Final risk approval decision | 最終風險批准決定
            risk_metrics.risk_approval = self._make_risk_decision(risk_metrics, signal)
            
            # 8. Calculate maximum loss amount | 計算最大損失金額
            risk_metrics.max_loss_amount = risk_amount
            
        except Exception as e:
            logger.error(f"Error in trade risk evaluation: {e}")
            risk_metrics.risk_warnings.append(f"Risk evaluation error: {str(e)}")
            risk_metrics.risk_approval = False
        
        return risk_metrics
    
    def calculate_position_size(self, signal: TradingSignal, market_data: pd.DataFrame, 
                              account_balance: float) -> float:
        """
        Calculate optimal position size | 計算最佳倉位大小
        
        Uses the configured position sizing method with confidence adjustments.
        使用配置的倉位大小方法配合信心調整。
        """
        base_size = 0.0
        
        try:
            method = self.risk_parameters.position_sizing_method
            
            if method == PositionSizeMethod.FIXED_PERCENTAGE:
                base_size = account_balance * self.risk_parameters.max_position_size
                
            elif method == PositionSizeMethod.KELLY_CRITERION:
                base_size = self._calculate_kelly_position_size(signal, account_balance)
                
            elif method == PositionSizeMethod.VOLATILITY_ADJUSTED:
                base_size = self._calculate_volatility_adjusted_size(signal, market_data, account_balance)
                
            elif method == PositionSizeMethod.ATR_BASED:
                base_size = self._calculate_atr_based_size(signal, market_data, account_balance)
                
            elif method == PositionSizeMethod.CONFIDENCE_WEIGHTED:
                base_size = self._calculate_confidence_weighted_size(signal, account_balance)
                
            else:
                # Default to fixed percentage | 默認為固定百分比
                base_size = account_balance * self.risk_parameters.max_position_size
            
            # Apply confidence scaling | 應用信心縮放
            confidence_factor = self._calculate_confidence_adjustment(signal)
            adjusted_size = base_size * confidence_factor
            
            # Apply limits | 應用限制
            min_size = account_balance * self.risk_parameters.min_position_size
            max_size = account_balance * self.risk_parameters.max_position_size
            
            final_size = np.clip(adjusted_size, min_size, max_size)
            
            logger.debug(f"Position size calculation: base={base_size:.2f}, "
                        f"confidence_factor={confidence_factor:.3f}, "
                        f"final={final_size:.2f}")
            
            return final_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            # Return minimum safe position size | 返回最小安全倉位大小
            return account_balance * self.risk_parameters.min_position_size
    
    def calculate_stop_loss(self, signal: TradingSignal, entry_price: float, 
                           market_data: pd.DataFrame) -> float:
        """
        Calculate dynamic stop loss level | 計算動態止損水平
        
        Uses ATR-based stop loss with signal-specific adjustments.
        使用基於ATR的止損配合信號特定調整。
        """
        try:
            # Calculate ATR | 計算ATR
            atr = self._calculate_atr(market_data)
            
            # Base stop loss using ATR multiplier | 使用ATR倍數的基礎止損
            atr_distance = atr * self.risk_parameters.stop_loss_atr_multiplier
            
            # Adjust based on signal type | 根據信號類型調整
            if signal.signal_type == SignalType.BUY:
                stop_loss = entry_price - atr_distance
            elif signal.signal_type == SignalType.SELL:
                stop_loss = entry_price + atr_distance
            else:  # HOLD - no stop loss | HOLD - 無止損
                return entry_price
            
            # Adjust for signal confidence | 根據信號信心調整
            confidence_adjustment = 1.0 + (1.0 - signal.confidence) * 0.5
            
            if signal.signal_type == SignalType.BUY:
                stop_loss = entry_price - (atr_distance * confidence_adjustment)
            else:
                stop_loss = entry_price + (atr_distance * confidence_adjustment)
            
            return max(0.0, stop_loss)  # Ensure non-negative price
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            # Return conservative stop loss | 返回保守止損
            return entry_price * (0.98 if signal.signal_type == SignalType.BUY else 1.02)
    
    def calculate_take_profit(self, signal: TradingSignal, entry_price: float, 
                            market_data: pd.DataFrame) -> float:
        """
        Calculate dynamic take profit level | 計算動態止盈水平
        
        Uses ATR-based take profit with risk-reward ratio optimization.
        使用基於ATR的止盈配合風險收益比優化。
        """
        try:
            # Calculate ATR | 計算ATR
            atr = self._calculate_atr(market_data)
            
            # Base take profit using ATR multiplier | 使用ATR倍數的基礎止盈
            atr_distance = atr * self.risk_parameters.take_profit_atr_multiplier
            
            # Adjust based on signal type | 根據信號類型調整
            if signal.signal_type == SignalType.BUY:
                take_profit = entry_price + atr_distance
            elif signal.signal_type == SignalType.SELL:
                take_profit = entry_price - atr_distance
            else:  # HOLD - no take profit | HOLD - 無止盈
                return entry_price
            
            # Ensure minimum risk-reward ratio | 確保最小風險收益比
            stop_loss = self.calculate_stop_loss(signal, entry_price, market_data)
            risk_distance = abs(entry_price - stop_loss)
            min_reward_distance = risk_distance * self.risk_parameters.min_risk_reward_ratio
            
            if signal.signal_type == SignalType.BUY:
                min_take_profit = entry_price + min_reward_distance
                take_profit = max(take_profit, min_take_profit)
            else:
                min_take_profit = entry_price - min_reward_distance
                take_profit = min(take_profit, min_take_profit)
            
            # Adjust for signal strength | 根據信號強度調整
            strength_multiplier = 0.8 + (signal.strength * 0.4)  # 0.8 to 1.2 range
            
            if signal.signal_type == SignalType.BUY:
                take_profit = entry_price + ((take_profit - entry_price) * strength_multiplier)
            else:
                take_profit = entry_price - ((entry_price - take_profit) * strength_multiplier)
            
            return max(0.0, take_profit)  # Ensure non-negative price
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            # Return conservative take profit | 返回保守止盈
            return entry_price * (1.03 if signal.signal_type == SignalType.BUY else 0.97)
    
    def update_trailing_stops(self, market_data: pd.DataFrame):
        """
        Update trailing stops for open positions | 更新開倉倉位的移動止損
        
        Args:
            market_data: Current market data | 當前市場數據
        """
        if not self.risk_parameters.trailing_stop_enabled:
            return
        
        try:
            current_prices = self._get_current_prices(market_data)
            atr_values = {}
            
            for symbol, position in self.open_positions.items():
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                position.update_current_price(current_price)
                
                # Calculate ATR for trailing stop | 計算移動止損的ATR
                if symbol not in atr_values:
                    symbol_data = market_data[market_data.get('Symbol', market_data.index) == symbol]
                    if not symbol_data.empty:
                        atr_values[symbol] = self._calculate_atr(symbol_data)
                    else:
                        continue
                
                atr = atr_values[symbol]
                trailing_distance = atr * self.risk_parameters.trailing_stop_atr_multiplier
                
                # Update trailing stop | 更新移動止損
                if position.side == 'long':
                    new_trailing_stop = current_price - trailing_distance
                    if position.trailing_stop is None or new_trailing_stop > position.trailing_stop:
                        position.trailing_stop = new_trailing_stop
                        logger.debug(f"Updated trailing stop for {symbol}: {new_trailing_stop:.5f}")
                else:  # short position
                    new_trailing_stop = current_price + trailing_distance
                    if position.trailing_stop is None or new_trailing_stop < position.trailing_stop:
                        position.trailing_stop = new_trailing_stop
                        logger.debug(f"Updated trailing stop for {symbol}: {new_trailing_stop:.5f}")
                        
        except Exception as e:
            logger.error(f"Error updating trailing stops: {e}")
    
    def check_exit_conditions(self, market_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Check exit conditions for open positions | 檢查開倉倉位的離場條件
        
        Args:
            market_data: Current market data | 當前市場數據
            
        Returns:
            List of positions to close | 要關閉的倉位列表
        """
        exit_signals = []
        
        try:
            current_prices = self._get_current_prices(market_data)
            
            for symbol, position in self.open_positions.items():
                if symbol not in current_prices:
                    continue
                
                current_price = current_prices[symbol]
                position.update_current_price(current_price)
                
                # Check stop loss | 檢查止損
                if position.stop_loss is not None:
                    if (position.side == 'long' and current_price <= position.stop_loss) or \
                       (position.side == 'short' and current_price >= position.stop_loss):
                        exit_signals.append({
                            'symbol': symbol,
                            'position': position,
                            'exit_reason': 'stop_loss',
                            'exit_price': current_price,
                            'pnl': position.unrealized_pnl
                        })
                        continue
                
                # Check take profit | 檢查止盈
                if position.take_profit is not None:
                    if (position.side == 'long' and current_price >= position.take_profit) or \
                       (position.side == 'short' and current_price <= position.take_profit):
                        exit_signals.append({
                            'symbol': symbol,
                            'position': position,
                            'exit_reason': 'take_profit',
                            'exit_price': current_price,
                            'pnl': position.unrealized_pnl
                        })
                        continue
                
                # Check trailing stop | 檢查移動止損
                if position.trailing_stop is not None:
                    if (position.side == 'long' and current_price <= position.trailing_stop) or \
                       (position.side == 'short' and current_price >= position.trailing_stop):
                        exit_signals.append({
                            'symbol': symbol,
                            'position': position,
                            'exit_reason': 'trailing_stop',
                            'exit_price': current_price,
                            'pnl': position.unrealized_pnl
                        })
                        continue
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
        
        return exit_signals
    
    def add_position(self, symbol: str, side: str, size: float, entry_price: float,
                    stop_loss: float, take_profit: float, signal_confidence: float = 0.0):
        """
        Add a new position to tracking | 添加新倉位到追蹤
        
        Args:
            symbol: Trading symbol | 交易品種
            side: Position side ('long' or 'short') | 倉位方向
            size: Position size | 倉位大小
            entry_price: Entry price | 入場價格
            stop_loss: Stop loss level | 止損水平
            take_profit: Take profit level | 止盈水平
            signal_confidence: Signal confidence score | 信號信心評分
        """
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(),
            signal_confidence=signal_confidence
        )
        
        self.open_positions[symbol] = position
        self.daily_trade_count += 1
        
        logger.info(f"Added {side} position for {symbol}: size={size:.2f}, "
                   f"entry={entry_price:.5f}, SL={stop_loss:.5f}, TP={take_profit:.5f}")
    
    def close_position(self, symbol: str, exit_price: float, exit_reason: str = "manual"):
        """
        Close a position and update performance metrics | 關閉倉位並更新績效指標
        
        Args:
            symbol: Trading symbol | 交易品種
            exit_price: Exit price | 離場價格
            exit_reason: Reason for exit | 離場原因
        """
        if symbol not in self.open_positions:
            logger.warning(f"Attempted to close non-existent position: {symbol}")
            return
        
        position = self.open_positions[symbol]
        position.update_current_price(exit_price)
        
        # Calculate final PnL | 計算最終盈虧
        final_pnl = position.unrealized_pnl
        
        # Update performance metrics | 更新績效指標
        self.total_pnl += final_pnl
        self.current_balance += final_pnl
        self.daily_pnl += final_pnl
        
        # Record trade | 記錄交易
        trade_record = {
            'symbol': symbol,
            'side': position.side,
            'size': position.size,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'exit_reason': exit_reason,
            'pnl': final_pnl,
            'signal_confidence': position.signal_confidence,
            'duration_minutes': (datetime.now() - position.entry_time).total_seconds() / 60
        }
        
        self.trade_history.append(trade_record)
        
        # Update performance statistics | 更新績效統計
        self._update_performance_stats()
        
        # Remove from open positions | 從開倉倉位中移除
        del self.open_positions[symbol]
        
        logger.info(f"Closed {position.side} position for {symbol}: "
                   f"exit={exit_price:.5f}, PnL=${final_pnl:.2f}, reason={exit_reason}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive risk management summary | 獲取綜合風險管理摘要
        
        Returns:
            Risk summary dictionary | 風險摘要字典
        """
        portfolio_value = self.current_balance
        total_exposure = sum(pos.size * pos.current_price for pos in self.open_positions.values())
        
        return {
            'account_summary': {
                'current_balance': self.current_balance,
                'initial_balance': self.initial_balance,
                'total_pnl': self.total_pnl,
                'total_return_pct': (self.current_balance / self.initial_balance - 1) * 100,
                'max_drawdown': self.max_drawdown,
                'daily_pnl': self.daily_pnl
            },
            'position_summary': {
                'open_positions': len(self.open_positions),
                'max_allowed_positions': self.risk_parameters.max_open_positions,
                'total_exposure': total_exposure,
                'portfolio_utilization_pct': (total_exposure / portfolio_value) * 100 if portfolio_value > 0 else 0
            },
            'daily_limits': {
                'daily_trades': self.daily_trade_count,
                'max_daily_trades': self.risk_parameters.max_daily_trades,
                'daily_pnl': self.daily_pnl,
                'daily_loss_limit': self.risk_parameters.daily_loss_limit * portfolio_value
            },
            'performance_metrics': {
                'win_rate': self.win_rate,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'profit_factor': self.profit_factor,
                'total_trades': len(self.trade_history)
            },
            'risk_parameters': self.risk_parameters.to_dict(),
            'emergency_stop': self.emergency_stop,
            'risk_violations': self.risk_violations[-10:] if self.risk_violations else []
        }
    
    # Private helper methods | 私有輔助方法
    
    def _check_basic_constraints(self, signal: TradingSignal, risk_metrics: RiskMetrics) -> bool:
        """Check basic trading constraints | 檢查基本交易約束"""
        # Check emergency stop | 檢查緊急停止
        if self.emergency_stop:
            risk_metrics.risk_warnings.append("Emergency stop activated")
            return False
        
        # Check signal confidence | 檢查信號信心
        if signal.confidence < self.risk_parameters.min_signal_confidence:
            risk_metrics.risk_warnings.append(f"Signal confidence too low: {signal.confidence:.2f}")
            return False
        
        # Check maximum positions | 檢查最大倉位數
        if len(self.open_positions) >= self.risk_parameters.max_open_positions:
            risk_metrics.risk_warnings.append("Maximum open positions reached")
            return False
        
        # Check daily trade limit | 檢查每日交易限制
        if self.daily_trade_count >= self.risk_parameters.max_daily_trades:
            risk_metrics.risk_warnings.append("Daily trade limit reached")
            return False
        
        # Check daily loss limit | 檢查每日損失限制
        daily_loss_limit = self.risk_parameters.daily_loss_limit * self.current_balance
        if self.daily_pnl <= -daily_loss_limit:
            risk_metrics.risk_warnings.append("Daily loss limit exceeded")
            return False
        
        # Check maximum drawdown | 檢查最大回撤
        if abs(self.max_drawdown) >= self.risk_parameters.max_drawdown_limit:
            risk_metrics.risk_warnings.append("Maximum drawdown limit exceeded")
            self.emergency_stop = True
            return False
        
        return True
    
    def _calculate_portfolio_risk(self) -> float:
        """Calculate current portfolio risk exposure | 計算當前投資組合風險暴露"""
        if not self.open_positions:
            return 0.0
        
        total_risk = 0.0
        for position in self.open_positions.values():
            if position.stop_loss is not None:
                risk_amount = abs(position.entry_price - position.stop_loss) * position.size
                risk_pct = risk_amount / self.current_balance
                total_risk += risk_pct
        
        return total_risk
    
    def _calculate_confidence_adjustment(self, signal: TradingSignal) -> float:
        """Calculate position size adjustment based on signal confidence | 根據信號信心計算倉位大小調整"""
        base_factor = self.risk_parameters.confidence_scaling_factor
        confidence_boost = (signal.confidence - 0.5) * base_factor
        return max(0.5, min(1.5, 1.0 + confidence_boost))  # Clamp between 0.5x and 1.5x
    
    def _make_risk_decision(self, risk_metrics: RiskMetrics, signal: TradingSignal) -> bool:
        """Make final risk approval decision | 做出最終風險批准決定"""
        # Check risk-reward ratio | 檢查風險收益比
        if risk_metrics.risk_reward_ratio < self.risk_parameters.min_risk_reward_ratio:
            risk_metrics.risk_warnings.append(f"Poor risk-reward ratio: {risk_metrics.risk_reward_ratio:.2f}")
            return False
        
        # Check portfolio risk | 檢查投資組合風險
        if risk_metrics.portfolio_risk >= self.risk_parameters.max_portfolio_risk:
            risk_metrics.risk_warnings.append(f"Portfolio risk too high: {risk_metrics.portfolio_risk:.2f}")
            return False
        
        # All checks passed | 所有檢查通過
        return True
    
    def _calculate_kelly_position_size(self, signal: TradingSignal, account_balance: float) -> float:
        """Calculate Kelly Criterion position size | 計算凱利公式倉位大小"""
        if not self.trade_history or self.win_rate == 0:
            return account_balance * 0.01  # Conservative default
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = loss probability
        win_prob = self.win_rate
        loss_prob = 1 - win_prob
        
        if self.avg_loss == 0 or win_prob == 0:
            return account_balance * 0.01
        
        avg_win_abs = abs(self.avg_win)
        avg_loss_abs = abs(self.avg_loss)
        
        odds = avg_win_abs / avg_loss_abs if avg_loss_abs > 0 else 1.0
        
        kelly_fraction = (odds * win_prob - loss_prob) / odds
        kelly_fraction = max(0.0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Apply confidence scaling | 應用信心縮放
        kelly_fraction *= signal.confidence
        
        return account_balance * kelly_fraction
    
    def _calculate_volatility_adjusted_size(self, signal: TradingSignal, 
                                          market_data: pd.DataFrame, account_balance: float) -> float:
        """Calculate volatility-adjusted position size | 計算波動性調整的倉位大小"""
        try:
            # Calculate recent volatility | 計算近期波動性
            returns = market_data['Close'].pct_change().dropna()
            if len(returns) < 20:
                return account_balance * self.risk_parameters.max_position_size
            
            recent_vol = returns.rolling(20).std().iloc[-1]
            historical_vol = returns.std()
            
            if historical_vol == 0:
                vol_adjustment = 1.0
            else:
                vol_adjustment = historical_vol / recent_vol if recent_vol > 0 else 1.0
            
            # Scale base position by volatility | 按波動性縮放基礎倉位
            base_size = account_balance * self.risk_parameters.max_position_size
            adjusted_size = base_size * vol_adjustment
            
            return np.clip(adjusted_size, 
                          account_balance * self.risk_parameters.min_position_size,
                          account_balance * self.risk_parameters.max_position_size)
            
        except Exception as e:
            logger.warning(f"Error in volatility adjustment: {e}")
            return account_balance * self.risk_parameters.max_position_size
    
    def _calculate_atr_based_size(self, signal: TradingSignal, 
                                 market_data: pd.DataFrame, account_balance: float) -> float:
        """Calculate ATR-based position size | 計算基於ATR的倉位大小"""
        try:
            atr = self._calculate_atr(market_data)
            if atr == 0:
                return account_balance * self.risk_parameters.max_position_size
            
            # Position size based on fixed risk amount and ATR | 基於固定風險金額和ATR的倉位大小
            risk_amount = account_balance * self.risk_parameters.max_risk_per_trade
            atr_multiplier = self.risk_parameters.stop_loss_atr_multiplier
            
            position_size = risk_amount / (atr * atr_multiplier)
            
            return np.clip(position_size,
                          account_balance * self.risk_parameters.min_position_size,
                          account_balance * self.risk_parameters.max_position_size)
            
        except Exception as e:
            logger.warning(f"Error in ATR-based sizing: {e}")
            return account_balance * self.risk_parameters.max_position_size
    
    def _calculate_confidence_weighted_size(self, signal: TradingSignal, account_balance: float) -> float:
        """Calculate confidence-weighted position size | 計算信心權重的倉位大小"""
        base_size = account_balance * self.risk_parameters.max_position_size
        
        # Weight by signal confidence and strength | 按信號信心和強度加權
        confidence_weight = signal.confidence
        strength_weight = signal.strength
        
        combined_weight = (confidence_weight * 0.7) + (strength_weight * 0.3)
        
        return base_size * combined_weight
    
    def _calculate_atr(self, market_data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range | 計算平均真實範圍"""
        try:
            if len(market_data) < period + 1:
                return 0.0
            
            high = market_data['High']
            low = market_data['Low'] 
            close = market_data['Close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating ATR: {e}")
            return 0.0
    
    def _get_current_prices(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Extract current prices from market data | 從市場數據中提取當前價格"""
        current_prices = {}
        
        try:
            if 'Symbol' in market_data.columns:
                # Multi-symbol data | 多品種數據
                for symbol in market_data['Symbol'].unique():
                    symbol_data = market_data[market_data['Symbol'] == symbol]
                    if not symbol_data.empty:
                        current_prices[symbol] = symbol_data['Close'].iloc[-1]
            else:
                # Single symbol data - use index or default | 單品種數據
                if not market_data.empty:
                    symbol = market_data.index.name if market_data.index.name else 'DEFAULT'
                    current_prices[symbol] = market_data['Close'].iloc[-1]
        
        except Exception as e:
            logger.warning(f"Error extracting current prices: {e}")
        
        return current_prices
    
    def _update_performance_stats(self):
        """Update performance statistics from trade history | 從交易歷史更新績效統計"""
        if not self.trade_history:
            return
        
        # Calculate win rate | 計算勝率
        winning_trades = [t for t in self.trade_history if t['pnl'] > 0]
        losing_trades = [t for t in self.trade_history if t['pnl'] < 0]
        
        self.win_rate = len(winning_trades) / len(self.trade_history)
        
        # Calculate average win/loss | 計算平均盈虧
        if winning_trades:
            self.avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades)
        else:
            self.avg_win = 0.0
        
        if losing_trades:
            self.avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades)
        else:
            self.avg_loss = 0.0
        
        # Calculate profit factor | 計算獲利因子
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        
        if gross_loss > 0:
            self.profit_factor = gross_profit / gross_loss
        else:
            self.profit_factor = float('inf') if gross_profit > 0 else 0.0
        
        # Update drawdown | 更新回撤
        self.equity_curve.append(self.current_balance)
        if len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            if peak > self.peak_equity:
                self.peak_equity = peak
            current_drawdown = (self.current_balance - self.peak_equity) / self.peak_equity
            if current_drawdown < self.max_drawdown:
                self.max_drawdown = current_drawdown


def create_risk_manager_preset(risk_level: RiskLevel, account_balance: float = 100000.0) -> AdvancedRiskManager:
    """
    Create risk manager with preset configurations | 創建預設配置的風險管理器
    
    Args:
        risk_level: Risk level preset | 風險等級預設
        account_balance: Initial account balance | 初始賬戶餘額
        
    Returns:
        Configured risk manager | 配置好的風險管理器
    """
    if risk_level == RiskLevel.CONSERVATIVE:
        params = RiskParameters(
            max_position_size=0.01,  # 1% max position
            min_position_size=0.002,  # 0.2% min position
            max_portfolio_risk=0.05,  # 5% max portfolio risk
            max_open_positions=3,
            stop_loss_atr_multiplier=1.5,
            take_profit_atr_multiplier=3.0,
            min_risk_reward_ratio=2.0,
            max_risk_per_trade=0.01,
            min_signal_confidence=0.75
        )
    elif risk_level == RiskLevel.MODERATE:
        params = RiskParameters(
            max_position_size=0.02,  # 2% max position
            min_position_size=0.005,  # 0.5% min position
            max_portfolio_risk=0.10,  # 10% max portfolio risk
            max_open_positions=5,
            stop_loss_atr_multiplier=2.0,
            take_profit_atr_multiplier=3.0,
            min_risk_reward_ratio=1.5,
            max_risk_per_trade=0.02,
            min_signal_confidence=0.6
        )
    elif risk_level == RiskLevel.AGGRESSIVE:
        params = RiskParameters(
            max_position_size=0.05,  # 5% max position
            min_position_size=0.01,  # 1% min position
            max_portfolio_risk=0.20,  # 20% max portfolio risk
            max_open_positions=8,
            stop_loss_atr_multiplier=2.5,
            take_profit_atr_multiplier=4.0,
            min_risk_reward_ratio=1.2,
            max_risk_per_trade=0.05,
            min_signal_confidence=0.5
        )
    else:  # CUSTOM - use defaults
        params = RiskParameters()
    
    return AdvancedRiskManager(params, account_balance)


# Backward compatibility aliases | 向後兼容別名
RiskManager = AdvancedRiskManager  # Main risk manager class alias