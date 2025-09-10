"""
Position Manager | 倉位管理器
==========================

Advanced position management system that tracks, monitors, and manages
all trading positions with real-time updates and risk controls.
高級倉位管理系統，可追蹤、監控和管理所有交易倉位，並提供實時更新和風險控制。
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from brokers.ig_markets import IGMarketsConnector
from core.risk_manager import Position, RiskLevel

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Position status enumeration | 倉位狀態枚舉"""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"
    ERROR = "error"


@dataclass
class ManagedPosition:
    """
    Managed position with advanced tracking | 具有高級追蹤的管理倉位
    """
    position_id: str
    symbol: str
    epic: str
    side: str  # 'long' or 'short'
    original_size: float
    current_size: float
    entry_price: float
    current_price: float = 0.0
    
    # Risk management | 風險管理
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    
    # Tracking information | 追蹤信息
    deal_id: Optional[str] = None
    open_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    status: PositionStatus = PositionStatus.OPEN
    
    # P&L tracking | 盈虧追蹤
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_pnl: float = 0.0
    
    # Performance metrics | 績效指標
    max_profit: float = 0.0
    max_loss: float = 0.0
    duration_minutes: float = 0.0
    
    # Metadata | 元數據
    strategy_signal: Optional[str] = None
    confidence: float = 0.0
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def update_price(self, new_price: float) -> None:
        """Update position with new market price | 使用新市場價格更新倉位"""
        self.current_price = new_price
        self.last_update = datetime.now()
        
        # Calculate unrealized P&L | 計算未實現盈虧
        if self.side == 'long':
            self.unrealized_pnl = (new_price - self.entry_price) * self.current_size
        else:  # short
            self.unrealized_pnl = (self.entry_price - new_price) * self.current_size
        
        # Update total P&L | 更新總盈虧
        self.total_pnl = self.realized_pnl + self.unrealized_pnl
        
        # Track max profit/loss | 追蹤最大盈利/虧損
        if self.total_pnl > self.max_profit:
            self.max_profit = self.total_pnl
        if self.total_pnl < self.max_loss:
            self.max_loss = self.total_pnl
        
        # Update duration | 更新持續時間
        self.duration_minutes = (self.last_update - self.open_time).total_seconds() / 60
    
    def check_exit_conditions(self) -> List[str]:
        """Check if position should be closed | 檢查倉位是否應該關閉"""
        exit_reasons = []
        
        if self.stop_loss and self._should_stop_loss():
            exit_reasons.append("stop_loss")
        
        if self.take_profit and self._should_take_profit():
            exit_reasons.append("take_profit")
        
        if self.trailing_stop and self._should_trailing_stop():
            exit_reasons.append("trailing_stop")
        
        return exit_reasons
    
    def _should_stop_loss(self) -> bool:
        """Check stop loss condition | 檢查止損條件"""
        if self.side == 'long':
            return self.current_price <= self.stop_loss
        else:  # short
            return self.current_price >= self.stop_loss
    
    def _should_take_profit(self) -> bool:
        """Check take profit condition | 檢查止盈條件"""
        if self.side == 'long':
            return self.current_price >= self.take_profit
        else:  # short
            return self.current_price <= self.take_profit
    
    def _should_trailing_stop(self) -> bool:
        """Check trailing stop condition | 檢查移動止損條件"""
        # Simplified trailing stop logic | 簡化的移動止損邏輯
        if self.side == 'long':
            trailing_level = self.current_price - self.trailing_stop
            return self.current_price <= trailing_level
        else:  # short
            trailing_level = self.current_price + self.trailing_stop
            return self.current_price >= trailing_level
    
    def partial_close(self, close_size: float, close_price: float) -> None:
        """Execute partial position close | 執行部分倉位關閉"""
        if close_size >= self.current_size:
            # Full close | 全部關閉
            self.realized_pnl = self.total_pnl
            self.current_size = 0.0
            self.status = PositionStatus.CLOSED
        else:
            # Partial close | 部分關閉
            close_ratio = close_size / self.current_size
            self.realized_pnl += self.unrealized_pnl * close_ratio
            self.current_size -= close_size
            self.status = PositionStatus.PARTIALLY_CLOSED
            
            # Recalculate unrealized P&L for remaining position | 重新計算剩餘倉位的未實現盈虧
            if self.side == 'long':
                self.unrealized_pnl = (self.current_price - self.entry_price) * self.current_size
            else:
                self.unrealized_pnl = (self.entry_price - self.current_price) * self.current_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary | 轉換為字典"""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'epic': self.epic,
            'side': self.side,
            'original_size': self.original_size,
            'current_size': self.current_size,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'deal_id': self.deal_id,
            'open_time': self.open_time.isoformat(),
            'last_update': self.last_update.isoformat(),
            'status': self.status.value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'max_profit': self.max_profit,
            'max_loss': self.max_loss,
            'duration_minutes': self.duration_minutes,
            'strategy_signal': self.strategy_signal,
            'confidence': self.confidence,
            'tags': self.tags,
            'notes': self.notes
        }


class PositionManager:
    """
    Position Manager - Advanced position tracking and management
    倉位管理器 - 高級倉位追蹤和管理
    
    Provides comprehensive position management including:
    - Real-time position tracking | 實時倉位追蹤
    - Risk monitoring and alerts | 風險監控和警報
    - Automated exit condition checking | 自動離場條件檢查
    - Performance analytics | 績效分析
    - Position lifecycle management | 倉位生命週期管理
    
    提供全面的倉位管理。
    """
    
    def __init__(self, ig_connector: Optional[IGMarketsConnector] = None,
                 max_positions: int = 10,
                 max_risk_per_position: float = 0.02,
                 max_total_risk: float = 0.06):
        """
        Initialize Position Manager | 初始化倉位管理器
        
        Args:
            ig_connector: IG Markets API connector (None for demo mode) | IG Markets API連接器（演示模式為None）
            max_positions: Maximum number of open positions | 最大開倉數量
            max_risk_per_position: Maximum risk per position (as fraction of account) | 每個倉位的最大風險
            max_total_risk: Maximum total portfolio risk | 最大總投資組合風險
        """
        self.ig_connector = ig_connector
        self.max_positions = max_positions
        self.max_risk_per_position = max_risk_per_position
        self.max_total_risk = max_total_risk
        
        # Position tracking | 倉位追蹤
        self.positions: Dict[str, ManagedPosition] = {}
        self.closed_positions: Dict[str, ManagedPosition] = {}
        self.position_history: List[ManagedPosition] = []
        
        # Risk monitoring | 風險監控
        self.total_exposure = 0.0
        self.total_unrealized_pnl = 0.0
        self.account_balance = 100000.0  # Will be updated from IG account
        
        # Performance tracking | 績效追蹤
        self.total_positions_opened = 0
        self.total_positions_closed = 0
        self.total_realized_pnl = 0.0
        self.win_count = 0
        self.loss_count = 0
        
        logger.info("Position Manager initialized")
    
    async def add_position(self, symbol: str, side: str, size: float, 
                          entry_price: float, **kwargs) -> str:
        """
        Add new position to management | 添加新倉位到管理中
        
        Args:
            symbol: Trading symbol | 交易品種
            side: Position side ('long' or 'short') | 倉位方向
            size: Position size | 倉位大小
            entry_price: Entry price | 入場價格
            **kwargs: Additional position parameters | 額外倉位參數
            
        Returns:
            str: Position ID | 倉位ID
        """
        try:
            # Check position limits | 檢查倉位限制
            if len(self.positions) >= self.max_positions:
                raise ValueError(f"Maximum positions limit reached ({self.max_positions})")
            
            # Create position ID | 創建倉位ID
            position_id = str(uuid.uuid4())
            
            # Map symbol to epic | 將品種映射到epic
            epic = self._map_symbol_to_epic(symbol)
            
            # Create managed position | 創建管理倉位
            position = ManagedPosition(
                position_id=position_id,
                symbol=symbol,
                epic=epic,
                side=side,
                original_size=size,
                current_size=size,
                entry_price=entry_price,
                current_price=entry_price,
                **kwargs
            )
            
            # Add to tracking | 添加到追蹤中
            self.positions[position_id] = position
            self.total_positions_opened += 1
            
            # Update risk metrics | 更新風險指標
            await self._update_risk_metrics()
            
            logger.info(f"✅ Position added: {position_id} ({side} {size} {symbol} @ {entry_price})")
            return position_id
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            raise
    
    async def update_positions(self) -> None:
        """Update all positions with current market data | 使用當前市場數據更新所有倉位"""
        try:
            if not self.positions:
                return
            
            # Get current market prices | 獲取當前市場價格
            symbols = list(set(pos.symbol for pos in self.positions.values()))
            
            for symbol in symbols:
                try:
                    # Skip market data updates in demo mode | 演示模式跳過市場數據更新
                    if self.ig_connector is None:
                        continue
                        
                    market_data = await self.ig_connector.get_market_data(
                        self._map_symbol_to_epic(symbol)
                    )
                    
                    if market_data and 'bid' in market_data:
                        current_price = market_data['bid']  # Use bid for current price
                        
                        # Update all positions for this symbol | 更新此品種的所有倉位
                        for position in self.positions.values():
                            if position.symbol == symbol:
                                position.update_price(current_price)
                
                except Exception as e:
                    logger.warning(f"Error updating price for {symbol}: {e}")
            
            # Update aggregate risk metrics | 更新聚合風險指標
            await self._update_risk_metrics()
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
    
    async def check_exit_conditions(self) -> List[Dict[str, Any]]:
        """Check exit conditions for all positions | 檢查所有倉位的離場條件"""
        exit_signals = []
        
        try:
            for position_id, position in list(self.positions.items()):
                exit_reasons = position.check_exit_conditions()
                
                if exit_reasons:
                    exit_signal = {
                        'position_id': position_id,
                        'symbol': position.symbol,
                        'epic': position.epic,
                        'exit_reasons': exit_reasons,
                        'current_price': position.current_price,
                        'size': position.current_size,
                        'unrealized_pnl': position.unrealized_pnl
                    }
                    exit_signals.append(exit_signal)
                    
                    logger.info(f"🚨 Exit condition detected for {position_id}: {exit_reasons}")
            
            return exit_signals
            
        except Exception as e:
            logger.error(f"Error checking exit conditions: {e}")
            return []
    
    async def close_position(self, position_id: str, 
                           close_size: Optional[float] = None,
                           reason: str = "manual") -> bool:
        """
        Close position (fully or partially) | 關閉倉位（全部或部分）
        
        Args:
            position_id: Position ID to close | 要關閉的倉位ID
            close_size: Size to close (None for full close) | 要關閉的大小（None為全部關閉）
            reason: Reason for closing | 關閉原因
            
        Returns:
            bool: True if close was successful | 關閉成功時返回True
        """
        try:
            if position_id not in self.positions:
                logger.warning(f"Position {position_id} not found")
                return False
            
            position = self.positions[position_id]
            position.status = PositionStatus.CLOSING
            
            # Determine close size | 確定關閉大小
            actual_close_size = close_size or position.current_size
            
            logger.info(f"🔄 Closing position {position_id}: {actual_close_size} of {position.current_size}")
            
            # Execute close via IG API | 通過IG API執行關閉
            if position.deal_id and self.ig_connector is not None:
                close_result = await self.ig_connector.close_position(
                    position.deal_id, 
                    actual_close_size
                )
                
                if not close_result.get('success'):
                    logger.error(f"IG API close failed: {close_result.get('reason')}")
                    position.status = PositionStatus.ERROR
                    return False
            
            # Update position | 更新倉位
            position.partial_close(actual_close_size, position.current_price)
            position.notes += f" | Closed: {reason}"
            
            # Move to closed positions if fully closed | 如果完全關閉則移動到已關閉倉位
            if position.status == PositionStatus.CLOSED:
                self.closed_positions[position_id] = position
                self.position_history.append(position)
                del self.positions[position_id]
                
                # Update statistics | 更新統計
                self.total_positions_closed += 1
                self.total_realized_pnl += position.realized_pnl
                
                if position.realized_pnl > 0:
                    self.win_count += 1
                else:
                    self.loss_count += 1
                
                logger.info(f"✅ Position {position_id} fully closed (PnL: ${position.realized_pnl:.2f})")
            else:
                logger.info(f"📊 Position {position_id} partially closed ({position.current_size} remaining)")
            
            # Update risk metrics | 更新風險指標
            await self._update_risk_metrics()
            
            return True
            
        except Exception as e:
            logger.error(f"Error closing position {position_id}: {e}")
            return False
    
    async def close_all_positions(self, reason: str = "close_all") -> int:
        """
        Close all open positions | 關閉所有開倉
        
        Args:
            reason: Reason for closing all positions | 關閉所有倉位的原因
            
        Returns:
            int: Number of positions successfully closed | 成功關閉的倉位數量
        """
        closed_count = 0
        
        try:
            position_ids = list(self.positions.keys())
            logger.info(f"🔄 Closing {len(position_ids)} positions...")
            
            for position_id in position_ids:
                if await self.close_position(position_id, reason=reason):
                    closed_count += 1
            
            logger.info(f"✅ Closed {closed_count} of {len(position_ids)} positions")
            return closed_count
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return closed_count
    
    def get_position(self, position_id: str) -> Optional[Dict[str, Any]]:
        """Get specific position details | 獲取特定倉位詳情"""
        if position_id in self.positions:
            return self.positions[position_id].to_dict()
        elif position_id in self.closed_positions:
            return self.closed_positions[position_id].to_dict()
        return None
    
    def get_all_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all positions (open and closed) | 獲取所有倉位（開倉和已關閉）"""
        return {
            'open_positions': [pos.to_dict() for pos in self.positions.values()],
            'closed_positions': [pos.to_dict() for pos in self.closed_positions.values()]
        }
    
    def get_positions_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all positions for specific symbol | 獲取特定品種的所有倉位"""
        symbol_positions = []
        
        for position in self.positions.values():
            if position.symbol == symbol:
                symbol_positions.append(position.to_dict())
        
        return symbol_positions
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary | 獲取綜合投資組合摘要"""
        return {
            'position_counts': {
                'open_positions': len(self.positions),
                'closed_positions': len(self.closed_positions),
                'total_opened': self.total_positions_opened,
                'total_closed': self.total_positions_closed
            },
            'risk_metrics': {
                'total_exposure': self.total_exposure,
                'total_unrealized_pnl': self.total_unrealized_pnl,
                'account_balance': self.account_balance,
                'risk_utilization': self.total_exposure / self.account_balance if self.account_balance > 0 else 0
            },
            'performance_metrics': {
                'total_realized_pnl': self.total_realized_pnl,
                'win_count': self.win_count,
                'loss_count': self.loss_count,
                'win_rate': self.win_count / max(1, self.win_count + self.loss_count),
                'avg_win': self._calculate_avg_win(),
                'avg_loss': self._calculate_avg_loss()
            },
            'position_breakdown': self._get_position_breakdown()
        }
    
    async def _update_risk_metrics(self) -> None:
        """Update aggregate risk metrics | 更新聚合風險指標"""
        try:
            self.total_exposure = sum(
                abs(pos.current_size * pos.current_price) 
                for pos in self.positions.values()
            )
            
            self.total_unrealized_pnl = sum(
                pos.unrealized_pnl for pos in self.positions.values()
            )
            
            # Update account balance from IG if possible | 如果可能，從IG更新帳戶餘額
            try:
                if self.ig_connector is not None:
                    status = self.ig_connector.get_status()
                    account_info = status.get('account_info', {})
                    if 'balance' in account_info:
                        self.account_balance = float(account_info['balance'])
            except:
                pass  # Keep existing balance if update fails
                
        except Exception as e:
            logger.error(f"Error updating risk metrics: {e}")
    
    def _map_symbol_to_epic(self, symbol: str) -> str:
        """Map symbol to IG epic code | 將品種映射到IG epic代碼"""
        symbol_mapping = {
            'EURUSD=X': 'CS.D.EURUSD.MINI.IP',
            'USDJPY=X': 'CS.D.USDJPY.MINI.IP',
            'GBPUSD=X': 'CS.D.GBPUSD.MINI.IP',
            'USDCHF=X': 'CS.D.USDCHF.MINI.IP',
            'AUDUSD=X': 'CS.D.AUDUSD.MINI.IP'
        }
        return symbol_mapping.get(symbol, symbol)
    
    def _calculate_avg_win(self) -> float:
        """Calculate average winning trade | 計算平均獲利交易"""
        winning_trades = [
            pos.realized_pnl for pos in self.position_history 
            if pos.realized_pnl > 0
        ]
        return sum(winning_trades) / len(winning_trades) if winning_trades else 0.0
    
    def _calculate_avg_loss(self) -> float:
        """Calculate average losing trade | 計算平均虧損交易"""
        losing_trades = [
            abs(pos.realized_pnl) for pos in self.position_history 
            if pos.realized_pnl < 0
        ]
        return sum(losing_trades) / len(losing_trades) if losing_trades else 0.0
    
    def _get_position_breakdown(self) -> Dict[str, Any]:
        """Get position breakdown by symbol and side | 按品種和方向獲取倉位分解"""
        breakdown = {
            'by_symbol': {},
            'by_side': {'long': 0, 'short': 0}
        }
        
        for position in self.positions.values():
            # By symbol | 按品種
            if position.symbol not in breakdown['by_symbol']:
                breakdown['by_symbol'][position.symbol] = {
                    'count': 0, 'total_size': 0.0, 'total_pnl': 0.0
                }
            
            symbol_data = breakdown['by_symbol'][position.symbol]
            symbol_data['count'] += 1
            symbol_data['total_size'] += position.current_size
            symbol_data['total_pnl'] += position.unrealized_pnl
            
            # By side | 按方向
            breakdown['by_side'][position.side] += 1
        
        return breakdown


# Factory function | 工廠函數
def create_position_manager(ig_connector: IGMarketsConnector, **kwargs) -> PositionManager:
    """
    Create a Position Manager instance | 創建倉位管理器實例
    
    Args:
        ig_connector: IG Markets API connector | IG Markets API連接器
        **kwargs: Additional configuration parameters | 額外配置參數
        
    Returns:
        PositionManager: Configured position manager instance | 配置好的倉位管理器實例
    """
    return PositionManager(ig_connector, **kwargs)