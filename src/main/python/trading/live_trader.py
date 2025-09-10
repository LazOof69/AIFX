"""
Live Trader | 實時交易器
=======================

Real-time trading execution engine that interfaces with IG Markets API
for live forex trading with AI-enhanced decision making.
與IG Markets API接口的實時交易執行引擎，用於AI增強決策的實時外匯交易。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid

from brokers.ig_markets import IGMarketsConnector, IGOrder, OrderType, OrderDirection
from core.trading_strategy import TradingDecision
from core.risk_manager import Position

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    """Trade execution status | 交易執行狀態"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class LiveTrade:
    """
    Live trade tracking record | 實時交易追蹤記錄
    """
    trade_id: str
    symbol: str
    decision: TradingDecision
    order: Optional[IGOrder] = None
    deal_reference: Optional[str] = None
    status: TradeStatus = TradeStatus.PENDING
    submit_time: Optional[datetime] = None
    fill_time: Optional[datetime] = None
    fill_price: Optional[float] = None
    fill_size: Optional[float] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.submit_time is None:
            self.submit_time = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary | 轉換為字典"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'status': self.status.value,
            'submit_time': self.submit_time.isoformat() if self.submit_time else None,
            'fill_time': self.fill_time.isoformat() if self.fill_time else None,
            'fill_price': self.fill_price,
            'fill_size': self.fill_size,
            'deal_reference': self.deal_reference,
            'error_message': self.error_message,
            'decision_action': self.decision.action,
            'decision_confidence': self.decision.confidence
        }


class LiveTrader:
    """
    Live Trader - Real-time trading execution engine
    實時交易器 - 實時交易執行引擎
    
    Handles the execution of trading decisions in live markets:
    - Converts trading decisions to IG Market orders | 將交易決策轉換為IG Market訂單
    - Manages order lifecycle and status tracking | 管理訂單生命週期和狀態追蹤
    - Implements risk controls and position sizing | 實施風險控制和倉位大小
    - Monitors fills and position updates | 監控成交和倉位更新
    
    在實時市場中處理交易決策的執行。
    """
    
    def __init__(self, ig_connector: IGMarketsConnector, 
                 max_position_size: float = 10000.0,
                 max_orders_per_minute: int = 10):
        """
        Initialize Live Trader | 初始化實時交易器
        
        Args:
            ig_connector: IG Markets API connector | IG Markets API連接器
            max_position_size: Maximum position size per trade | 每筆交易的最大倉位
            max_orders_per_minute: Rate limit for order submission | 訂單提交的速率限制
        """
        self.ig_connector = ig_connector
        self.max_position_size = max_position_size
        self.max_orders_per_minute = max_orders_per_minute
        
        # Trade tracking | 交易追蹤
        self.active_trades: Dict[str, LiveTrade] = {}
        self.completed_trades: Dict[str, LiveTrade] = {}
        self.trade_history: List[LiveTrade] = []
        
        # Rate limiting | 速率限制
        self.recent_orders: List[datetime] = []
        
        # Performance metrics | 績效指標
        self.total_orders_submitted = 0
        self.successful_fills = 0
        self.failed_orders = 0
        self.total_pnl = 0.0
        
        logger.info("Live Trader initialized")
    
    async def execute_decision(self, decision: TradingDecision) -> str:
        """
        Execute a trading decision | 執行交易決策
        
        Args:
            decision: Trading decision to execute | 要執行的交易決策
            
        Returns:
            str: Trade ID for tracking | 用於追蹤的交易ID
        """
        trade_id = str(uuid.uuid4())
        
        try:
            logger.info(f"🎯 Executing decision: {decision.action} {decision.symbol} "
                       f"(confidence={decision.confidence:.2f})")
            
            # Rate limiting check | 速率限制檢查
            if not self._check_rate_limit():
                error_msg = "Rate limit exceeded - order rejected"
                logger.warning(error_msg)
                await self._record_failed_trade(trade_id, decision, error_msg)
                return trade_id
            
            # Risk validation | 風險驗證
            if not self._validate_risk(decision):
                error_msg = "Risk validation failed - order rejected"
                logger.warning(error_msg)
                await self._record_failed_trade(trade_id, decision, error_msg)
                return trade_id
            
            # Create live trade record | 創建實時交易記錄
            live_trade = LiveTrade(
                trade_id=trade_id,
                symbol=decision.symbol,
                decision=decision
            )
            
            # Convert decision to IG order | 將決策轉換為IG訂單
            ig_order = self._convert_decision_to_order(decision)
            live_trade.order = ig_order
            
            # Submit order to IG Markets | 向IG Markets提交訂單
            result = await self._submit_order(live_trade)
            
            # Track the trade | 追蹤交易
            self.active_trades[trade_id] = live_trade
            
            # Update rate limiting | 更新速率限制
            self._update_rate_limit()
            
            logger.info(f"✅ Trade {trade_id} submitted successfully")
            return trade_id
            
        except Exception as e:
            error_msg = f"Order execution failed: {str(e)}"
            logger.error(error_msg)
            await self._record_failed_trade(trade_id, decision, error_msg)
            return trade_id
    
    async def close_position(self, symbol: str, size: Optional[float] = None) -> str:
        """
        Close an existing position | 關閉現有倉位
        
        Args:
            symbol: Symbol to close | 要關閉的品種
            size: Partial close size (None for full close) | 部分平倉大小（None為全部平倉）
            
        Returns:
            str: Trade ID for the close order | 平倉訂單的交易ID
        """
        trade_id = str(uuid.uuid4())
        
        try:
            logger.info(f"🔄 Closing position for {symbol} (size: {size or 'full'})")
            
            # Get current positions | 獲取當前倉位
            status = self.ig_connector.get_status()
            positions = status.get('positions', {})
            
            if symbol not in positions:
                error_msg = f"No open position found for {symbol}"
                logger.warning(error_msg)
                return trade_id
            
            position = positions[symbol]
            
            # Use IG API to close position | 使用IG API關閉倉位
            close_result = await self.ig_connector.close_position(
                position.get('deal_id', ''), 
                size
            )
            
            if close_result.get('success'):
                logger.info(f"✅ Position {symbol} closed successfully: {close_result.get('deal_reference')}")
            else:
                logger.error(f"❌ Failed to close position {symbol}: {close_result.get('reason')}")
            
            return trade_id
            
        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return trade_id
    
    async def update_trade_status(self) -> None:
        """
        Update status of all active trades | 更新所有活躍交易的狀態
        """
        try:
            for trade_id, live_trade in list(self.active_trades.items()):
                await self._check_trade_status(live_trade)
                
                # Move completed trades | 移動已完成的交易
                if live_trade.status in [TradeStatus.FILLED, TradeStatus.CANCELLED, 
                                       TradeStatus.REJECTED, TradeStatus.FAILED]:
                    self.completed_trades[trade_id] = live_trade
                    self.trade_history.append(live_trade)
                    del self.active_trades[trade_id]
                    
                    logger.debug(f"Trade {trade_id} moved to completed ({live_trade.status.value})")
                    
        except Exception as e:
            logger.error(f"Error updating trade status: {e}")
    
    async def _submit_order(self, live_trade: LiveTrade) -> Dict[str, Any]:
        """Submit order to IG Markets | 向IG Markets提交訂單"""
        try:
            result = await self.ig_connector.place_order(live_trade.order)
            
            if result.get('success'):
                live_trade.status = TradeStatus.SUBMITTED
                live_trade.deal_reference = result.get('deal_reference')
                self.total_orders_submitted += 1
                
                logger.info(f"Order submitted: {live_trade.deal_reference}")
                
            else:
                live_trade.status = TradeStatus.REJECTED
                live_trade.error_message = result.get('reason', 'Unknown error')
                self.failed_orders += 1
                
                logger.warning(f"Order rejected: {live_trade.error_message}")
            
            return result
            
        except Exception as e:
            live_trade.status = TradeStatus.FAILED
            live_trade.error_message = str(e)
            self.failed_orders += 1
            raise
    
    async def _check_trade_status(self, live_trade: LiveTrade) -> None:
        """Check and update individual trade status | 檢查並更新個別交易狀態"""
        try:
            if not live_trade.deal_reference:
                return
                
            # Get deal confirmation from IG | 從IG獲取交易確認
            confirmation = await self.ig_connector.get_deal_confirmation(live_trade.deal_reference)
            
            if 'confirmation' in confirmation:
                deal_status = confirmation['confirmation'].get('dealStatus', 'UNKNOWN')
                
                if deal_status == 'ACCEPTED':
                    live_trade.status = TradeStatus.FILLED
                    live_trade.fill_time = datetime.now()
                    
                    # Extract fill details | 提取成交詳情
                    deal_data = confirmation['confirmation']
                    live_trade.fill_price = deal_data.get('level', 0.0)
                    live_trade.fill_size = deal_data.get('size', 0.0)
                    
                    self.successful_fills += 1
                    logger.info(f"✅ Trade {live_trade.trade_id} filled at {live_trade.fill_price}")
                    
                elif deal_status == 'REJECTED':
                    live_trade.status = TradeStatus.REJECTED
                    live_trade.error_message = deal_data.get('reason', 'Order rejected by broker')
                    
                    logger.warning(f"❌ Trade {live_trade.trade_id} rejected: {live_trade.error_message}")
                    
        except Exception as e:
            logger.error(f"Error checking trade status for {live_trade.trade_id}: {e}")
    
    def _convert_decision_to_order(self, decision: TradingDecision) -> IGOrder:
        """Convert trading decision to IG order | 將交易決策轉換為IG訂單"""
        # Map decision action to order direction | 將決策行動映射到訂單方向
        direction = OrderDirection.BUY if decision.action == 'BUY' else OrderDirection.SELL
        
        # Use market orders for immediate execution | 使用市價單立即執行
        order_type = OrderType.MARKET
        
        # Apply position sizing limits | 應用倉位大小限制
        size = min(decision.position_size, self.max_position_size)
        
        # Map symbol (convert from Yahoo Finance format if needed) | 映射品種（如需要從Yahoo Finance格式轉換）
        epic = self._map_symbol_to_epic(decision.symbol)
        
        return IGOrder(
            order_type=order_type,
            direction=direction,
            epic=epic,
            size=size,
            currency_code="GBP",  # Default currency
            force_open=True,
            stop_loss=decision.stop_loss,
            take_profit=decision.take_profit,
            level=decision.entry_price  # For limit orders
        )
    
    def _map_symbol_to_epic(self, symbol: str) -> str:
        """Map Yahoo Finance symbol to IG epic code | 將Yahoo Finance品種映射到IG epic代碼"""
        symbol_mapping = {
            'EURUSD=X': 'CS.D.EURUSD.MINI.IP',
            'USDJPY=X': 'CS.D.USDJPY.MINI.IP',
            'GBPUSD=X': 'CS.D.GBPUSD.MINI.IP',
            'USDCHF=X': 'CS.D.USDCHF.MINI.IP',
            'AUDUSD=X': 'CS.D.AUDUSD.MINI.IP'
        }
        
        return symbol_mapping.get(symbol, symbol)
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit allows new order | 檢查速率限制是否允許新訂單"""
        now = datetime.now()
        cutoff_time = now - timedelta(minutes=1)
        
        # Remove old entries | 移除舊條目
        self.recent_orders = [order_time for order_time in self.recent_orders 
                             if order_time > cutoff_time]
        
        return len(self.recent_orders) < self.max_orders_per_minute
    
    def _update_rate_limit(self) -> None:
        """Update rate limiting tracker | 更新速率限制追蹤器"""
        self.recent_orders.append(datetime.now())
    
    def _validate_risk(self, decision: TradingDecision) -> bool:
        """Validate risk parameters for the decision | 驗證決策的風險參數"""
        try:
            # Check position size | 檢查倉位大小
            if decision.position_size <= 0 or decision.position_size > self.max_position_size:
                logger.warning(f"Invalid position size: {decision.position_size}")
                return False
            
            # Check confidence threshold | 檢查信心閾值
            if decision.confidence < 0.5:  # Minimum confidence threshold
                logger.warning(f"Low confidence decision: {decision.confidence}")
                return False
            
            # Check if symbol is supported | 檢查是否支持該品種
            epic = self._map_symbol_to_epic(decision.symbol)
            if epic == decision.symbol:  # No mapping found
                logger.warning(f"Unsupported symbol: {decision.symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Risk validation error: {e}")
            return False
    
    async def _record_failed_trade(self, trade_id: str, decision: TradingDecision, 
                                 error_message: str) -> None:
        """Record a failed trade attempt | 記錄失敗的交易嘗試"""
        failed_trade = LiveTrade(
            trade_id=trade_id,
            symbol=decision.symbol,
            decision=decision,
            status=TradeStatus.FAILED,
            error_message=error_message
        )
        
        self.completed_trades[trade_id] = failed_trade
        self.trade_history.append(failed_trade)
        self.failed_orders += 1
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get trader performance summary | 獲取交易器績效摘要"""
        total_trades = len(self.trade_history)
        
        if total_trades == 0:
            return {
                'total_orders': 0,
                'success_rate': 0.0,
                'fill_rate': 0.0,
                'active_trades': len(self.active_trades),
                'total_pnl': 0.0
            }
        
        fill_rate = self.successful_fills / max(1, self.total_orders_submitted)
        success_rate = self.successful_fills / total_trades
        
        return {
            'total_orders': self.total_orders_submitted,
            'successful_fills': self.successful_fills,
            'failed_orders': self.failed_orders,
            'success_rate': success_rate,
            'fill_rate': fill_rate,
            'active_trades': len(self.active_trades),
            'completed_trades': len(self.completed_trades),
            'total_pnl': self.total_pnl
        }
    
    def get_trade_status(self, trade_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific trade | 獲取特定交易的狀態"""
        # Check active trades first | 首先檢查活躍交易
        if trade_id in self.active_trades:
            return self.active_trades[trade_id].to_dict()
        
        # Check completed trades | 檢查已完成交易
        if trade_id in self.completed_trades:
            return self.completed_trades[trade_id].to_dict()
        
        return None
    
    def get_all_active_trades(self) -> List[Dict[str, Any]]:
        """Get all active trades | 獲取所有活躍交易"""
        return [trade.to_dict() for trade in self.active_trades.values()]
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trade history | 獲取最近的交易歷史"""
        recent_trades = sorted(self.trade_history, 
                             key=lambda t: t.submit_time or datetime.min, 
                             reverse=True)
        return [trade.to_dict() for trade in recent_trades[:limit]]


# Factory function for easy instantiation | 便於實例化的工廠函數
def create_live_trader(ig_connector: IGMarketsConnector, **kwargs) -> LiveTrader:
    """
    Create a Live Trader instance | 創建實時交易器實例
    
    Args:
        ig_connector: IG Markets API connector | IG Markets API連接器
        **kwargs: Additional configuration parameters | 額外配置參數
        
    Returns:
        LiveTrader: Configured live trader instance | 配置好的實時交易器實例
    """
    return LiveTrader(ig_connector, **kwargs)