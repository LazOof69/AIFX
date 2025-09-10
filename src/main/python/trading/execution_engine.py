"""
Execution Engine | 執行引擎
=======================

Advanced trade execution engine that orchestrates the complete trading workflow
from signal generation to position management with intelligent order routing.
高級交易執行引擎，協調從信號生成到倉位管理的完整交易工作流程，配合智能訂單路由。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import uuid

from core.trading_strategy import AIFXTradingStrategy, TradingDecision
from trading.live_trader import LiveTrader, TradeStatus
from trading.position_manager import PositionManager
from brokers.ig_markets import IGMarketsConnector

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode enumeration | 執行模式枚舉"""
    AGGRESSIVE = "aggressive"  # Immediate market orders
    CONSERVATIVE = "conservative"  # Limit orders with better pricing
    BALANCED = "balanced"  # Mix of market and limit orders
    SMART = "smart"  # AI-driven execution optimization


@dataclass
class ExecutionPlan:
    """
    Execution plan for trading decisions | 交易決策的執行計劃
    """
    plan_id: str
    decisions: List[TradingDecision]
    execution_mode: ExecutionMode
    priority: int = 1  # 1=highest, 5=lowest
    max_execution_time: timedelta = timedelta(minutes=5)
    created_time: datetime = None
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now()


@dataclass
class ExecutionResult:
    """
    Execution result tracking | 執行結果追蹤
    """
    plan_id: str
    trade_ids: List[str]
    execution_time: float
    successful_executions: int
    failed_executions: int
    total_decisions: int
    errors: List[str]
    performance_metrics: Dict[str, Any]


class ExecutionEngine:
    """
    Execution Engine - Orchestrates complete trading workflow
    執行引擎 - 協調完整的交易工作流程
    
    Provides sophisticated trade execution including:
    - Intelligent order routing and timing | 智能訂單路由和時機
    - Risk validation and position sizing | 風險驗證和倉位大小
    - Multi-asset execution coordination | 多資產執行協調
    - Performance monitoring and optimization | 績效監控和優化
    - Emergency controls and circuit breakers | 緊急控制和斷路器
    
    提供精細的交易執行。
    """
    
    def __init__(self, 
                 ig_connector: IGMarketsConnector,
                 live_trader: LiveTrader,
                 position_manager: PositionManager,
                 max_concurrent_executions: int = 3,
                 circuit_breaker_threshold: int = 5):
        """
        Initialize Execution Engine | 初始化執行引擎
        
        Args:
            ig_connector: IG Markets API connector | IG Markets API連接器
            live_trader: Live trading executor | 實時交易執行器
            position_manager: Position management system | 倉位管理系統
            max_concurrent_executions: Maximum concurrent executions | 最大併發執行數
            circuit_breaker_threshold: Failed executions before circuit break | 斷路前的失敗執行數
        """
        self.ig_connector = ig_connector
        self.live_trader = live_trader
        self.position_manager = position_manager
        self.max_concurrent_executions = max_concurrent_executions
        self.circuit_breaker_threshold = circuit_breaker_threshold
        
        # Execution state | 執行狀態
        self.is_running = False
        self.circuit_breaker_active = False
        self.execution_queue: List[ExecutionPlan] = []
        self.active_executions: Dict[str, ExecutionPlan] = {}
        self.completed_executions: Dict[str, ExecutionResult] = {}
        
        # Performance tracking | 績效追蹤
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.recent_failures = []
        self.avg_execution_time = 0.0
        
        # Configuration | 配置
        self.default_execution_mode = ExecutionMode.BALANCED
        self.max_slippage_tolerance = 0.0010  # 10 basis points
        self.max_execution_delay = timedelta(seconds=30)
        
        logger.info("Execution Engine initialized")
    
    async def start(self) -> None:
        """Start the execution engine | 啟動執行引擎"""
        if self.is_running:
            logger.warning("Execution engine already running")
            return
        
        self.is_running = True
        logger.info("🚀 Execution Engine started")
        
        # Start background tasks | 啟動後台任務
        asyncio.create_task(self._execution_loop())
        asyncio.create_task(self._monitoring_loop())
    
    async def stop(self) -> None:
        """Stop the execution engine | 停止執行引擎"""
        self.is_running = False
        
        # Wait for active executions to complete | 等待活躍執行完成
        if self.active_executions:
            logger.info(f"Waiting for {len(self.active_executions)} executions to complete...")
            timeout = 30  # seconds
            
            while self.active_executions and timeout > 0:
                await asyncio.sleep(1)
                timeout -= 1
        
        logger.info("🛑 Execution Engine stopped")
    
    async def execute_decisions(self, decisions: List[TradingDecision],
                              execution_mode: Optional[ExecutionMode] = None,
                              priority: int = 1) -> str:
        """
        Execute trading decisions | 執行交易決策
        
        Args:
            decisions: List of trading decisions to execute | 要執行的交易決策列表
            execution_mode: Execution mode (None for default) | 執行模式
            priority: Execution priority (1=highest, 5=lowest) | 執行優先級
            
        Returns:
            str: Execution plan ID for tracking | 用於追蹤的執行計劃ID
        """
        try:
            if not self.is_running:
                raise RuntimeError("Execution engine not running")
            
            if self.circuit_breaker_active:
                raise RuntimeError("Circuit breaker active - executions suspended")
            
            # Create execution plan | 創建執行計劃
            plan_id = str(uuid.uuid4())
            execution_plan = ExecutionPlan(
                plan_id=plan_id,
                decisions=decisions,
                execution_mode=execution_mode or self.default_execution_mode,
                priority=priority
            )
            
            # Validate execution plan | 驗證執行計劃
            if not await self._validate_execution_plan(execution_plan):
                raise ValueError("Execution plan validation failed")
            
            # Add to queue | 添加到隊列
            self.execution_queue.append(execution_plan)
            self.execution_queue.sort(key=lambda p: p.priority)  # Sort by priority
            
            logger.info(f"📋 Execution plan {plan_id} queued with {len(decisions)} decisions")
            return plan_id
            
        except Exception as e:
            logger.error(f"Error creating execution plan: {e}")
            raise
    
    async def cancel_execution(self, plan_id: str) -> bool:
        """
        Cancel pending execution plan | 取消待執行計劃
        
        Args:
            plan_id: Execution plan ID to cancel | 要取消的執行計劃ID
            
        Returns:
            bool: True if cancellation was successful | 取消成功時返回True
        """
        try:
            # Check if in queue | 檢查是否在隊列中
            for i, plan in enumerate(self.execution_queue):
                if plan.plan_id == plan_id:
                    del self.execution_queue[i]
                    logger.info(f"✅ Execution plan {plan_id} cancelled from queue")
                    return True
            
            # Check if actively executing | 檢查是否正在執行
            if plan_id in self.active_executions:
                logger.warning(f"Cannot cancel active execution {plan_id}")
                return False
            
            logger.warning(f"Execution plan {plan_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling execution {plan_id}: {e}")
            return False
    
    async def get_execution_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """
        Get execution status | 獲取執行狀態
        
        Args:
            plan_id: Execution plan ID | 執行計劃ID
            
        Returns:
            Dict: Execution status information | 執行狀態信息
        """
        # Check completed executions | 檢查已完成執行
        if plan_id in self.completed_executions:
            result = self.completed_executions[plan_id]
            return {
                'status': 'completed',
                'plan_id': result.plan_id,
                'execution_time': result.execution_time,
                'successful_executions': result.successful_executions,
                'failed_executions': result.failed_executions,
                'total_decisions': result.total_decisions,
                'errors': result.errors,
                'performance_metrics': result.performance_metrics
            }
        
        # Check active executions | 檢查活躍執行
        if plan_id in self.active_executions:
            plan = self.active_executions[plan_id]
            return {
                'status': 'executing',
                'plan_id': plan.plan_id,
                'decisions_count': len(plan.decisions),
                'execution_mode': plan.execution_mode.value,
                'started_time': plan.created_time.isoformat(),
                'priority': plan.priority
            }
        
        # Check queued executions | 檢查排隊執行
        for plan in self.execution_queue:
            if plan.plan_id == plan_id:
                queue_position = self.execution_queue.index(plan) + 1
                return {
                    'status': 'queued',
                    'plan_id': plan.plan_id,
                    'queue_position': queue_position,
                    'decisions_count': len(plan.decisions),
                    'execution_mode': plan.execution_mode.value,
                    'created_time': plan.created_time.isoformat(),
                    'priority': plan.priority
                }
        
        return None
    
    async def _execution_loop(self) -> None:
        """Main execution loop | 主執行循環"""
        logger.info("📊 Execution loop started")
        
        while self.is_running:
            try:
                # Check circuit breaker | 檢查斷路器
                await self._check_circuit_breaker()
                
                # Process execution queue | 處理執行隊列
                if (self.execution_queue and 
                    len(self.active_executions) < self.max_concurrent_executions and
                    not self.circuit_breaker_active):
                    
                    # Get highest priority execution | 獲取最高優先級執行
                    execution_plan = self.execution_queue.pop(0)
                    
                    # Start execution | 開始執行
                    asyncio.create_task(self._execute_plan(execution_plan))
                
                # Update trade status | 更新交易狀態
                await self.live_trader.update_trade_status()
                
                # Update position status | 更新倉位狀態
                await self.position_manager.update_positions()
                
                await asyncio.sleep(1)  # Loop frequency
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(5)  # Error recovery delay
    
    async def _monitoring_loop(self) -> None:
        """Monitoring and maintenance loop | 監控和維護循環"""
        logger.info("👁️ Monitoring loop started")
        
        while self.is_running:
            try:
                # Check for expired executions | 檢查過期執行
                await self._check_expired_executions()
                
                # Update performance metrics | 更新績效指標
                await self._update_performance_metrics()
                
                # Cleanup old records | 清理舊記錄
                await self._cleanup_old_records()
                
                await asyncio.sleep(10)  # Monitoring frequency
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    async def _execute_plan(self, plan: ExecutionPlan) -> None:
        """Execute a specific execution plan | 執行特定的執行計劃"""
        start_time = datetime.now()
        trade_ids = []
        errors = []
        
        try:
            logger.info(f"🎯 Executing plan {plan.plan_id} with {len(plan.decisions)} decisions")
            
            # Add to active executions | 添加到活躍執行
            self.active_executions[plan.plan_id] = plan
            
            # Execute each decision | 執行每個決策
            for decision in plan.decisions:
                try:
                    # Pre-execution validation | 執行前驗證
                    if not await self._validate_decision(decision):
                        errors.append(f"Decision validation failed for {decision.symbol}")
                        continue
                    
                    # Execute the decision | 執行決策
                    trade_id = await self.live_trader.execute_decision(decision)
                    trade_ids.append(trade_id)
                    
                    # Add position to position manager | 添加倉位到倉位管理器
                    if decision.action in ['BUY', 'SELL']:
                        await self.position_manager.add_position(
                            symbol=decision.symbol,
                            side='long' if decision.action == 'BUY' else 'short',
                            size=decision.position_size,
                            entry_price=decision.entry_price or 0.0,
                            stop_loss=decision.stop_loss,
                            take_profit=decision.take_profit,
                            confidence=decision.confidence,
                            strategy_signal=decision.reasoning
                        )
                    
                    logger.debug(f"✅ Executed decision for {decision.symbol}: {trade_id}")
                    
                except Exception as e:
                    error_msg = f"Decision execution failed for {decision.symbol}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            # Create execution result | 創建執行結果
            execution_time = (datetime.now() - start_time).total_seconds()
            successful_count = len(trade_ids) - len(errors)
            
            result = ExecutionResult(
                plan_id=plan.plan_id,
                trade_ids=trade_ids,
                execution_time=execution_time,
                successful_executions=successful_count,
                failed_executions=len(errors),
                total_decisions=len(plan.decisions),
                errors=errors,
                performance_metrics=await self._calculate_execution_metrics(plan, trade_ids)
            )
            
            # Store result | 存儲結果
            self.completed_executions[plan.plan_id] = result
            
            # Update statistics | 更新統計
            self.total_executions += 1
            if len(errors) == 0:
                self.successful_executions += 1
            else:
                self.failed_executions += 1
                self.recent_failures.append(datetime.now())
            
            logger.info(f"📊 Plan {plan.plan_id} completed: {successful_count}/{len(plan.decisions)} successful")
            
        except Exception as e:
            logger.error(f"Error executing plan {plan.plan_id}: {e}")
            errors.append(str(e))
        
        finally:
            # Remove from active executions | 從活躍執行中移除
            if plan.plan_id in self.active_executions:
                del self.active_executions[plan.plan_id]
    
    async def _validate_execution_plan(self, plan: ExecutionPlan) -> bool:
        """Validate execution plan | 驗證執行計劃"""
        try:
            # Check if any decisions | 檢查是否有決策
            if not plan.decisions:
                logger.warning("Execution plan has no decisions")
                return False
            
            # Check resource availability | 檢查資源可用性
            if len(self.active_executions) >= self.max_concurrent_executions:
                logger.warning("Maximum concurrent executions reached")
                return False
            
            # Validate each decision | 驗證每個決策
            for decision in plan.decisions:
                if not await self._validate_decision(decision):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating execution plan: {e}")
            return False
    
    async def _validate_decision(self, decision: TradingDecision) -> bool:
        """Validate individual trading decision | 驗證個別交易決策"""
        try:
            # Check basic decision parameters | 檢查基本決策參數
            if not decision.symbol or not decision.action:
                return False
            
            if decision.position_size <= 0:
                return False
            
            if decision.confidence < 0.1:  # Minimum confidence threshold
                return False
            
            # Check position limits | 檢查倉位限制
            current_positions = len(self.position_manager.positions)
            if current_positions >= self.position_manager.max_positions:
                logger.warning(f"Position limit reached: {current_positions}")
                return False
            
            # Check symbol support | 檢查品種支持
            supported_symbols = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'USDCHF=X', 'AUDUSD=X']
            if decision.symbol not in supported_symbols:
                logger.warning(f"Unsupported symbol: {decision.symbol}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating decision: {e}")
            return False
    
    async def _check_circuit_breaker(self) -> None:
        """Check and update circuit breaker status | 檢查並更新斷路器狀態"""
        try:
            # Clean old failures | 清理舊失敗記錄
            cutoff_time = datetime.now() - timedelta(minutes=15)
            self.recent_failures = [
                failure_time for failure_time in self.recent_failures 
                if failure_time > cutoff_time
            ]
            
            # Check if circuit breaker should be activated | 檢查是否應激活斷路器
            if len(self.recent_failures) >= self.circuit_breaker_threshold:
                if not self.circuit_breaker_active:
                    self.circuit_breaker_active = True
                    logger.warning(f"🚨 Circuit breaker ACTIVATED - {len(self.recent_failures)} recent failures")
            else:
                if self.circuit_breaker_active:
                    self.circuit_breaker_active = False
                    logger.info("✅ Circuit breaker DEACTIVATED - failure rate normalized")
                    
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
    
    async def _check_expired_executions(self) -> None:
        """Check for expired executions | 檢查過期執行"""
        try:
            current_time = datetime.now()
            expired_plans = []
            
            for plan_id, plan in self.active_executions.items():
                if current_time - plan.created_time > plan.max_execution_time:
                    expired_plans.append(plan_id)
            
            for plan_id in expired_plans:
                logger.warning(f"⏰ Execution plan {plan_id} expired")
                # Could implement timeout handling here
                
        except Exception as e:
            logger.error(f"Error checking expired executions: {e}")
    
    async def _update_performance_metrics(self) -> None:
        """Update performance metrics | 更新績效指標"""
        try:
            if self.total_executions > 0:
                execution_times = [
                    result.execution_time for result in self.completed_executions.values()
                ]
                self.avg_execution_time = sum(execution_times) / len(execution_times)
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _cleanup_old_records(self) -> None:
        """Cleanup old execution records | 清理舊執行記錄"""
        try:
            # Keep only last 100 completed executions | 只保留最後100個已完成執行
            if len(self.completed_executions) > 100:
                oldest_plans = sorted(
                    self.completed_executions.items(),
                    key=lambda x: x[1].performance_metrics.get('completion_time', datetime.min)
                )
                
                for plan_id, _ in oldest_plans[:-100]:
                    del self.completed_executions[plan_id]
                    
                logger.debug(f"Cleaned up {len(oldest_plans) - 100} old execution records")
                
        except Exception as e:
            logger.error(f"Error cleaning up records: {e}")
    
    async def _calculate_execution_metrics(self, plan: ExecutionPlan, 
                                         trade_ids: List[str]) -> Dict[str, Any]:
        """Calculate execution performance metrics | 計算執行績效指標"""
        try:
            metrics = {
                'completion_time': datetime.now(),
                'execution_mode': plan.execution_mode.value,
                'priority': plan.priority,
                'trade_count': len(trade_ids),
                'decision_count': len(plan.decisions)
            }
            
            # Add trader performance if available | 如果可用則添加交易器績效
            trader_performance = self.live_trader.get_performance_summary()
            metrics.update({
                'trader_success_rate': trader_performance.get('success_rate', 0),
                'trader_fill_rate': trader_performance.get('fill_rate', 0)
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating execution metrics: {e}")
            return {}
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get comprehensive engine status | 獲取綜合引擎狀態"""
        return {
            'is_running': self.is_running,
            'circuit_breaker_active': self.circuit_breaker_active,
            'queue_size': len(self.execution_queue),
            'active_executions': len(self.active_executions),
            'completed_executions': len(self.completed_executions),
            'performance_metrics': {
                'total_executions': self.total_executions,
                'successful_executions': self.successful_executions,
                'failed_executions': self.failed_executions,
                'success_rate': self.successful_executions / max(1, self.total_executions),
                'avg_execution_time': self.avg_execution_time,
                'recent_failures': len(self.recent_failures)
            },
            'configuration': {
                'max_concurrent_executions': self.max_concurrent_executions,
                'circuit_breaker_threshold': self.circuit_breaker_threshold,
                'default_execution_mode': self.default_execution_mode.value,
                'max_slippage_tolerance': self.max_slippage_tolerance,
                'max_execution_delay_seconds': self.max_execution_delay.total_seconds()
            }
        }


# Factory function | 工廠函數
def create_execution_engine(ig_connector: IGMarketsConnector,
                          live_trader: LiveTrader,
                          position_manager: PositionManager,
                          **kwargs) -> ExecutionEngine:
    """
    Create an Execution Engine instance | 創建執行引擎實例
    
    Args:
        ig_connector: IG Markets API connector | IG Markets API連接器
        live_trader: Live trading executor | 實時交易執行器
        position_manager: Position management system | 倉位管理系統
        **kwargs: Additional configuration parameters | 額外配置參數
        
    Returns:
        ExecutionEngine: Configured execution engine instance | 配置好的執行引擎實例
    """
    return ExecutionEngine(ig_connector, live_trader, position_manager, **kwargs)