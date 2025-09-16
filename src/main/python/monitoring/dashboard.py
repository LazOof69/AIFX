"""
Trading Dashboard | 交易儀表板
===========================

Real-time trading dashboard that provides comprehensive monitoring of the
AIFX trading system with live updates and performance metrics.
提供AIFX交易系統全面監控的即時交易儀表板，配合實時更新和績效指標。
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
import time
import os
from pathlib import Path

# For simple dashboard display (could be extended to web interface)
import threading

logger = logging.getLogger(__name__)


@dataclass
class DashboardMetrics:
    """
    Dashboard metrics data structure | 儀表板指標數據結構
    """
    # System status | 系統狀態
    system_status: str = "unknown"
    uptime_seconds: float = 0.0
    last_update: datetime = None
    
    # Trading metrics | 交易指標
    total_positions: int = 0
    open_positions: int = 0
    daily_trades: int = 0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    # Performance metrics | 績效指標
    win_rate: float = 0.0
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    
    # Market data | 市場數據
    active_symbols: List[str] = None
    last_prices: Dict[str, float] = None
    
    # System health | 系統健康狀況
    api_status: str = "unknown"
    circuit_breaker_active: bool = False
    recent_errors: int = 0
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()
        if self.active_symbols is None:
            self.active_symbols = []
        if self.last_prices is None:
            self.last_prices = {}


class TradingDashboard:
    """
    Trading Dashboard - Real-time monitoring interface
    交易儀表板 - 即時監控介面
    
    Provides comprehensive real-time monitoring including:
    - System status and health monitoring | 系統狀態和健康監控
    - Live trading metrics and performance | 實時交易指標和績效
    - Position tracking and P&L updates | 倉位追蹤和盈虧更新
    - Market data and price feeds | 市場數據和價格源
    - Alert and notification system | 警報和通知系統
    
    提供全面的即時監控。
    """
    
    def __init__(self, 
                 update_interval: int = 5,
                 dashboard_port: Optional[int] = 8088,
                 enable_logging: bool = True):
        """
        Initialize Trading Dashboard | 初始化交易儀表板
        
        Args:
            update_interval: Dashboard update interval in seconds | 儀表板更新間隔（秒）
            dashboard_port: Port for web dashboard (None to disable) | 網頁儀表板端口
            enable_logging: Enable dashboard logging | 啟用儀表板日誌
        """
        self.update_interval = update_interval
        self.dashboard_port = dashboard_port
        self.enable_logging = enable_logging
        
        # Dashboard state | 儀表板狀態
        self.is_running = False
        self.start_time = None
        self.metrics = DashboardMetrics()
        
        # Data sources | 數據源
        self.ig_connector = None
        self.live_trader = None
        self.position_manager = None
        self.execution_engine = None
        
        # Update callbacks | 更新回調
        self.update_callbacks: List[Callable[[DashboardMetrics], None]] = []
        
        # Alert system | 警報系統
        self.alerts: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            'max_drawdown_pct': -10.0,  # Alert if drawdown > 10%
            'min_success_rate': 0.3,    # Alert if success rate < 30%
            'max_open_positions': 8,    # Alert if too many positions
            'max_daily_trades': 15      # Alert if too many daily trades
        }
        
        # Dashboard display | 儀表板顯示
        self.console_output = True
        self.last_console_update = None
        
        logger.info("Trading Dashboard initialized")
    
    def register_components(self, **components) -> None:
        """
        Register trading system components | 註冊交易系統組件
        
        Args:
            **components: Trading system components to monitor | 要監控的交易系統組件
        """
        self.ig_connector = components.get('ig_connector')
        self.live_trader = components.get('live_trader')
        self.position_manager = components.get('position_manager')
        self.execution_engine = components.get('execution_engine')
        
        logger.info(f"Registered {len(components)} components for monitoring")
    
    def add_update_callback(self, callback: Callable[[DashboardMetrics], None]) -> None:
        """
        Add callback for dashboard updates | 添加儀表板更新回調
        
        Args:
            callback: Function to call on each update | 每次更新時調用的函數
        """
        self.update_callbacks.append(callback)
    
    async def start_monitoring(self) -> None:
        """Start the dashboard monitoring | 啟動儀表板監控"""
        return await self.start()
    
    async def start(self) -> None:
        """Start the dashboard monitoring | 啟動儀表板監控"""
        if self.is_running:
            logger.warning("Dashboard already running")
            return
        
        self.is_running = True
        self.start_time = datetime.now()
        
        logger.info("🖥️ Trading Dashboard started")
        
        # Start monitoring loop | 啟動監控循環
        asyncio.create_task(self._monitoring_loop())
        
        # Start console display if enabled | 如果啟用則開始控制台顯示
        if self.console_output:
            asyncio.create_task(self._console_display_loop())
    
    async def stop(self) -> None:
        """Stop the dashboard monitoring | 停止儀表板監控"""
        self.is_running = False
        logger.info("🛑 Trading Dashboard stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop | 主監控循環"""
        logger.info("📊 Dashboard monitoring loop started")
        
        while self.is_running:
            try:
                # Update metrics | 更新指標
                await self._update_metrics()
                
                # Check alerts | 檢查警報
                await self._check_alerts()
                
                # Call update callbacks | 調用更新回調
                for callback in self.update_callbacks:
                    try:
                        callback(self.metrics)
                    except Exception as e:
                        logger.error(f"Error in update callback: {e}")
                
                # Log metrics if enabled | 如果啟用則記錄指標
                if self.enable_logging:
                    await self._log_metrics()
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Error recovery delay
    
    async def _console_display_loop(self) -> None:
        """Console display loop | 控制台顯示循環"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Update console every 10 seconds or on significant changes
                # 每10秒或在重要變化時更新控制台
                if (self.last_console_update is None or 
                    (current_time - self.last_console_update).total_seconds() >= 10):
                    
                    self._display_console_dashboard()
                    self.last_console_update = current_time
                
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in console display: {e}")
                await asyncio.sleep(5)
    
    async def _update_metrics(self) -> None:
        """Update all dashboard metrics | 更新所有儀表板指標"""
        try:
            # Update system status | 更新系統狀態
            if self.start_time:
                self.metrics.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            
            # Update from IG connector | 從IG連接器更新
            if self.ig_connector:
                ig_status = self.ig_connector.get_status()
                self.metrics.api_status = ig_status.get('connection_status', 'unknown')
                self.metrics.system_status = "running" if self.metrics.api_status == "connected" else "disconnected"
            
            # Update from position manager | 從倉位管理器更新
            if self.position_manager:
                portfolio_summary = self.position_manager.get_portfolio_summary()
                
                self.metrics.total_positions = portfolio_summary['position_counts']['total_opened']
                self.metrics.open_positions = portfolio_summary['position_counts']['open_positions']
                self.metrics.total_pnl = portfolio_summary['performance_metrics']['total_realized_pnl']
                self.metrics.unrealized_pnl = portfolio_summary['risk_metrics']['total_unrealized_pnl']
                self.metrics.win_rate = portfolio_summary['performance_metrics']['win_rate'] * 100
            
            # Update from live trader | 從實時交易器更新
            if self.live_trader:
                trader_performance = self.live_trader.get_performance_summary()
                self.metrics.success_rate = trader_performance.get('success_rate', 0) * 100
                self.metrics.daily_trades = len(self.live_trader.get_recent_trades(50))  # Approximate daily trades
            
            # Update from execution engine | 從執行引擎更新
            if self.execution_engine:
                engine_status = self.execution_engine.get_engine_status()
                self.metrics.circuit_breaker_active = engine_status['circuit_breaker_active']
                self.metrics.avg_execution_time = engine_status['performance_metrics']['avg_execution_time']
                self.metrics.recent_errors = engine_status['performance_metrics']['recent_failures']
            
            # Update market data | 更新市場數據
            if self.ig_connector:
                await self._update_market_data()
            
            # Update timestamp | 更新時間戳
            self.metrics.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def _update_market_data(self) -> None:
        """Update market data metrics | 更新市場數據指標"""
        try:
            symbols = ["CS.D.EURUSD.MINI.IP", "CS.D.USDJPY.MINI.IP"]
            self.metrics.active_symbols = []
            self.metrics.last_prices = {}
            
            for symbol in symbols:
                try:
                    market_data = await self.ig_connector.get_market_data(symbol)
                    if market_data and 'bid' in market_data:
                        self.metrics.active_symbols.append(symbol)
                        self.metrics.last_prices[symbol] = market_data['bid']
                except Exception as e:
                    logger.debug(f"Error getting market data for {symbol}: {e}")
                    
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def _check_alerts(self) -> None:
        """Check for alert conditions | 檢查警報條件"""
        try:
            current_time = datetime.now()
            new_alerts = []
            
            # Check drawdown alert | 檢查回撤警報
            if (self.metrics.total_pnl < 0 and 
                abs(self.metrics.total_pnl) / 100000 > abs(self.alert_thresholds['max_drawdown_pct'] / 100)):
                new_alerts.append({
                    'type': 'high_drawdown',
                    'severity': 'warning',
                    'message': f"High drawdown detected: ${self.metrics.total_pnl:.2f}",
                    'timestamp': current_time
                })
            
            # Check success rate alert | 檢查成功率警報
            # Only alert on low success rate if we have actual trading history
            # 只有在有實際交易歷史時才對低成功率發出警報
            if (self.metrics.daily_trades > 5 and
                self.metrics.success_rate < self.alert_thresholds['min_success_rate'] * 100):
                new_alerts.append({
                    'type': 'low_success_rate',
                    'severity': 'warning',
                    'message': f"Low success rate: {self.metrics.success_rate:.1f}%",
                    'timestamp': current_time
                })
            
            # Check position limit alert | 檢查倉位限制警報
            if self.metrics.open_positions > self.alert_thresholds['max_open_positions']:
                new_alerts.append({
                    'type': 'position_limit',
                    'severity': 'warning',
                    'message': f"High position count: {self.metrics.open_positions}",
                    'timestamp': current_time
                })
            
            # Check daily trade limit | 檢查每日交易限制
            if self.metrics.daily_trades > self.alert_thresholds['max_daily_trades']:
                new_alerts.append({
                    'type': 'trade_limit',
                    'severity': 'info',
                    'message': f"High daily trade count: {self.metrics.daily_trades}",
                    'timestamp': current_time
                })
            
            # Check circuit breaker | 檢查斷路器
            if self.metrics.circuit_breaker_active:
                new_alerts.append({
                    'type': 'circuit_breaker',
                    'severity': 'critical',
                    'message': "Circuit breaker is active",
                    'timestamp': current_time
                })
            
            # Add new alerts | 添加新警報
            self.alerts.extend(new_alerts)
            
            # Log new alerts | 記錄新警報
            for alert in new_alerts:
                severity_icon = {'info': 'ℹ️', 'warning': '⚠️', 'critical': '🚨'}.get(alert['severity'], '❗')
                logger.warning(f"{severity_icon} ALERT: {alert['message']}")
            
            # Keep only recent alerts (last hour) | 只保留最近的警報（最後一小時）
            cutoff_time = current_time - timedelta(hours=1)
            self.alerts = [alert for alert in self.alerts if alert['timestamp'] > cutoff_time]
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def _display_console_dashboard(self) -> None:
        """Display dashboard in console | 在控制台顯示儀表板"""
        try:
            # Clear screen (works on most terminals) | 清除屏幕
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Dashboard header | 儀表板標頭
            print("=" * 80)
            print("🚀 AIFX TRADING SYSTEM DASHBOARD | AIFX 交易系統儀表板")
            print("=" * 80)
            
            # System status | 系統狀態
            status_icon = "🟢" if self.metrics.system_status == "running" else "🔴"
            uptime_str = self._format_uptime(self.metrics.uptime_seconds)
            
            print(f"{status_icon} System Status: {self.metrics.system_status.upper()}")
            print(f"⏱️ Uptime: {uptime_str}")
            print(f"🔗 API Status: {self.metrics.api_status}")
            print(f"🕒 Last Update: {self.metrics.last_update.strftime('%H:%M:%S')}")
            print()
            
            # Trading metrics | 交易指標
            print("📊 TRADING METRICS | 交易指標")
            print("-" * 40)
            pnl_icon = "💰" if self.metrics.total_pnl >= 0 else "📉"
            print(f"💼 Total Positions: {self.metrics.total_positions}")
            print(f"📈 Open Positions: {self.metrics.open_positions}")
            print(f"📅 Daily Trades: {self.metrics.daily_trades}")
            print(f"{pnl_icon} Total P&L: ${self.metrics.total_pnl:.2f}")
            print(f"💵 Unrealized P&L: ${self.metrics.unrealized_pnl:.2f}")
            print()
            
            # Performance metrics | 績效指標
            print("🎯 PERFORMANCE METRICS | 績效指標")
            print("-" * 40)
            print(f"🏆 Win Rate: {self.metrics.win_rate:.1f}%")
            print(f"✅ Success Rate: {self.metrics.success_rate:.1f}%")
            print(f"⚡ Avg Execution Time: {self.metrics.avg_execution_time:.3f}s")
            print()
            
            # Market data | 市場數據
            if self.metrics.last_prices:
                print("📈 MARKET DATA | 市場數據")
                print("-" * 40)
                for symbol, price in self.metrics.last_prices.items():
                    display_symbol = symbol.replace('CS.D.', '').replace('.MINI.IP', '')
                    print(f"💱 {display_symbol}: {price:.5f}")
                print()
            
            # System health | 系統健康狀況
            print("🏥 SYSTEM HEALTH | 系統健康狀況")
            print("-" * 40)
            breaker_icon = "🚨" if self.metrics.circuit_breaker_active else "✅"
            print(f"{breaker_icon} Circuit Breaker: {'ACTIVE' if self.metrics.circuit_breaker_active else 'OK'}")
            print(f"❗ Recent Errors: {self.metrics.recent_errors}")
            print()
            
            # Recent alerts | 最近警報
            recent_alerts = [alert for alert in self.alerts 
                           if (datetime.now() - alert['timestamp']).total_seconds() < 300]  # Last 5 minutes
            
            if recent_alerts:
                print("🚨 RECENT ALERTS | 最近警報")
                print("-" * 40)
                for alert in recent_alerts[-3:]:  # Show last 3 alerts
                    severity_icon = {'info': 'ℹ️', 'warning': '⚠️', 'critical': '🚨'}.get(alert['severity'], '❗')
                    time_str = alert['timestamp'].strftime('%H:%M:%S')
                    print(f"{severity_icon} [{time_str}] {alert['message']}")
                print()
            
            # Footer | 頁腳
            print("=" * 80)
            print(f"Dashboard refreshes every {self.update_interval} seconds | Press Ctrl+C to exit")
            print("=" * 80)
            
        except Exception as e:
            logger.error(f"Error displaying console dashboard: {e}")
    
    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human readable format | 格式化運行時間為人可讀格式"""
        try:
            hours, remainder = divmod(int(seconds), 3600)
            minutes, secs = divmod(remainder, 60)
            
            if hours > 0:
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            else:
                return f"{minutes:02d}:{secs:02d}"
                
        except Exception:
            return "00:00"
    
    async def _log_metrics(self) -> None:
        """Log metrics to file | 將指標記錄到文件"""
        try:
            # Create logs directory if it doesn't exist | 如果不存在則創建日誌目錄
            logs_dir = Path("logs")
            logs_dir.mkdir(exist_ok=True)
            
            # Log to daily file | 記錄到日誌文件
            log_file = logs_dir / f"dashboard_{datetime.now().strftime('%Y%m%d')}.log"
            
            log_entry = {
                'timestamp': self.metrics.last_update.isoformat(),
                'metrics': asdict(self.metrics)
            }
            
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')
                
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    def get_current_metrics(self) -> DashboardMetrics:
        """Get current dashboard metrics | 獲取當前儀表板指標"""
        return self.metrics
    
    def get_alerts(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get dashboard alerts | 獲取儀表板警報
        
        Args:
            severity: Filter by severity ('info', 'warning', 'critical') | 按嚴重性過濾
            
        Returns:
            List: Filtered alerts | 過濾後的警報
        """
        if severity:
            return [alert for alert in self.alerts if alert['severity'] == severity]
        return self.alerts.copy()
    
    def export_metrics(self, output_file: str) -> bool:
        """
        Export current metrics to file | 導出當前指標到文件
        
        Args:
            output_file: Output file path | 輸出文件路徑
            
        Returns:
            bool: True if export successful | 導出成功時返回True
        """
        try:
            export_data = {
                'export_time': datetime.now().isoformat(),
                'metrics': asdict(self.metrics),
                'alerts': self.alerts,
                'system_info': {
                    'update_interval': self.update_interval,
                    'uptime_seconds': self.metrics.uptime_seconds
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Metrics exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return False


# Factory function | 工廠函數
def create_trading_dashboard(**kwargs) -> TradingDashboard:
    """
    Create a Trading Dashboard instance | 創建交易儀表板實例
    
    Args:
        **kwargs: Dashboard configuration parameters | 儀表板配置參數
        
    Returns:
        TradingDashboard: Configured dashboard instance | 配置好的儀表板實例
    """
    return TradingDashboard(**kwargs)