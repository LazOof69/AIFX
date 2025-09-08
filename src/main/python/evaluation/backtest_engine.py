"""
Comprehensive Backtesting Framework | 綜合回測框架

Advanced backtesting engine for the AIFX trading system with realistic trade simulation,
transaction costs, slippage modeling, and comprehensive performance analysis.
AIFX交易系統的高級回測引擎，具有真實的交易模擬、交易成本、滑點建模和綜合績效分析。

This module provides sophisticated backtesting capabilities for strategy validation.
此模組為策略驗證提供精細的回測能力。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import warnings
from concurrent.futures import ThreadPoolExecutor
import json

# Core components | 核心組件
from core.trading_strategy import AIFXTradingStrategy, StrategyConfig, TradingMode, TradingDecision
from core.risk_manager import AdvancedRiskManager, Position, RiskLevel
from evaluation.performance_metrics import TradingPerformanceMetrics

# Data processing | 數據處理
from utils.data_loader import DataLoader
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtesting mode enumeration | 回測模式枚舉"""
    SINGLE_STRATEGY = "single_strategy"
    MULTI_STRATEGY = "multi_strategy"
    PARAMETER_OPTIMIZATION = "parameter_optimization"
    WALK_FORWARD = "walk_forward"


class ExecutionModel(Enum):
    """Trade execution model | 交易執行模型"""
    MARKET_ON_OPEN = "market_on_open"
    MARKET_ON_CLOSE = "market_on_close"
    LIMIT_ORDER = "limit_order"
    REALISTIC = "realistic"


@dataclass
class BacktestConfig:
    """
    Backtesting configuration | 回測配置
    """
    # Basic settings | 基本設置
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    initial_capital: float = 100000.0
    timeframe: str = "1H"
    
    # Execution settings | 執行設置
    execution_model: ExecutionModel = ExecutionModel.REALISTIC
    commission_rate: float = 0.0001  # 1 basis point
    spread_cost: float = 0.0002  # 2 basis points
    slippage_factor: float = 0.0001  # 1 basis point
    
    # Data settings | 數據設置
    data_source: str = "yahoo"
    benchmark_symbol: str = "EURUSD=X"
    trading_symbols: List[str] = field(default_factory=lambda: ["EURUSD=X", "USDJPY=X"])
    
    # Performance settings | 績效設置
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    calculate_var: bool = True
    var_confidence: float = 0.05  # 95% VaR
    
    # Optimization settings | 優化設置
    enable_optimization: bool = False
    optimization_metric: str = "sharpe_ratio"  # sharpe_ratio, total_return, max_drawdown
    optimization_periods: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary | 轉換為字典"""
        return {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'timeframe': self.timeframe,
            'execution_model': self.execution_model.value,
            'commission_rate': self.commission_rate,
            'spread_cost': self.spread_cost,
            'slippage_factor': self.slippage_factor,
            'data_source': self.data_source,
            'benchmark_symbol': self.benchmark_symbol,
            'trading_symbols': self.trading_symbols,
            'risk_free_rate': self.risk_free_rate,
            'calculate_var': self.calculate_var,
            'var_confidence': self.var_confidence,
            'enable_optimization': self.enable_optimization,
            'optimization_metric': self.optimization_metric,
            'optimization_periods': self.optimization_periods
        }


@dataclass
class TradeRecord:
    """
    Individual trade record | 個別交易記錄
    """
    trade_id: str
    symbol: str
    side: str  # 'long' or 'short'
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    duration_hours: float = 0.0
    exit_reason: str = "unknown"
    signal_confidence: float = 0.0
    
    def calculate_pnl(self):
        """Calculate trade PnL | 計算交易盈虧"""
        if self.exit_time is None or self.exit_price == 0:
            return
        
        # Calculate gross PnL | 計算毛盈虧
        if self.side == 'long':
            self.gross_pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # short
            self.gross_pnl = (self.entry_price - self.exit_price) * self.quantity
        
        # Calculate net PnL (after costs) | 計算淨盈虧（扣除成本）
        total_costs = self.commission + abs(self.slippage * self.quantity)
        self.net_pnl = self.gross_pnl - total_costs
        
        # Calculate duration | 計算持續時間
        if self.exit_time:
            self.duration_hours = (self.exit_time - self.entry_time).total_seconds() / 3600
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary | 轉換為字典"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'side': self.side,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'commission': self.commission,
            'slippage': self.slippage,
            'gross_pnl': self.gross_pnl,
            'net_pnl': self.net_pnl,
            'duration_hours': self.duration_hours,
            'exit_reason': self.exit_reason,
            'signal_confidence': self.signal_confidence
        }


@dataclass
class BacktestResults:
    """
    Comprehensive backtesting results | 綜合回測結果
    """
    # Basic information | 基本信息
    strategy_name: str = ""
    start_date: datetime = None
    end_date: datetime = None
    total_duration_days: float = 0.0
    
    # Portfolio performance | 投資組合績效
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    
    # Risk metrics | 風險指標
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    value_at_risk: float = 0.0
    
    # Trading statistics | 交易統計
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Cost analysis | 成本分析
    total_commission: float = 0.0
    total_slippage: float = 0.0
    total_costs: float = 0.0
    cost_pct_of_returns: float = 0.0
    
    # Detailed records | 詳細記錄
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_returns: pd.Series = field(default_factory=pd.Series)
    
    # Benchmark comparison | 基準比較
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    tracking_error: float = 0.0
    information_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary | 轉換為字典"""
        return {
            'strategy_name': self.strategy_name,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'end_date': self.end_date.isoformat() if self.end_date else None,
            'total_duration_days': self.total_duration_days,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'total_return_pct': self.total_return_pct,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'value_at_risk': self.value_at_risk,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'profit_factor': self.profit_factor,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'total_costs': self.total_costs,
            'cost_pct_of_returns': self.cost_pct_of_returns,
            'benchmark_return': self.benchmark_return,
            'excess_return': self.excess_return,
            'tracking_error': self.tracking_error,
            'information_ratio': self.information_ratio
        }


class BacktestEngine:
    """
    Comprehensive backtesting engine | 綜合回測引擎
    
    Provides advanced backtesting capabilities with realistic trade execution,
    comprehensive performance analysis, and optimization features.
    提供高級回測能力，包括真實交易執行、綜合績效分析和優化功能。
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine | 初始化回測引擎
        
        Args:
            config: Backtesting configuration | 回測配置
        """
        self.config = config
        self.data_loader = DataLoader()
        self.performance_metrics = TradingPerformanceMetrics()
        
        # Backtesting state | 回測狀態
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.current_time: Optional[datetime] = None
        self.current_portfolio_value = config.initial_capital
        
        # Trade tracking | 交易追蹤
        self.active_trades: Dict[str, TradeRecord] = {}
        self.completed_trades: List[TradeRecord] = []
        self.trade_counter = 0
        
        # Performance tracking | 績效追蹤
        self.equity_history = []
        self.daily_returns = []
        self.portfolio_values = []
        self.timestamps = []
        
        logger.info(f"Initialized BacktestEngine: {config.start_date} to {config.end_date}")
    
    def load_data(self, symbols: Optional[List[str]] = None) -> bool:
        """
        Load market data for backtesting | 載入回測的市場數據
        
        Args:
            symbols: List of symbols to load (uses config if None) | 要載入的品種列表
            
        Returns:
            True if data loaded successfully | 如果數據載入成功則為True
        """
        try:
            symbols = symbols or self.config.trading_symbols
            
            logger.info(f"Loading market data for symbols: {symbols}")
            
            # Load data for each symbol | 為每個品種載入數據
            for symbol in symbols:
                try:
                    data = self.data_loader.load_data(
                        symbol=symbol,
                        start_date=self.config.start_date,
                        end_date=self.config.end_date,
                        timeframe=self.config.timeframe
                    )
                    
                    if data.empty:
                        logger.warning(f"No data loaded for symbol: {symbol}")
                        continue
                    
                    # Validate data quality | 驗證數據品質
                    if not self._validate_market_data(data, symbol):
                        logger.warning(f"Data quality issues for symbol: {symbol}")
                        continue
                    
                    self.market_data[symbol] = data
                    logger.info(f"Loaded {len(data)} rows for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error loading data for {symbol}: {e}")
                    continue
            
            # Load benchmark data | 載入基準數據
            if self.config.benchmark_symbol:
                try:
                    benchmark_data = self.data_loader.load_data(
                        symbol=self.config.benchmark_symbol,
                        start_date=self.config.start_date,
                        end_date=self.config.end_date,
                        timeframe=self.config.timeframe
                    )
                    
                    if not benchmark_data.empty:
                        self.benchmark_data = benchmark_data
                        logger.info(f"Loaded benchmark data: {self.config.benchmark_symbol}")
                    
                except Exception as e:
                    logger.warning(f"Error loading benchmark data: {e}")
            
            if not self.market_data:
                logger.error("No market data loaded successfully")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in data loading: {e}")
            return False
    
    def run_backtest(self, strategy: AIFXTradingStrategy, 
                    progress_callback: Optional[Callable[[float], None]] = None) -> BacktestResults:
        """
        Run comprehensive backtesting | 運行綜合回測
        
        Args:
            strategy: Trading strategy to backtest | 要回測的交易策略
            progress_callback: Optional callback for progress updates | 進度更新的可選回調
            
        Returns:
            Comprehensive backtesting results | 綜合回測結果
        """
        logger.info(f"Starting backtest for strategy: {strategy.config.strategy_name}")
        
        try:
            # Initialize backtesting | 初始化回測
            self._initialize_backtest(strategy)
            
            # Get unified timeline | 獲取統一時間線
            timeline = self._create_unified_timeline()
            total_steps = len(timeline)
            
            if total_steps == 0:
                raise ValueError("No data available for backtesting period")
            
            # Main backtesting loop | 主回測循環
            for i, timestamp in enumerate(timeline):
                try:
                    self.current_time = timestamp
                    
                    # Get market data for current timestamp | 獲取當前時間戳的市場數據
                    current_market_data = self._get_market_data_at_time(timestamp)
                    
                    if not current_market_data:
                        continue
                    
                    # Update portfolio value | 更新投資組合價值
                    self._update_portfolio_value(current_market_data)
                    
                    # Check exit conditions for active trades | 檢查活躍交易的離場條件
                    self._check_exit_conditions(current_market_data, strategy)
                    
                    # Run strategy cycle | 運行策略週期
                    cycle_results = strategy.run_strategy_cycle(current_market_data)
                    
                    # Process any new trading decisions | 處理任何新的交易決定
                    self._process_trading_decisions(cycle_results, current_market_data)
                    
                    # Record portfolio state | 記錄投資組合狀態
                    self._record_portfolio_state(timestamp)
                    
                    # Update progress | 更新進度
                    if progress_callback and i % max(1, total_steps // 100) == 0:
                        progress = (i + 1) / total_steps
                        progress_callback(progress)
                
                except Exception as e:
                    logger.warning(f"Error processing timestamp {timestamp}: {e}")
                    continue
            
            # Finalize backtest | 完成回測
            results = self._finalize_backtest(strategy)
            
            logger.info(f"Backtest completed: {results.total_trades} trades, "
                       f"{results.total_return_pct:.2f}% return")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            raise
    
    def run_parameter_optimization(self, strategy_class: type, 
                                 parameter_ranges: Dict[str, List[Any]],
                                 n_trials: int = 100) -> Dict[str, Any]:
        """
        Run parameter optimization | 運行參數優化
        
        Args:
            strategy_class: Strategy class to optimize | 要優化的策略類
            parameter_ranges: Parameter ranges to test | 要測試的參數範圍
            n_trials: Number of optimization trials | 優化試驗次數
            
        Returns:
            Optimization results | 優化結果
        """
        logger.info(f"Starting parameter optimization with {n_trials} trials")
        
        optimization_results = []
        
        try:
            # Generate parameter combinations | 生成參數組合
            parameter_combinations = self._generate_parameter_combinations(parameter_ranges, n_trials)
            
            # Run backtests for each combination | 為每個組合運行回測
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                
                for i, params in enumerate(parameter_combinations):
                    # Create strategy configuration | 創建策略配置
                    config = self._create_config_from_parameters(params)
                    
                    # Submit backtest job | 提交回測任務
                    future = executor.submit(self._run_single_optimization_trial, 
                                           strategy_class, config, i)
                    futures.append((future, params))
                
                # Collect results | 收集結果
                for future, params in futures:
                    try:
                        result = future.result(timeout=300)  # 5-minute timeout
                        optimization_results.append({
                            'parameters': params,
                            'performance': result
                        })
                    except Exception as e:
                        logger.warning(f"Optimization trial failed: {e}")
            
            # Analyze optimization results | 分析優化結果
            best_result = self._find_best_optimization_result(optimization_results)
            
            logger.info(f"Optimization completed. Best {self.config.optimization_metric}: "
                       f"{best_result['performance'][self.config.optimization_metric]:.4f}")
            
            return {
                'best_parameters': best_result['parameters'],
                'best_performance': best_result['performance'],
                'all_results': optimization_results,
                'total_trials': len(optimization_results)
            }
            
        except Exception as e:
            logger.error(f"Error in parameter optimization: {e}")
            return {'error': str(e)}
    
    def compare_strategies(self, strategies: List[AIFXTradingStrategy]) -> Dict[str, Any]:
        """
        Compare multiple strategies | 比較多個策略
        
        Args:
            strategies: List of strategies to compare | 要比較的策略列表
            
        Returns:
            Strategy comparison results | 策略比較結果
        """
        logger.info(f"Comparing {len(strategies)} strategies")
        
        comparison_results = {}
        
        try:
            # Run backtest for each strategy | 為每個策略運行回測
            for strategy in strategies:
                try:
                    results = self.run_backtest(strategy)
                    comparison_results[strategy.config.strategy_name] = results.to_dict()
                    
                except Exception as e:
                    logger.error(f"Error backtesting strategy {strategy.config.strategy_name}: {e}")
                    comparison_results[strategy.config.strategy_name] = {'error': str(e)}
            
            # Generate comparison metrics | 生成比較指標
            comparison_metrics = self._generate_comparison_metrics(comparison_results)
            
            return {
                'individual_results': comparison_results,
                'comparison_metrics': comparison_metrics,
                'ranking': self._rank_strategies(comparison_results)
            }
            
        except Exception as e:
            logger.error(f"Error in strategy comparison: {e}")
            return {'error': str(e)}
    
    def generate_report(self, results: BacktestResults, 
                       output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive backtesting report | 生成綜合回測報告
        
        Args:
            results: Backtesting results | 回測結果
            output_path: Optional path to save report | 保存報告的可選路徑
            
        Returns:
            Report data | 報告數據
        """
        try:
            report = {
                'executive_summary': self._generate_executive_summary(results),
                'performance_analysis': self._generate_performance_analysis(results),
                'risk_analysis': self._generate_risk_analysis(results),
                'trade_analysis': self._generate_trade_analysis(results),
                'cost_analysis': self._generate_cost_analysis(results),
                'recommendations': self._generate_recommendations(results),
                'appendices': {
                    'detailed_trades': [trade.to_dict() for trade in results.trades],
                    'equity_curve': results.equity_curve.to_dict(),
                    'configuration': self.config.to_dict()
                }
            }
            
            # Save report if path provided | 如果提供路徑則保存報告
            if output_path:
                self._save_report(report, output_path)
                logger.info(f"Backtest report saved to: {output_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {'error': str(e)}
    
    # Private helper methods | 私有輔助方法
    
    def _initialize_backtest(self, strategy: AIFXTradingStrategy):
        """Initialize backtesting state | 初始化回測狀態"""
        # Reset state | 重置狀態
        self.active_trades.clear()
        self.completed_trades.clear()
        self.trade_counter = 0
        self.current_portfolio_value = self.config.initial_capital
        
        # Reset tracking arrays | 重置追蹤數組
        self.equity_history.clear()
        self.daily_returns.clear()
        self.portfolio_values.clear()
        self.timestamps.clear()
        
        # Configure strategy for backtesting | 配置策略進行回測
        strategy.config.trading_mode = TradingMode.BACKTEST
        strategy.config.account_balance = self.config.initial_capital
        strategy.start_strategy()
        
        logger.info("Backtest initialization completed")
    
    def _create_unified_timeline(self) -> pd.DatetimeIndex:
        """Create unified timeline from all market data | 從所有市場數據創建統一時間線"""
        all_timestamps = set()
        
        for symbol, data in self.market_data.items():
            all_timestamps.update(data.index)
        
        timeline = pd.DatetimeIndex(sorted(all_timestamps))
        
        # Filter by date range | 按日期範圍過濾
        start_date = pd.to_datetime(self.config.start_date)
        end_date = pd.to_datetime(self.config.end_date)
        timeline = timeline[(timeline >= start_date) & (timeline <= end_date)]
        
        return timeline
    
    def _get_market_data_at_time(self, timestamp: pd.Timestamp) -> Dict[str, pd.DataFrame]:
        """Get market data for all symbols at specific timestamp | 獲取特定時間戳所有品種的市場數據"""
        current_data = {}
        
        for symbol, data in self.market_data.items():
            # Get data up to current timestamp | 獲取到當前時間戳的數據
            historical_data = data[data.index <= timestamp]
            
            if not historical_data.empty:
                current_data[symbol] = historical_data
        
        return current_data
    
    def _update_portfolio_value(self, market_data: Dict[str, pd.DataFrame]):
        """Update portfolio value based on current positions | 根據當前倉位更新投資組合價值"""
        total_position_value = 0.0
        
        for trade_id, trade in self.active_trades.items():
            symbol = trade.symbol
            if symbol in market_data and not market_data[symbol].empty:
                current_price = market_data[symbol]['Close'].iloc[-1]
                
                # Calculate position value | 計算倉位價值
                if trade.side == 'long':
                    position_value = (current_price - trade.entry_price) * trade.quantity
                else:  # short
                    position_value = (trade.entry_price - current_price) * trade.quantity
                
                total_position_value += position_value
        
        self.current_portfolio_value = self.config.initial_capital + total_position_value
    
    def _check_exit_conditions(self, market_data: Dict[str, pd.DataFrame], 
                             strategy: AIFXTradingStrategy):
        """Check exit conditions for active trades | 檢查活躍交易的離場條件"""
        trades_to_close = []
        
        for trade_id, trade in self.active_trades.items():
            symbol = trade.symbol
            if symbol not in market_data or market_data[symbol].empty:
                continue
            
            current_price = market_data[symbol]['Close'].iloc[-1]
            
            # Check stop loss and take profit | 檢查止損和止盈
            should_close = False
            exit_reason = "unknown"
            
            if trade.side == 'long':
                # Long position checks | 多頭倉位檢查
                if hasattr(trade, 'stop_loss') and current_price <= getattr(trade, 'stop_loss', 0):
                    should_close = True
                    exit_reason = "stop_loss"
                elif hasattr(trade, 'take_profit') and current_price >= getattr(trade, 'take_profit', float('inf')):
                    should_close = True
                    exit_reason = "take_profit"
            else:  # short position
                # Short position checks | 空頭倉位檢查
                if hasattr(trade, 'stop_loss') and current_price >= getattr(trade, 'stop_loss', float('inf')):
                    should_close = True
                    exit_reason = "stop_loss"
                elif hasattr(trade, 'take_profit') and current_price <= getattr(trade, 'take_profit', 0):
                    should_close = True
                    exit_reason = "take_profit"
            
            if should_close:
                trades_to_close.append((trade_id, current_price, exit_reason))
        
        # Close identified trades | 關閉識別的交易
        for trade_id, exit_price, exit_reason in trades_to_close:
            self._close_trade(trade_id, exit_price, exit_reason)
    
    def _process_trading_decisions(self, cycle_results: Dict[str, Any], 
                                 market_data: Dict[str, pd.DataFrame]):
        """Process trading decisions from strategy cycle | 處理策略週期的交易決定"""
        # This would normally receive decisions from the strategy
        # For now, we'll simulate based on cycle results
        # 這通常會從策略接收決定，現在我們將基於週期結果進行模擬
        
        if cycle_results.get('trades_executed', 0) > 0:
            # Simulate trade execution based on market data | 基於市場數據模擬交易執行
            for symbol, data in market_data.items():
                if data.empty:
                    continue
                
                current_price = data['Close'].iloc[-1]
                
                # Simple simulation - this would be replaced with actual strategy decisions
                # 簡單模擬 - 這將被實際策略決定替換
                if len(self.active_trades) < 3:  # Limit concurrent trades
                    self._open_trade(symbol, 'long', 1000.0, current_price, 0.8)
    
    def _open_trade(self, symbol: str, side: str, quantity: float, 
                   entry_price: float, signal_confidence: float = 0.0):
        """Open a new trade | 開啟新交易"""
        self.trade_counter += 1
        trade_id = f"trade_{self.trade_counter}"
        
        # Calculate transaction costs | 計算交易成本
        commission = quantity * entry_price * self.config.commission_rate
        slippage = self._calculate_slippage(quantity, entry_price)
        
        # Adjust entry price for slippage | 調整入場價格的滑點
        if side == 'long':
            actual_entry_price = entry_price + slippage
        else:
            actual_entry_price = entry_price - slippage
        
        trade = TradeRecord(
            trade_id=trade_id,
            symbol=symbol,
            side=side,
            entry_time=self.current_time,
            entry_price=actual_entry_price,
            quantity=quantity,
            commission=commission,
            slippage=slippage,
            signal_confidence=signal_confidence
        )
        
        self.active_trades[trade_id] = trade
        
        logger.debug(f"Opened trade: {trade_id} {side} {symbol} "
                    f"quantity={quantity} price={actual_entry_price:.5f}")
    
    def _close_trade(self, trade_id: str, exit_price: float, exit_reason: str = "unknown"):
        """Close an active trade | 關閉活躍交易"""
        if trade_id not in self.active_trades:
            logger.warning(f"Attempted to close non-existent trade: {trade_id}")
            return
        
        trade = self.active_trades[trade_id]
        
        # Calculate additional transaction costs for exit | 計算離場的額外交易成本
        exit_commission = trade.quantity * exit_price * self.config.commission_rate
        exit_slippage = self._calculate_slippage(trade.quantity, exit_price)
        
        # Adjust exit price for slippage | 調整離場價格的滑點
        if trade.side == 'long':
            actual_exit_price = exit_price - exit_slippage
        else:
            actual_exit_price = exit_price + exit_slippage
        
        # Update trade record | 更新交易記錄
        trade.exit_time = self.current_time
        trade.exit_price = actual_exit_price
        trade.commission += exit_commission
        trade.slippage += exit_slippage
        trade.exit_reason = exit_reason
        
        # Calculate final PnL | 計算最終盈虧
        trade.calculate_pnl()
        
        # Move to completed trades | 移至已完成交易
        self.completed_trades.append(trade)
        del self.active_trades[trade_id]
        
        logger.debug(f"Closed trade: {trade_id} PnL=${trade.net_pnl:.2f} reason={exit_reason}")
    
    def _calculate_slippage(self, quantity: float, price: float) -> float:
        """Calculate slippage based on order size | 根據訂單大小計算滑點"""
        # Simple slippage model - could be made more sophisticated
        # 簡單滑點模型 - 可以做得更精細
        base_slippage = price * self.config.slippage_factor
        size_adjustment = np.log(1 + quantity / 10000)  # Size impact
        
        return base_slippage * (1 + size_adjustment)
    
    def _record_portfolio_state(self, timestamp: pd.Timestamp):
        """Record current portfolio state | 記錄當前投資組合狀態"""
        self.timestamps.append(timestamp)
        self.portfolio_values.append(self.current_portfolio_value)
        
        # Calculate daily return | 計算日收益
        if len(self.portfolio_values) > 1:
            daily_return = (self.current_portfolio_value / self.portfolio_values[-2]) - 1
            self.daily_returns.append(daily_return)
        else:
            self.daily_returns.append(0.0)
        
        self.equity_history.append({
            'timestamp': timestamp,
            'portfolio_value': self.current_portfolio_value,
            'active_trades': len(self.active_trades),
            'daily_return': self.daily_returns[-1] if self.daily_returns else 0.0
        })
    
    def _finalize_backtest(self, strategy: AIFXTradingStrategy) -> BacktestResults:
        """Finalize backtesting and calculate results | 完成回測並計算結果"""
        # Close any remaining active trades | 關閉任何剩餘的活躍交易
        for trade_id in list(self.active_trades.keys()):
            trade = self.active_trades[trade_id]
            symbol = trade.symbol
            
            if symbol in self.market_data:
                final_price = self.market_data[symbol]['Close'].iloc[-1]
                self._close_trade(trade_id, final_price, "backtest_end")
        
        # Create results object | 創建結果對象
        results = BacktestResults()
        results.strategy_name = strategy.config.strategy_name
        results.start_date = pd.to_datetime(self.config.start_date)
        results.end_date = pd.to_datetime(self.config.end_date)
        results.total_duration_days = (results.end_date - results.start_date).days
        
        # Portfolio performance | 投資組合績效
        results.initial_capital = self.config.initial_capital
        results.final_capital = self.current_portfolio_value
        results.total_return = results.final_capital - results.initial_capital
        results.total_return_pct = (results.final_capital / results.initial_capital - 1) * 100
        
        # Calculate annualized return | 計算年化回報
        years = results.total_duration_days / 365.25
        if years > 0:
            results.annualized_return = ((results.final_capital / results.initial_capital) ** (1/years) - 1) * 100
        
        # Risk metrics | 風險指標
        if self.daily_returns:
            returns_series = pd.Series(self.daily_returns)
            results.volatility = returns_series.std() * np.sqrt(252) * 100  # Annualized volatility
            
            # Sharpe ratio | 夏普比率
            excess_return = results.annualized_return - self.config.risk_free_rate * 100
            if results.volatility > 0:
                results.sharpe_ratio = excess_return / results.volatility
            
            # Sortino ratio | 索提諾比率
            downside_returns = returns_series[returns_series < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std() * np.sqrt(252) * 100
                if downside_std > 0:
                    results.sortino_ratio = excess_return / downside_std
            
            # Maximum drawdown | 最大回撤
            portfolio_series = pd.Series(self.portfolio_values)
            running_max = portfolio_series.expanding().max()
            drawdown = (portfolio_series - running_max) / running_max
            results.max_drawdown = drawdown.min() * results.initial_capital
            results.max_drawdown_pct = drawdown.min() * 100
            
            # Value at Risk | 風險價值
            if self.config.calculate_var:
                results.value_at_risk = np.percentile(returns_series, self.config.var_confidence * 100)
        
        # Trading statistics | 交易統計
        results.trades = self.completed_trades
        results.total_trades = len(self.completed_trades)
        
        if results.total_trades > 0:
            winning_trades = [t for t in self.completed_trades if t.net_pnl > 0]
            losing_trades = [t for t in self.completed_trades if t.net_pnl < 0]
            
            results.winning_trades = len(winning_trades)
            results.losing_trades = len(losing_trades)
            results.win_rate = results.winning_trades / results.total_trades * 100
            
            if winning_trades:
                results.avg_win = sum(t.net_pnl for t in winning_trades) / len(winning_trades)
            
            if losing_trades:
                results.avg_loss = sum(t.net_pnl for t in losing_trades) / len(losing_trades)
                results.avg_loss = abs(results.avg_loss)  # Make positive for reporting
            
            # Profit factor | 獲利因子
            gross_profit = sum(t.net_pnl for t in winning_trades)
            gross_loss = abs(sum(t.net_pnl for t in losing_trades))
            
            if gross_loss > 0:
                results.profit_factor = gross_profit / gross_loss
            else:
                results.profit_factor = float('inf') if gross_profit > 0 else 0.0
        
        # Cost analysis | 成本分析
        results.total_commission = sum(t.commission for t in self.completed_trades)
        results.total_slippage = sum(abs(t.slippage * t.quantity) for t in self.completed_trades)
        results.total_costs = results.total_commission + results.total_slippage
        
        if abs(results.total_return) > 0:
            results.cost_pct_of_returns = (results.total_costs / abs(results.total_return)) * 100
        
        # Create equity curve DataFrame | 創建權益曲線DataFrame
        if self.equity_history:
            results.equity_curve = pd.DataFrame(self.equity_history)
            results.equity_curve.set_index('timestamp', inplace=True)
        
        # Daily returns series | 日收益系列
        if self.daily_returns and self.timestamps:
            results.daily_returns = pd.Series(
                self.daily_returns, 
                index=self.timestamps[1:] if len(self.timestamps) > len(self.daily_returns) else self.timestamps
            )
        
        # Benchmark comparison | 基準比較
        if self.benchmark_data is not None:
            self._calculate_benchmark_comparison(results)
        
        return results
    
    def _calculate_benchmark_comparison(self, results: BacktestResults):
        """Calculate benchmark comparison metrics | 計算基準比較指標"""
        try:
            # Calculate benchmark return | 計算基準回報
            benchmark_start = self.benchmark_data['Close'].iloc[0]
            benchmark_end = self.benchmark_data['Close'].iloc[-1]
            results.benchmark_return = ((benchmark_end / benchmark_start) - 1) * 100
            
            # Excess return | 超額回報
            results.excess_return = results.total_return_pct - results.benchmark_return
            
            # Tracking error and information ratio | 跟踪誤差和信息比率
            if len(results.daily_returns) > 0 and len(self.benchmark_data) > 0:
                benchmark_returns = self.benchmark_data['Close'].pct_change().dropna()
                
                # Align timeframes | 對齊時間框架
                common_dates = results.daily_returns.index.intersection(benchmark_returns.index)
                
                if len(common_dates) > 10:  # Need sufficient data points
                    strategy_returns = results.daily_returns.loc[common_dates]
                    benchmark_returns = benchmark_returns.loc[common_dates]
                    
                    excess_returns = strategy_returns - benchmark_returns
                    results.tracking_error = excess_returns.std() * np.sqrt(252) * 100
                    
                    if results.tracking_error > 0:
                        results.information_ratio = (excess_returns.mean() * 252 * 100) / results.tracking_error
            
        except Exception as e:
            logger.warning(f"Error calculating benchmark comparison: {e}")
    
    def _validate_market_data(self, data: pd.DataFrame, symbol: str) -> bool:
        """Validate market data quality | 驗證市場數據品質"""
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check required columns | 檢查必需列
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check for excessive missing data | 檢查過多缺失數據
        missing_pct = data[required_columns].isnull().sum() / len(data)
        if (missing_pct > 0.1).any():  # More than 10% missing
            logger.warning(f"High missing data percentage for {symbol}")
            return False
        
        # Check for price anomalies | 檢查價格異常
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if (data[col] <= 0).any():
                logger.warning(f"Invalid prices (<=0) found in {symbol}")
                return False
        
        # Check High >= Low | 檢查最高價 >= 最低價
        if (data['High'] < data['Low']).any():
            logger.warning(f"High < Low anomalies found in {symbol}")
            return False
        
        return True
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List[Any]], 
                                       n_trials: int) -> List[Dict[str, Any]]:
        """Generate parameter combinations for optimization | 生成優化的參數組合"""
        # This is a simplified implementation - could use more sophisticated methods like Optuna
        # 這是一個簡化實現 - 可以使用更精細的方法如Optuna
        
        combinations = []
        np.random.seed(42)  # For reproducibility
        
        for _ in range(n_trials):
            combination = {}
            for param_name, param_range in parameter_ranges.items():
                if isinstance(param_range[0], (int, float)):
                    # Numeric parameter | 數值參數
                    min_val, max_val = min(param_range), max(param_range)
                    if isinstance(param_range[0], int):
                        combination[param_name] = np.random.randint(min_val, max_val + 1)
                    else:
                        combination[param_name] = np.random.uniform(min_val, max_val)
                else:
                    # Categorical parameter | 分類參數
                    combination[param_name] = np.random.choice(param_range)
            
            combinations.append(combination)
        
        return combinations
    
    def _create_config_from_parameters(self, params: Dict[str, Any]) -> StrategyConfig:
        """Create strategy configuration from parameters | 從參數創建策略配置"""
        # This would map optimization parameters to strategy configuration
        # 這將把優化參數映射到策略配置
        config = StrategyConfig()
        
        # Example mapping - would be customized for specific parameters
        # 示例映射 - 將為特定參數自定義
        if 'risk_level' in params:
            config.risk_level = params['risk_level']
        if 'min_signal_agreement' in params:
            config.min_signal_agreement = params['min_signal_agreement']
        
        return config
    
    def _run_single_optimization_trial(self, strategy_class: type, 
                                     config: StrategyConfig, trial_id: int) -> Dict[str, float]:
        """Run single optimization trial | 運行單一優化試驗"""
        try:
            # Create strategy instance | 創建策略實例
            strategy = strategy_class(config)
            
            # Run backtest | 運行回測
            results = self.run_backtest(strategy)
            
            # Return key metrics | 返回關鍵指標
            return {
                'total_return_pct': results.total_return_pct,
                'sharpe_ratio': results.sharpe_ratio,
                'max_drawdown_pct': results.max_drawdown_pct,
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor,
                'total_trades': results.total_trades
            }
            
        except Exception as e:
            logger.warning(f"Optimization trial {trial_id} failed: {e}")
            return {
                'total_return_pct': -100.0,
                'sharpe_ratio': -10.0,
                'max_drawdown_pct': 100.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0
            }
    
    def _find_best_optimization_result(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find best optimization result based on configured metric | 根據配置指標找到最佳優化結果"""
        if not results:
            raise ValueError("No optimization results to evaluate")
        
        metric = self.config.optimization_metric
        
        if metric == 'max_drawdown':
            # For drawdown, lower is better | 對於回撤，越低越好
            best_result = min(results, key=lambda x: abs(x['performance'][metric + '_pct']))
        else:
            # For other metrics, higher is better | 對於其他指標，越高越好
            best_result = max(results, key=lambda x: x['performance'][metric])
        
        return best_result
    
    def _generate_comparison_metrics(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate strategy comparison metrics | 生成策略比較指標"""
        valid_results = {name: result for name, result in comparison_results.items() 
                        if 'error' not in result}
        
        if not valid_results:
            return {'error': 'No valid results for comparison'}
        
        # Calculate comparison statistics | 計算比較統計
        metrics = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate']
        comparison_metrics = {}
        
        for metric in metrics:
            values = [result[metric] for result in valid_results.values()]
            comparison_metrics[metric] = {
                'best': max(values) if metric != 'max_drawdown_pct' else min(values),
                'worst': min(values) if metric != 'max_drawdown_pct' else max(values),
                'average': np.mean(values),
                'std': np.std(values)
            }
        
        return comparison_metrics
    
    def _rank_strategies(self, comparison_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank strategies based on performance | 根據績效排名策略"""
        valid_results = [(name, result) for name, result in comparison_results.items() 
                        if 'error' not in result]
        
        # Simple ranking based on Sharpe ratio | 基於夏普比率的簡單排名
        ranked = sorted(valid_results, key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        
        ranking = []
        for i, (name, result) in enumerate(ranked):
            ranking.append({
                'rank': i + 1,
                'strategy_name': name,
                'sharpe_ratio': result['sharpe_ratio'],
                'total_return_pct': result['total_return_pct'],
                'max_drawdown_pct': result['max_drawdown_pct']
            })
        
        return ranking
    
    def _generate_executive_summary(self, results: BacktestResults) -> Dict[str, Any]:
        """Generate executive summary | 生成執行摘要"""
        return {
            'strategy_performance': f"{results.total_return_pct:.2f}% total return over {results.total_duration_days} days",
            'annualized_return': f"{results.annualized_return:.2f}%",
            'risk_metrics': f"Sharpe: {results.sharpe_ratio:.2f}, Max DD: {results.max_drawdown_pct:.2f}%",
            'trading_activity': f"{results.total_trades} trades with {results.win_rate:.1f}% win rate",
            'recommendation': self._get_strategy_recommendation(results)
        }
    
    def _generate_performance_analysis(self, results: BacktestResults) -> Dict[str, Any]:
        """Generate performance analysis | 生成績效分析"""
        return {
            'returns': {
                'total_return': results.total_return,
                'total_return_pct': results.total_return_pct,
                'annualized_return': results.annualized_return,
                'benchmark_comparison': results.excess_return
            },
            'risk_metrics': {
                'volatility': results.volatility,
                'sharpe_ratio': results.sharpe_ratio,
                'sortino_ratio': results.sortino_ratio,
                'max_drawdown': results.max_drawdown_pct,
                'value_at_risk': results.value_at_risk
            },
            'consistency': {
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor,
                'avg_win_loss_ratio': results.avg_win / max(results.avg_loss, 0.01)
            }
        }
    
    def _generate_risk_analysis(self, results: BacktestResults) -> Dict[str, Any]:
        """Generate risk analysis | 生成風險分析"""
        return {
            'drawdown_analysis': {
                'max_drawdown': results.max_drawdown_pct,
                'drawdown_recovery': 'Analysis would require additional calculations'
            },
            'volatility_analysis': {
                'annualized_volatility': results.volatility,
                'downside_deviation': 'Calculated in Sortino ratio'
            },
            'tail_risk': {
                'value_at_risk': results.value_at_risk,
                'expected_shortfall': 'Would require additional calculations'
            }
        }
    
    def _generate_trade_analysis(self, results: BacktestResults) -> Dict[str, Any]:
        """Generate trade analysis | 生成交易分析"""
        if not results.trades:
            return {'message': 'No trades to analyze'}
        
        # Calculate additional trade statistics | 計算額外交易統計
        trade_durations = [t.duration_hours for t in results.trades if t.duration_hours > 0]
        trade_returns = [t.net_pnl / (t.quantity * t.entry_price) for t in results.trades if t.quantity > 0 and t.entry_price > 0]
        
        return {
            'trade_statistics': {
                'total_trades': results.total_trades,
                'winning_trades': results.winning_trades,
                'losing_trades': results.losing_trades,
                'win_rate': results.win_rate,
                'avg_win': results.avg_win,
                'avg_loss': results.avg_loss,
                'profit_factor': results.profit_factor
            },
            'trade_timing': {
                'avg_duration_hours': np.mean(trade_durations) if trade_durations else 0,
                'median_duration_hours': np.median(trade_durations) if trade_durations else 0,
                'longest_trade_hours': max(trade_durations) if trade_durations else 0,
                'shortest_trade_hours': min(trade_durations) if trade_durations else 0
            },
            'return_distribution': {
                'avg_return_pct': np.mean(trade_returns) * 100 if trade_returns else 0,
                'return_std_pct': np.std(trade_returns) * 100 if trade_returns else 0,
                'best_trade_pct': max(trade_returns) * 100 if trade_returns else 0,
                'worst_trade_pct': min(trade_returns) * 100 if trade_returns else 0
            }
        }
    
    def _generate_cost_analysis(self, results: BacktestResults) -> Dict[str, Any]:
        """Generate cost analysis | 生成成本分析"""
        return {
            'transaction_costs': {
                'total_commission': results.total_commission,
                'total_slippage': results.total_slippage,
                'total_costs': results.total_costs,
                'cost_per_trade': results.total_costs / max(results.total_trades, 1),
                'cost_pct_of_returns': results.cost_pct_of_returns
            },
            'cost_impact': {
                'gross_return': results.total_return + results.total_costs,
                'net_return': results.total_return,
                'cost_drag': results.total_costs / results.initial_capital * 100
            }
        }
    
    def _generate_recommendations(self, results: BacktestResults) -> List[str]:
        """Generate strategy recommendations | 生成策略建議"""
        recommendations = []
        
        # Performance-based recommendations | 基於績效的建議
        if results.sharpe_ratio < 0.5:
            recommendations.append("Consider improving risk-adjusted returns (Sharpe ratio < 0.5)")
        
        if results.max_drawdown_pct > 20:
            recommendations.append("High drawdown detected - consider stronger risk management")
        
        if results.win_rate < 40:
            recommendations.append("Low win rate - review entry signal quality")
        
        if results.total_trades < 30:
            recommendations.append("Limited trade sample - extend backtest period for statistical significance")
        
        if results.cost_pct_of_returns > 20:
            recommendations.append("High transaction costs - optimize trade frequency or execution")
        
        # Positive recommendations | 積極建議
        if results.sharpe_ratio > 1.5:
            recommendations.append("Excellent risk-adjusted performance - consider position sizing optimization")
        
        if results.win_rate > 60 and results.profit_factor > 1.5:
            recommendations.append("Strong trade quality metrics - suitable for live trading consideration")
        
        if not recommendations:
            recommendations.append("Strategy shows balanced performance across key metrics")
        
        return recommendations
    
    def _get_strategy_recommendation(self, results: BacktestResults) -> str:
        """Get overall strategy recommendation | 獲取整體策略建議"""
        score = 0
        
        # Score based on key metrics | 基於關鍵指標評分
        if results.sharpe_ratio > 1.0:
            score += 2
        elif results.sharpe_ratio > 0.5:
            score += 1
        
        if results.max_drawdown_pct < 10:
            score += 2
        elif results.max_drawdown_pct < 20:
            score += 1
        
        if results.win_rate > 55:
            score += 1
        
        if results.profit_factor > 1.5:
            score += 1
        
        # Generate recommendation | 生成建議
        if score >= 5:
            return "STRONG BUY - Excellent performance across all metrics"
        elif score >= 3:
            return "BUY - Good performance with acceptable risk"
        elif score >= 1:
            return "HOLD - Marginal performance, needs improvement"
        else:
            return "SELL - Poor performance, significant issues"
    
    def _save_report(self, report: Dict[str, Any], output_path: str):
        """Save report to file | 保存報告到文件"""
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving report: {e}")
            raise