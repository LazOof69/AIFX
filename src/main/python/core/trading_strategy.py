"""
Trading Strategy Engine | 交易策略引擎

Comprehensive trading strategy implementation that integrates AI models, technical indicators,
signal combination, risk management, and execution logic for forex trading.
綜合交易策略實現，整合AI模型、技術指標、信號組合、風險管理和外匯交易執行邏輯。

This is the main orchestration engine for the AIFX trading system.
這是AIFX交易系統的主要編排引擎。
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
from abc import ABC, abstractmethod

# Core components | 核心組件
from .signal_combiner import BaseSignalCombiner, TradingSignal, SignalType, SignalAggregator
from .ai_signal_combiner import AISignalCombiner
from .technical_signal_combiner import TechnicalSignalCombiner
from .risk_manager import AdvancedRiskManager, RiskParameters, RiskLevel, create_risk_manager_preset
from .confidence_scorer import AdvancedConfidenceScorer, ConfidenceComponents
from models.base_model import BaseModel

# Data processing | 數據處理
from utils.data_preprocessor import DataPreprocessor
from utils.feature_generator import FeatureGenerator

logger = logging.getLogger(__name__)


class TradingMode(Enum):
    """Trading mode enumeration | 交易模式枚舉"""
    LIVE = "live"
    BACKTEST = "backtest"
    PAPER = "paper"
    SIMULATION = "simulation"


class StrategyState(Enum):
    """Strategy execution state | 策略執行狀態"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class StrategyConfig:
    """
    Trading strategy configuration | 交易策略配置
    """
    # Basic settings | 基本設置
    strategy_name: str = "AIFX_Strategy"
    trading_symbols: List[str] = None
    timeframe: str = "1H"
    trading_mode: TradingMode = TradingMode.BACKTEST
    
    # Signal settings | 信號設置
    enable_ai_signals: bool = True
    enable_technical_signals: bool = True
    min_signal_agreement: float = 0.6
    signal_combination_method: str = "weighted_average"
    
    # Risk management | 風險管理
    risk_level: RiskLevel = RiskLevel.MODERATE
    account_balance: float = 100000.0
    max_simultaneous_trades: int = 5
    
    # Execution settings | 執行設置
    enable_live_trading: bool = False
    enable_notifications: bool = True
    log_level: str = "INFO"
    
    # Performance settings | 績效設置
    benchmark_symbol: str = "EURUSD=X"
    performance_update_frequency: int = 100  # trades
    
    def __post_init__(self):
        if self.trading_symbols is None:
            self.trading_symbols = ["EURUSD=X", "USDJPY=X"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary | 轉換為字典"""
        return {
            'strategy_name': self.strategy_name,
            'trading_symbols': self.trading_symbols,
            'timeframe': self.timeframe,
            'trading_mode': self.trading_mode.value,
            'enable_ai_signals': self.enable_ai_signals,
            'enable_technical_signals': self.enable_technical_signals,
            'min_signal_agreement': self.min_signal_agreement,
            'signal_combination_method': self.signal_combination_method,
            'risk_level': self.risk_level.value,
            'account_balance': self.account_balance,
            'max_simultaneous_trades': self.max_simultaneous_trades,
            'enable_live_trading': self.enable_live_trading,
            'enable_notifications': self.enable_notifications,
            'log_level': self.log_level,
            'benchmark_symbol': self.benchmark_symbol,
            'performance_update_frequency': self.performance_update_frequency
        }


@dataclass
class TradingDecision:
    """
    Trading decision output | 交易決定輸出
    """
    action: str  # 'BUY', 'SELL', 'HOLD', 'CLOSE'
    symbol: str
    confidence: float
    position_size: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: str = ""
    risk_metrics: Optional[Dict[str, Any]] = None
    signal_sources: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.signal_sources is None:
            self.signal_sources = []
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies | 交易策略的抽象基類
    """
    
    def __init__(self, config: StrategyConfig):
        """Initialize base strategy | 初始化基礎策略"""
        self.config = config
        self.state = StrategyState.STOPPED
        self.performance_stats = {}
        self.execution_log = []
        
    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame) -> List[TradingSignal]:
        """Generate trading signals | 生成交易信號"""
        pass
    
    @abstractmethod
    def make_trading_decision(self, signals: List[TradingSignal], 
                            market_data: pd.DataFrame) -> List[TradingDecision]:
        """Make trading decisions based on signals | 根據信號做出交易決定"""
        pass
    
    @abstractmethod
    def execute_decision(self, decision: TradingDecision) -> Dict[str, Any]:
        """Execute trading decision | 執行交易決定"""
        pass


class AIFXTradingStrategy(BaseStrategy):
    """
    AIFX comprehensive trading strategy | AIFX綜合交易策略
    
    Integrates AI models, technical analysis, advanced risk management,
    and intelligent signal combination for forex trading.
    整合AI模型、技術分析、高級風險管理和智能信號組合的外匯交易策略。
    """
    
    def __init__(self, config: StrategyConfig, ai_models: Optional[Dict[str, BaseModel]] = None):
        """
        Initialize AIFX trading strategy | 初始化AIFX交易策略
        
        Args:
            config: Strategy configuration | 策略配置
            ai_models: Dictionary of trained AI models | 訓練好的AI模型字典
        """
        super().__init__(config)
        
        # Initialize core components | 初始化核心組件
        self.ai_models = ai_models or {}
        
        # Data processing | 數據處理
        self.data_preprocessor = DataPreprocessor()
        self.feature_generator = FeatureGenerator()
        
        # Signal generation | 信號生成
        self.signal_aggregator = SignalAggregator()
        
        # AI signal combiner | AI信號組合器
        if self.config.enable_ai_signals and self.ai_models:
            self.ai_signal_combiner = AISignalCombiner(self.ai_models)
        else:
            self.ai_signal_combiner = None
            
        # Technical signal combiner | 技術信號組合器
        if self.config.enable_technical_signals:
            self.technical_signal_combiner = TechnicalSignalCombiner()
        else:
            self.technical_signal_combiner = None
        
        # Risk management | 風險管理
        self.risk_manager = create_risk_manager_preset(
            self.config.risk_level, 
            self.config.account_balance
        )
        
        # Advanced confidence scoring | 高級信心評分
        self.confidence_scorer = AdvancedConfidenceScorer(
            lookback_periods=100,
            volatility_window=20,
            agreement_threshold=self.config.min_signal_agreement
        )
        
        # Performance tracking | 績效追蹤
        self.trade_count = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        self.last_performance_update = 0
        
        # Market data cache | 市場數據緩存
        self.market_data_cache = {}
        self.last_data_update = {}
        
        logger.info(f"Initialized AIFX Trading Strategy: {config.strategy_name}")
        logger.info(f"AI Models: {list(self.ai_models.keys())}")
        logger.info(f"Trading Symbols: {config.trading_symbols}")
        logger.info(f"Risk Level: {config.risk_level.value}")
    
    def add_ai_model(self, model_name: str, model: BaseModel):
        """
        Add AI model to the strategy | 添加AI模型到策略
        
        Args:
            model_name: Name identifier for the model | 模型名稱標識符
            model: Trained AI model instance | 訓練好的AI模型實例
        """
        if not model.is_trained:
            raise ValueError(f"Model {model_name} must be trained before adding to strategy")
        
        self.ai_models[model_name] = model
        
        # Reinitialize AI signal combiner if needed | 如果需要則重新初始化AI信號組合器
        if self.config.enable_ai_signals and not self.ai_signal_combiner:
            self.ai_signal_combiner = AISignalCombiner(self.ai_models)
        elif self.ai_signal_combiner:
            self.ai_signal_combiner.add_model(model_name, model)
        
        logger.info(f"Added AI model {model_name} to strategy")
    
    def update_market_data(self, symbol: str, data: pd.DataFrame):
        """
        Update market data cache | 更新市場數據緩存
        
        Args:
            symbol: Trading symbol | 交易品種
            data: Market data DataFrame | 市場數據DataFrame
        """
        # Validate data | 驗證數據
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Preprocess data | 預處理數據
        try:
            processed_data = self.data_preprocessor.preprocess_data(data)
            self.market_data_cache[symbol] = processed_data
            self.last_data_update[symbol] = datetime.now()
            logger.debug(f"Updated market data for {symbol}: {len(data)} rows")
        except Exception as e:
            logger.error(f"Error updating market data for {symbol}: {e}")
            raise
    
    def generate_signals(self, market_data: pd.DataFrame) -> List[TradingSignal]:
        """
        Generate comprehensive trading signals | 生成綜合交易信號
        
        Combines AI model predictions with technical analysis signals.
        結合AI模型預測和技術分析信號。
        
        Args:
            market_data: Market data for signal generation | 用於信號生成的市場數據
            
        Returns:
            List of combined trading signals | 組合交易信號列表
        """
        all_signals = []
        
        try:
            # Clear previous signals | 清除先前信號
            self.signal_aggregator.clear()
            
            # Generate features for AI models | 為AI模型生成特徵
            features = None
            if self.config.enable_ai_signals and self.ai_signal_combiner:
                try:
                    features = self.feature_generator.generate_features(market_data)
                    ai_signals = self.ai_signal_combiner.convert_ai_predictions_to_signals(features)
                    
                    for signal in ai_signals:
                        self.signal_aggregator.add_signal(signal)
                        all_signals.append(signal)
                    
                    logger.debug(f"Generated {len(ai_signals)} AI signals")
                    
                except Exception as e:
                    logger.warning(f"Error generating AI signals: {e}")
            
            # Generate technical signals | 生成技術信號
            if self.config.enable_technical_signals and self.technical_signal_combiner:
                try:
                    tech_signals = self.technical_signal_combiner.convert_technical_indicators_to_signals(market_data)
                    
                    for signal in tech_signals:
                        self.signal_aggregator.add_signal(signal)
                        all_signals.append(signal)
                    
                    logger.debug(f"Generated {len(tech_signals)} technical signals")
                    
                except Exception as e:
                    logger.warning(f"Error generating technical signals: {e}")
            
            # Log signal summary | 記錄信號摘要
            if all_signals:
                signal_summary = self.signal_aggregator.get_summary()
                logger.info(f"Signal Summary: {signal_summary}")
            else:
                logger.warning("No signals generated")
            
            return all_signals
            
        except Exception as e:
            logger.error(f"Error in signal generation: {e}")
            return []
    
    def make_trading_decision(self, signals: List[TradingSignal], 
                            market_data: pd.DataFrame) -> List[TradingDecision]:
        """
        Make intelligent trading decisions | 做出智能交易決定
        
        Combines signals, evaluates risk, and generates executable trading decisions.
        結合信號、評估風險並生成可執行的交易決定。
        
        Args:
            signals: List of trading signals | 交易信號列表
            market_data: Current market data | 當前市場數據
            
        Returns:
            List of trading decisions | 交易決定列表
        """
        decisions = []
        
        if not signals:
            logger.debug("No signals provided for decision making")
            return decisions
        
        try:
            # Get current price | 獲取當前價格
            current_price = market_data['Close'].iloc[-1] if not market_data.empty else 0.0
            symbol = market_data.index.name or "DEFAULT"  # Assume symbol from index name
            
            # 1. Combine signals using appropriate combiner | 使用適當的組合器組合信號
            combined_signal = self._combine_all_signals(signals, market_data)
            
            if combined_signal is None or combined_signal.signal_type == SignalType.HOLD:
                logger.debug("Combined signal indicates HOLD")
                return decisions
            
            # 2. Evaluate risk for the potential trade | 評估潛在交易的風險
            risk_metrics = self.risk_manager.evaluate_trade_risk(
                combined_signal, market_data, current_price
            )
            
            # 3. Check if trade is approved by risk management | 檢查交易是否被風險管理批准
            if not risk_metrics.risk_approval:
                logger.info(f"Trade rejected by risk management: {risk_metrics.risk_warnings}")
                return decisions
            
            # 4. Create trading decision | 創建交易決定
            action = "BUY" if combined_signal.signal_type == SignalType.BUY else "SELL"
            
            decision = TradingDecision(
                action=action,
                symbol=symbol,
                confidence=combined_signal.confidence,
                position_size=risk_metrics.position_size,
                entry_price=current_price,
                stop_loss=risk_metrics.stop_loss_level,
                take_profit=risk_metrics.take_profit_level,
                reasoning=self._generate_decision_reasoning(combined_signal, risk_metrics, signals),
                risk_metrics=risk_metrics.__dict__,
                signal_sources=[s.source for s in signals],
                timestamp=datetime.now()
            )
            
            decisions.append(decision)
            
            logger.info(f"Generated trading decision: {action} {symbol} "
                       f"(confidence={combined_signal.confidence:.3f}, "
                       f"size=${risk_metrics.position_size:.2f})")
            
        except Exception as e:
            logger.error(f"Error making trading decision: {e}")
        
        return decisions
    
    def execute_decision(self, decision: TradingDecision) -> Dict[str, Any]:
        """
        Execute trading decision | 執行交易決定
        
        Args:
            decision: Trading decision to execute | 要執行的交易決定
            
        Returns:
            Execution result | 執行結果
        """
        execution_result = {
            'decision_id': id(decision),
            'symbol': decision.symbol,
            'action': decision.action,
            'requested_size': decision.position_size,
            'executed': False,
            'error': None,
            'execution_time': datetime.now()
        }
        
        try:
            if self.state != StrategyState.RUNNING:
                execution_result['error'] = f"Strategy not running (state: {self.state.value})"
                return execution_result
            
            # Check if this is a position opening decision | 檢查這是否是開倉決定
            if decision.action in ['BUY', 'SELL']:
                # Add position to risk manager tracking | 添加倉位到風險管理器追蹤
                side = 'long' if decision.action == 'BUY' else 'short'
                
                self.risk_manager.add_position(
                    symbol=decision.symbol,
                    side=side,
                    size=decision.position_size,
                    entry_price=decision.entry_price,
                    stop_loss=decision.stop_loss,
                    take_profit=decision.take_profit,
                    signal_confidence=decision.confidence
                )
                
                # Update execution result | 更新執行結果
                execution_result.update({
                    'executed': True,
                    'executed_size': decision.position_size,
                    'entry_price': decision.entry_price,
                    'stop_loss': decision.stop_loss,
                    'take_profit': decision.take_profit
                })
                
                # Update trade statistics | 更新交易統計
                self.trade_count += 1
                
                logger.info(f"Executed {decision.action}: {decision.symbol} "
                           f"size=${decision.position_size:.2f} @ {decision.entry_price:.5f}")
            
            elif decision.action == 'CLOSE':
                # Close existing position | 關閉現有倉位
                if decision.symbol in self.risk_manager.open_positions:
                    self.risk_manager.close_position(
                        decision.symbol, 
                        decision.entry_price,  # Using entry_price as exit_price for close
                        "strategy_signal"
                    )
                    execution_result['executed'] = True
                    logger.info(f"Closed position: {decision.symbol}")
                else:
                    execution_result['error'] = f"No open position for {decision.symbol}"
            
            # Record execution in log | 在日誌中記錄執行
            self.execution_log.append(execution_result.copy())
            
            # Update performance if needed | 如果需要則更新績效
            if self.trade_count - self.last_performance_update >= self.config.performance_update_frequency:
                self._update_performance_stats()
                self.last_performance_update = self.trade_count
            
        except Exception as e:
            logger.error(f"Error executing decision: {e}")
            execution_result['error'] = str(e)
            execution_result['executed'] = False
        
        return execution_result
    
    def run_strategy_cycle(self, market_data_updates: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Run a complete strategy cycle | 運行完整的策略週期
        
        Args:
            market_data_updates: Dictionary of symbol -> market data | 品種->市場數據的字典
            
        Returns:
            Cycle execution results | 週期執行結果
        """
        cycle_results = {
            'cycle_start': datetime.now(),
            'signals_generated': 0,
            'decisions_made': 0,
            'trades_executed': 0,
            'errors': [],
            'positions_updated': False,
            'risk_summary': None
        }
        
        try:
            if self.state != StrategyState.RUNNING:
                cycle_results['errors'].append(f"Strategy not running (state: {self.state.value})")
                return cycle_results
            
            all_decisions = []
            
            # Process each symbol | 處理每個品種
            for symbol, market_data in market_data_updates.items():
                try:
                    # Update market data cache | 更新市場數據緩存
                    self.update_market_data(symbol, market_data)
                    
                    # Generate signals | 生成信號
                    signals = self.generate_signals(market_data)
                    cycle_results['signals_generated'] += len(signals)
                    
                    # Make trading decisions | 做出交易決定
                    decisions = self.make_trading_decision(signals, market_data)
                    cycle_results['decisions_made'] += len(decisions)
                    
                    all_decisions.extend(decisions)
                    
                except Exception as e:
                    error_msg = f"Error processing {symbol}: {str(e)}"
                    cycle_results['errors'].append(error_msg)
                    logger.error(error_msg)
            
            # Execute all approved decisions | 執行所有批准的決定
            for decision in all_decisions:
                try:
                    execution_result = self.execute_decision(decision)
                    if execution_result['executed']:
                        cycle_results['trades_executed'] += 1
                    else:
                        cycle_results['errors'].append(
                            f"Failed to execute {decision.action} for {decision.symbol}: "
                            f"{execution_result.get('error', 'Unknown error')}"
                        )
                except Exception as e:
                    error_msg = f"Execution error for {decision.symbol}: {str(e)}"
                    cycle_results['errors'].append(error_msg)
                    logger.error(error_msg)
            
            # Update trailing stops and check exit conditions | 更新移動止損並檢查離場條件
            try:
                # Update trailing stops | 更新移動止損
                combined_market_data = pd.concat(market_data_updates.values(), ignore_index=True)
                self.risk_manager.update_trailing_stops(combined_market_data)
                
                # Check exit conditions | 檢查離場條件
                exit_signals = self.risk_manager.check_exit_conditions(combined_market_data)
                for exit_signal in exit_signals:
                    try:
                        self.risk_manager.close_position(
                            exit_signal['symbol'],
                            exit_signal['exit_price'],
                            exit_signal['exit_reason']
                        )
                        cycle_results['trades_executed'] += 1
                        logger.info(f"Auto-closed position {exit_signal['symbol']} "
                                   f"due to {exit_signal['exit_reason']}")
                    except Exception as e:
                        cycle_results['errors'].append(f"Error closing position {exit_signal['symbol']}: {str(e)}")
                
                cycle_results['positions_updated'] = True
                
            except Exception as e:
                cycle_results['errors'].append(f"Error updating positions: {str(e)}")
            
            # Get risk summary | 獲取風險摘要
            try:
                cycle_results['risk_summary'] = self.risk_manager.get_risk_summary()
            except Exception as e:
                cycle_results['errors'].append(f"Error getting risk summary: {str(e)}")
            
        except Exception as e:
            cycle_results['errors'].append(f"Critical cycle error: {str(e)}")
            logger.error(f"Critical error in strategy cycle: {e}")
        
        cycle_results['cycle_end'] = datetime.now()
        cycle_results['cycle_duration'] = (cycle_results['cycle_end'] - cycle_results['cycle_start']).total_seconds()
        
        return cycle_results
    
    def start_strategy(self) -> bool:
        """
        Start the trading strategy | 啟動交易策略
        
        Returns:
            True if started successfully | 如果啟動成功則為True
        """
        try:
            if self.state == StrategyState.RUNNING:
                logger.warning("Strategy is already running")
                return True
            
            # Validate configuration | 驗證配置
            validation_errors = self._validate_strategy_config()
            if validation_errors:
                logger.error(f"Strategy configuration errors: {validation_errors}")
                self.state = StrategyState.ERROR
                return False
            
            # Initialize components | 初始化組件
            self._initialize_strategy_components()
            
            self.state = StrategyState.RUNNING
            logger.info(f"Started trading strategy: {self.config.strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting strategy: {e}")
            self.state = StrategyState.ERROR
            return False
    
    def stop_strategy(self) -> bool:
        """
        Stop the trading strategy | 停止交易策略
        
        Returns:
            True if stopped successfully | 如果停止成功則為True
        """
        try:
            if self.state == StrategyState.STOPPED:
                logger.warning("Strategy is already stopped")
                return True
            
            # Close all open positions if in live mode | 如果在實盤模式則關閉所有開倉
            if self.config.trading_mode == TradingMode.LIVE:
                self._close_all_positions("strategy_shutdown")
            
            self.state = StrategyState.STOPPED
            logger.info(f"Stopped trading strategy: {self.config.strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping strategy: {e}")
            self.state = StrategyState.ERROR
            return False
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """
        Get comprehensive strategy status | 獲取綜合策略狀態
        
        Returns:
            Strategy status dictionary | 策略狀態字典
        """
        return {
            'strategy_name': self.config.strategy_name,
            'state': self.state.value,
            'trading_mode': self.config.trading_mode.value,
            'uptime': self._calculate_uptime(),
            'trade_statistics': {
                'total_trades': self.trade_count,
                'successful_trades': self.successful_trades,
                'success_rate': self.successful_trades / max(1, self.trade_count),
                'total_pnl': self.total_pnl
            },
            'open_positions': len(self.risk_manager.open_positions),
            'risk_summary': self.risk_manager.get_risk_summary(),
            'configuration': self.config.to_dict(),
            'last_execution_time': self.execution_log[-1]['execution_time'] if self.execution_log else None,
            'components_status': {
                'ai_signal_combiner': self.ai_signal_combiner is not None,
                'technical_signal_combiner': self.technical_signal_combiner is not None,
                'ai_models_loaded': len(self.ai_models),
                'data_cache_symbols': list(self.market_data_cache.keys())
            }
        }
    
    # Private helper methods | 私有輔助方法
    
    def _combine_all_signals(self, signals: List[TradingSignal], 
                           market_data: pd.DataFrame) -> Optional[TradingSignal]:
        """Combine all signals into a single trading signal | 將所有信號組合成單一交易信號"""
        if not signals:
            return None
        
        # Separate AI and technical signals | 分離AI和技術信號
        ai_signals = [s for s in signals if s.source.startswith('AI_')]
        tech_signals = [s for s in signals if not s.source.startswith('AI_')]
        
        combined_signals = []
        
        # Combine AI signals if available | 如果可用則組合AI信號
        if ai_signals and self.ai_signal_combiner:
            try:
                ai_combined = self.ai_signal_combiner.combine_signals(ai_signals)
                combined_signals.append(ai_combined)
            except Exception as e:
                logger.warning(f"Error combining AI signals: {e}")
        
        # Combine technical signals if available | 如果可用則組合技術信號
        if tech_signals and self.technical_signal_combiner:
            try:
                tech_combined = self.technical_signal_combiner.combine_signals(tech_signals)
                combined_signals.append(tech_combined)
            except Exception as e:
                logger.warning(f"Error combining technical signals: {e}")
        
        # Final combination of AI and technical combined signals | AI和技術組合信號的最終組合
        if not combined_signals:
            return None
        elif len(combined_signals) == 1:
            return combined_signals[0]
        else:
            # Use simple averaging for final combination | 使用簡單平均進行最終組合
            return self._simple_signal_combination(combined_signals)
    
    def _simple_signal_combination(self, signals: List[TradingSignal]) -> TradingSignal:
        """Simple signal combination method | 簡單信號組合方法"""
        if not signals:
            return TradingSignal(SignalType.HOLD, 0.5, 0.0, "Empty")
        
        # Calculate weighted average | 計算權重平均
        weighted_values = []
        total_confidence = 0.0
        
        for signal in signals:
            weighted_value = signal.get_weighted_signal()
            weighted_values.append(weighted_value)
            total_confidence += signal.confidence
        
        combined_value = np.mean(weighted_values)
        avg_confidence = total_confidence / len(signals)
        
        # Determine signal type | 確定信號類型
        if combined_value > 0.1:
            signal_type = SignalType.BUY
            strength = min(abs(combined_value), 1.0)
        elif combined_value < -0.1:
            signal_type = SignalType.SELL
            strength = min(abs(combined_value), 1.0)
        else:
            signal_type = SignalType.HOLD
            strength = 0.5
        
        return TradingSignal(
            signal_type=signal_type,
            strength=strength,
            confidence=avg_confidence,
            source="Final_Combined",
            timestamp=datetime.now(),
            metadata={
                'combination_method': 'simple_average',
                'input_signals': len(signals),
                'combined_value': combined_value
            }
        )
    
    def _generate_decision_reasoning(self, combined_signal: TradingSignal, 
                                   risk_metrics: Any, signals: List[TradingSignal]) -> str:
        """Generate human-readable reasoning for trading decision | 生成交易決定的人可讀推理"""
        reasoning_parts = []
        
        # Signal analysis | 信號分析
        reasoning_parts.append(f"Combined signal: {combined_signal.signal_type.name} "
                              f"(strength={combined_signal.strength:.2f}, "
                              f"confidence={combined_signal.confidence:.2f})")
        
        # Source analysis | 來源分析
        signal_sources = list(set(s.source for s in signals))
        reasoning_parts.append(f"Based on {len(signals)} signals from {len(signal_sources)} sources: "
                              f"{', '.join(signal_sources)}")
        
        # Risk analysis | 風險分析
        reasoning_parts.append(f"Risk-reward ratio: {risk_metrics.risk_reward_ratio:.2f}")
        reasoning_parts.append(f"Position size: ${risk_metrics.position_size:.2f}")
        
        return " | ".join(reasoning_parts)
    
    def _validate_strategy_config(self) -> List[str]:
        """Validate strategy configuration | 驗證策略配置"""
        errors = []
        
        if not self.config.trading_symbols:
            errors.append("No trading symbols specified")
        
        if self.config.enable_ai_signals and not self.ai_models:
            errors.append("AI signals enabled but no AI models provided")
        
        if self.config.account_balance <= 0:
            errors.append("Invalid account balance")
        
        if self.config.min_signal_agreement < 0 or self.config.min_signal_agreement > 1:
            errors.append("Invalid signal agreement threshold")
        
        return errors
    
    def _initialize_strategy_components(self):
        """Initialize strategy components | 初始化策略組件"""
        # Reset performance tracking | 重置績效追蹤
        self.trade_count = 0
        self.successful_trades = 0
        self.total_pnl = 0.0
        self.execution_log.clear()
        
        # Initialize risk manager | 初始化風險管理器
        self.risk_manager.current_balance = self.config.account_balance
        
        logger.info("Strategy components initialized")
    
    def _update_performance_stats(self):
        """Update performance statistics | 更新績效統計"""
        if not self.risk_manager.trade_history:
            return
        
        # Calculate basic stats | 計算基本統計
        total_trades = len(self.risk_manager.trade_history)
        winning_trades = [t for t in self.risk_manager.trade_history if t['pnl'] > 0]
        
        self.trade_count = total_trades
        self.successful_trades = len(winning_trades)
        self.total_pnl = sum(t['pnl'] for t in self.risk_manager.trade_history)
        
        logger.info(f"Performance Update: {total_trades} trades, "
                   f"{len(winning_trades)} wins, ${self.total_pnl:.2f} PnL")
    
    def _close_all_positions(self, reason: str):
        """Close all open positions | 關閉所有開倉"""
        for symbol in list(self.risk_manager.open_positions.keys()):
            try:
                position = self.risk_manager.open_positions[symbol]
                self.risk_manager.close_position(symbol, position.current_price, reason)
                logger.info(f"Closed position {symbol} due to {reason}")
            except Exception as e:
                logger.error(f"Error closing position {symbol}: {e}")
    
    def _calculate_uptime(self) -> str:
        """Calculate strategy uptime | 計算策略運行時間"""
        # This would be implemented with actual start time tracking
        # For now, return a placeholder | 現在返回佔位符
        return "N/A"