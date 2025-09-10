#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIFX Main Trading System | AIFX 主交易系統
================================

Complete integrated trading system that combines AI models, technical analysis,
and IG Markets API for live forex trading.
結合AI模型、技術分析和IG Markets API的完整整合交易系統，用於實時外匯交易。

This is the main entry point for the AIFX trading system.
這是AIFX交易系統的主要入口點。

Usage:
    python main_trading_system.py --mode=live --config=config/trading-config.yaml
    python main_trading_system.py --mode=backtest --start=2023-01-01 --end=2023-12-31
    python main_trading_system.py --mode=paper --demo=true
"""

import sys
import os
import asyncio
import logging
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import signal
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "main" / "python"))

# Core imports
from core.trading_strategy import AIFXTradingStrategy, StrategyConfig, TradingMode
from core.risk_manager import RiskLevel
from utils.logger import setup_logger
from models.base_model import BaseModel
from brokers.ig_markets import IGMarketsConnector, create_ig_connector

# Initialize logging
logger = logging.getLogger(__name__)


class TradingSystemMode:
    """Trading system operation modes | 交易系統運行模式"""
    LIVE = "live"
    BACKTEST = "backtest"
    PAPER = "paper"
    SIMULATION = "simulation"


class AIFXTradingSystem:
    """
    AIFX Main Trading System | AIFX 主交易系統
    
    Orchestrates the complete trading workflow including:
    - Market data ingestion | 市場數據攝取
    - AI model predictions | AI模型預測
    - Signal generation and combination | 信號生成與組合
    - Risk management | 風險管理
    - Trade execution | 交易執行
    - Performance monitoring | 績效監控
    
    協調完整的交易工作流程。
    """
    
    def __init__(self, config_path: str, mode: str = TradingSystemMode.PAPER):
        """
        Initialize AIFX Trading System | 初始化AIFX交易系統
        
        Args:
            config_path: Path to trading configuration | 交易配置路徑
            mode: Trading system mode | 交易系統模式
        """
        self.config_path = config_path
        self.mode = mode
        self.is_running = False
        self.shutdown_requested = False
        
        # Core components | 核心組件
        self.ig_connector: Optional[IGMarketsConnector] = None
        self.trading_strategy: Optional[AIFXTradingStrategy] = None
        self.ai_models: Dict[str, BaseModel] = {}
        
        # Trading state | 交易狀態
        self.current_positions = {}
        self.active_orders = {}
        self.performance_stats = {}
        self.last_signal_time = None
        
        # System configuration | 系統配置
        self.trading_symbols = ["CS.D.EURUSD.MINI.IP", "CS.D.USDJPY.MINI.IP"]
        self.update_interval = 60  # seconds
        self.max_daily_trades = 10
        self.daily_trade_count = 0
        
        logger.info(f"Initialized AIFX Trading System in {mode} mode")
    
    async def initialize(self) -> bool:
        """
        Initialize all system components | 初始化所有系統組件
        
        Returns:
            bool: True if initialization successful | 初始化成功時返回True
        """
        try:
            logger.info("🚀 Starting AIFX Trading System initialization...")
            
            # 1. Setup signal handlers | 設置信號處理器
            self._setup_signal_handlers()
            
            # 2. Initialize IG Markets API | 初始化IG Markets API
            if not await self._initialize_ig_connector():
                logger.error("Failed to initialize IG Markets connector")
                return False
            
            # 3. Load and initialize AI models | 載入並初始化AI模型
            if not await self._initialize_ai_models():
                logger.warning("AI models initialization failed - continuing without AI signals")
            
            # 4. Initialize trading strategy | 初始化交易策略
            if not await self._initialize_trading_strategy():
                logger.error("Failed to initialize trading strategy")
                return False
            
            # 5. Verify system readiness | 驗證系統就緒狀態
            if not await self._verify_system_readiness():
                logger.error("System readiness verification failed")
                return False
            
            logger.info("✅ AIFX Trading System initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
    
    async def start_trading(self) -> None:
        """
        Start the main trading loop | 啟動主交易循環
        """
        if not self.trading_strategy or not self.ig_connector:
            raise RuntimeError("System not properly initialized")
        
        self.is_running = True
        self.daily_trade_count = 0
        
        logger.info(f"🎯 Starting {self.mode} trading mode...")
        logger.info(f"📊 Monitoring symbols: {self.trading_symbols}")
        logger.info(f"⏱️ Update interval: {self.update_interval} seconds")
        
        try:
            # Start the main trading loop | 啟動主交易循環
            await self._main_trading_loop()
            
        except KeyboardInterrupt:
            logger.info("Trading interrupted by user")
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            raise
        finally:
            await self.shutdown()
    
    async def _main_trading_loop(self) -> None:
        """
        Main trading loop - the heart of the system | 主交易循環 - 系統核心
        """
        logger.info("💓 Main trading loop started")
        
        while self.is_running and not self.shutdown_requested:
            try:
                cycle_start = datetime.now()
                
                # 1. Check daily limits | 檢查每日限制
                if self._check_daily_limits():
                    logger.info(f"Daily trade limit reached ({self.daily_trade_count}/{self.max_daily_trades})")
                    await asyncio.sleep(self.update_interval)
                    continue
                
                # 2. Update market data | 更新市場數據
                market_data_updates = await self._fetch_current_market_data()
                
                if not market_data_updates:
                    logger.warning("No market data received - retrying...")
                    await asyncio.sleep(self.update_interval)
                    continue
                
                # 3. Update position status | 更新倉位狀態
                await self._update_position_status()
                
                # 4. Run strategy cycle | 運行策略週期
                cycle_results = self.trading_strategy.run_strategy_cycle(market_data_updates)
                
                # 5. Process trading signals | 處理交易信號
                if cycle_results.get('decisions_made', 0) > 0:
                    await self._process_trading_signals(cycle_results)
                
                # 6. Update performance metrics | 更新績效指標
                await self._update_performance_metrics()
                
                # 7. Log cycle summary | 記錄循環摘要
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                await self._log_cycle_summary(cycle_results, cycle_duration)
                
                # 8. Sleep until next cycle | 睡眠直到下一個循環
                await asyncio.sleep(max(0, self.update_interval - cycle_duration))
                
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _initialize_ig_connector(self) -> bool:
        """Initialize IG Markets connector | 初始化IG Markets連接器"""
        try:
            logger.info("🔗 Initializing IG Markets API connection...")
            
            self.ig_connector = create_ig_connector(self.config_path)
            
            # Determine if we should use demo account | 確定是否使用模擬帳戶
            use_demo = (self.mode in [TradingSystemMode.PAPER, TradingSystemMode.SIMULATION])
            
            success = await self.ig_connector.connect(demo=use_demo)
            
            if success:
                status = self.ig_connector.get_status()
                logger.info(f"✅ Connected to IG Markets via {self.ig_connector.auth_method}")
                logger.info(f"📊 Account: {status.get('account_info', {}).get('account_name', 'N/A')}")
                return True
            else:
                logger.error("Failed to connect to IG Markets")
                return False
                
        except Exception as e:
            logger.error(f"IG connector initialization failed: {e}")
            return False
    
    async def _initialize_ai_models(self) -> bool:
        """Initialize AI models | 初始化AI模型"""
        try:
            logger.info("🤖 Initializing AI models...")
            
            # Try to load trained models | 嘗試載入訓練好的模型
            from models.xgboost_model import XGBoostModel
            from models.random_forest_model import RandomForestModel
            from models.lstm_model import LSTMModel
            
            model_configs = {
                'xgboost': XGBoostModel,
                'random_forest': RandomForestModel,
                'lstm': LSTMModel
            }
            
            models_loaded = 0
            
            for model_name, model_class in model_configs.items():
                try:
                    model = model_class()
                    
                    # Check if model has been trained | 檢查模型是否已訓練
                    if hasattr(model, 'is_trained') and model.is_trained:
                        self.ai_models[model_name] = model
                        models_loaded += 1
                        logger.info(f"✅ Loaded {model_name} model")
                    else:
                        logger.warning(f"⚠️ {model_name} model not trained - skipping")
                        
                except Exception as e:
                    logger.warning(f"Failed to load {model_name} model: {e}")
            
            if models_loaded > 0:
                logger.info(f"🎯 Successfully loaded {models_loaded} AI models")
                return True
            else:
                logger.warning("No trained AI models found - system will use technical signals only")
                return False
                
        except Exception as e:
            logger.error(f"AI models initialization error: {e}")
            return False
    
    async def _initialize_trading_strategy(self) -> bool:
        """Initialize trading strategy | 初始化交易策略"""
        try:
            logger.info("📊 Initializing trading strategy...")
            
            # Create strategy configuration | 創建策略配置
            strategy_config = StrategyConfig(
                strategy_name="AIFX_Live_Strategy",
                trading_symbols=self.trading_symbols,
                timeframe="1H",
                trading_mode=self._get_trading_mode(),
                enable_ai_signals=(len(self.ai_models) > 0),
                enable_technical_signals=True,
                min_signal_agreement=0.6,
                risk_level=RiskLevel.MODERATE,
                account_balance=100000.0,  # This will be updated from IG account
                max_simultaneous_trades=3,
                enable_live_trading=(self.mode == TradingSystemMode.LIVE),
                enable_notifications=True
            )
            
            # Create strategy instance | 創建策略實例
            self.trading_strategy = AIFXTradingStrategy(strategy_config, self.ai_models)
            
            # Start the strategy | 啟動策略
            if self.trading_strategy.start_strategy():
                logger.info("✅ Trading strategy initialized and started")
                return True
            else:
                logger.error("Failed to start trading strategy")
                return False
                
        except Exception as e:
            logger.error(f"Trading strategy initialization failed: {e}")
            return False
    
    def _get_trading_mode(self) -> TradingMode:
        """Get trading mode enum from string | 從字符串獲取交易模式枚舉"""
        mode_mapping = {
            TradingSystemMode.LIVE: TradingMode.LIVE,
            TradingSystemMode.BACKTEST: TradingMode.BACKTEST,
            TradingSystemMode.PAPER: TradingMode.PAPER,
            TradingSystemMode.SIMULATION: TradingMode.SIMULATION
        }
        return mode_mapping.get(self.mode, TradingMode.PAPER)
    
    async def _verify_system_readiness(self) -> bool:
        """Verify all systems are ready for trading | 驗證所有系統準備就緒"""
        try:
            logger.info("🔍 Verifying system readiness...")
            
            # Check IG connector status | 檢查IG連接器狀態
            if not self.ig_connector or not self.ig_connector.status.value == "connected":
                logger.error("IG connector not ready")
                return False
            
            # Check strategy status | 檢查策略狀態
            if not self.trading_strategy or self.trading_strategy.state.value != "running":
                logger.error("Trading strategy not ready")
                return False
            
            # Test market data access | 測試市場數據訪問
            test_data = await self.ig_connector.get_market_data(self.trading_symbols[0])
            if not test_data or 'bid' not in test_data:
                logger.error("Market data access test failed")
                return False
            
            logger.info("✅ System readiness verification completed")
            return True
            
        except Exception as e:
            logger.error(f"System readiness verification failed: {e}")
            return False
    
    async def _fetch_current_market_data(self) -> Dict[str, Any]:
        """Fetch current market data for all symbols | 獲取所有品種的當前市場數據"""
        market_data = {}
        
        try:
            for symbol in self.trading_symbols:
                data = await self.ig_connector.get_market_data(symbol)
                if data and 'bid' in data:
                    # Convert to DataFrame format expected by strategy | 轉換為策略期望的DataFrame格式
                    import pandas as pd
                    
                    df_data = {
                        'Open': [data['bid']],  # Simplified - would need proper OHLC data
                        'High': [data['bid']],
                        'Low': [data['bid']],
                        'Close': [data['bid']],
                        'Volume': [1000],
                        'timestamp': [datetime.now()]
                    }
                    
                    df = pd.DataFrame(df_data)
                    df.set_index('timestamp', inplace=True)
                    df.index.name = symbol  # Set symbol as index name
                    
                    market_data[symbol] = df
                    
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}
    
    async def _update_position_status(self) -> None:
        """Update current position status | 更新當前倉位狀態"""
        try:
            status = self.ig_connector.get_status()
            self.current_positions = status.get('positions', {})
            
            # Log position updates | 記錄倉位更新
            if self.current_positions:
                logger.debug(f"Current positions: {len(self.current_positions)}")
                for pos_id, position in self.current_positions.items():
                    logger.debug(f"  {pos_id}: {position.get('direction')} {position.get('instrument_name')}")
            
        except Exception as e:
            logger.error(f"Error updating position status: {e}")
    
    async def _process_trading_signals(self, cycle_results: Dict[str, Any]) -> None:
        """Process trading signals and execute trades | 處理交易信號並執行交易"""
        try:
            decisions_made = cycle_results.get('decisions_made', 0)
            logger.info(f"📈 Processing {decisions_made} trading decisions...")
            
            # In a real implementation, this would extract actual decisions from cycle_results
            # For now, we'll simulate based on the results
            # 在實際實現中，這將從cycle_results中提取實際決策
            # 現在我們基於結果進行模擬
            
            if decisions_made > 0:
                # Log signal processing | 記錄信號處理
                signals_generated = cycle_results.get('signals_generated', 0)
                logger.info(f"🎯 Signals generated: {signals_generated}")
                logger.info(f"💼 Decisions made: {decisions_made}")
                
                # Update daily trade count | 更新每日交易計數
                self.daily_trade_count += decisions_made
                self.last_signal_time = datetime.now()
                
                # In live mode, actual trades would be executed here
                # 在實時模式下，實際交易將在此處執行
                if self.mode == TradingSystemMode.LIVE:
                    logger.info("🔴 LIVE MODE: Trade execution would occur here")
                else:
                    logger.info(f"📝 {self.mode.upper()} MODE: Trade simulation completed")
            
        except Exception as e:
            logger.error(f"Error processing trading signals: {e}")
    
    async def _update_performance_metrics(self) -> None:
        """Update system performance metrics | 更新系統績效指標"""
        try:
            # Get strategy status | 獲取策略狀態
            strategy_status = self.trading_strategy.get_strategy_status()
            
            # Update performance stats | 更新績效統計
            self.performance_stats = {
                'total_trades': strategy_status.get('trade_statistics', {}).get('total_trades', 0),
                'success_rate': strategy_status.get('trade_statistics', {}).get('success_rate', 0),
                'daily_trades': self.daily_trade_count,
                'open_positions': len(self.current_positions),
                'last_update': datetime.now(),
                'system_uptime': self._calculate_uptime(),
                'last_signal_time': self.last_signal_time
            }
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {e}")
    
    async def _log_cycle_summary(self, cycle_results: Dict[str, Any], duration: float) -> None:
        """Log trading cycle summary | 記錄交易循環摘要"""
        try:
            summary = {
                'timestamp': datetime.now().isoformat(),
                'duration_seconds': round(duration, 2),
                'signals_generated': cycle_results.get('signals_generated', 0),
                'decisions_made': cycle_results.get('decisions_made', 0),
                'trades_executed': cycle_results.get('trades_executed', 0),
                'errors': len(cycle_results.get('errors', [])),
                'daily_trade_count': self.daily_trade_count,
                'open_positions': len(self.current_positions)
            }
            
            logger.info(f"📊 Cycle Summary: {summary}")
            
            # Log any errors | 記錄任何錯誤
            if cycle_results.get('errors'):
                for error in cycle_results['errors']:
                    logger.warning(f"Cycle error: {error}")
            
        except Exception as e:
            logger.error(f"Error logging cycle summary: {e}")
    
    def _check_daily_limits(self) -> bool:
        """Check if daily trading limits have been reached | 檢查是否達到每日交易限制"""
        return self.daily_trade_count >= self.max_daily_trades
    
    def _calculate_uptime(self) -> str:
        """Calculate system uptime | 計算系統運行時間"""
        # This would track actual start time in a real implementation
        # 在實際實現中，這將追蹤實際啟動時間
        return "N/A - Would track actual uptime"
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown | 設置優雅關閉的信號處理器"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum} - initiating graceful shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the trading system | 優雅地關閉交易系統"""
        logger.info("🛑 Initiating trading system shutdown...")
        
        try:
            self.is_running = False
            
            # Stop trading strategy | 停止交易策略
            if self.trading_strategy:
                self.trading_strategy.stop_strategy()
                logger.info("✅ Trading strategy stopped")
            
            # Disconnect from IG Markets | 斷開IG Markets連接
            if self.ig_connector:
                await self.ig_connector.disconnect()
                logger.info("✅ IG Markets connection closed")
            
            # Log final performance summary | 記錄最終績效摘要
            logger.info("📊 Final Performance Summary:")
            for key, value in self.performance_stats.items():
                logger.info(f"   {key}: {value}")
            
            logger.info("🎯 AIFX Trading System shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status | 獲取當前系統狀態"""
        return {
            'mode': self.mode,
            'is_running': self.is_running,
            'performance_stats': self.performance_stats,
            'trading_symbols': self.trading_symbols,
            'ai_models_loaded': list(self.ai_models.keys()),
            'ig_connector_status': self.ig_connector.status.value if self.ig_connector else 'disconnected',
            'strategy_status': self.trading_strategy.state.value if self.trading_strategy else 'stopped'
        }


async def main():
    """Main entry point for AIFX Trading System | AIFX交易系統主入口點"""
    parser = argparse.ArgumentParser(description='AIFX Trading System')
    parser.add_argument('--mode', choices=['live', 'backtest', 'paper', 'simulation'], 
                       default='paper', help='Trading mode')
    parser.add_argument('--config', default='config/trading-config.yaml', 
                       help='Configuration file path')
    parser.add_argument('--start', help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--demo', action='store_true', help='Use demo account')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging | 設置日誌
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(level=log_level)
    
    logger = logging.getLogger(__name__)
    
    # Display startup banner | 顯示啟動橫幅
    print("\n" + "=" * 80)
    print("🚀 AIFX | AI-Enhanced Forex Trading System")
    print("🚀 AIFX | AI增強外匯交易系統")
    print("=" * 80)
    print(f"Mode | 模式: {args.mode.upper()}")
    print(f"Config | 配置: {args.config}")
    if args.start and args.end:
        print(f"Period | 期間: {args.start} to {args.end}")
    print(f"Demo Account | 模擬帳戶: {args.demo}")
    print("=" * 80)
    
    try:
        # Initialize trading system | 初始化交易系統
        system = AIFXTradingSystem(args.config, args.mode)
        
        # Initialize all components | 初始化所有組件
        if await system.initialize():
            # Start trading | 開始交易
            await system.start_trading()
        else:
            logger.error("System initialization failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Trading system interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Trading system error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))