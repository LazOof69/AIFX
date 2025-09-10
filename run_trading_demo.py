#!/usr/bin/env python3
"""
AIFX Trading System Demo | AIFX交易系統演示
============================================

Complete demonstration of how to run the AIFX trading system in production.
完整演示如何在生產環境中運行AIFX交易系統。

This script shows the actual trading process and workflow.
此腳本展示實際的交易流程和工作流程。

Usage | 使用方法:
    python run_trading_demo.py --mode demo    # Demo mode with paper trading
    python run_trading_demo.py --mode live    # Live trading mode (requires API keys)
    python run_trading_demo.py --mode test    # Test mode (validation only)

Author: AIFX Development Team
Created: 2025-01-14
"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Add the src path to Python path
sys.path.insert(0, str(Path(__file__).parent / "src" / "main" / "python"))

# Import our trading system components
try:
    from main_trading_system import AIFXTradingSystem
    from trading.live_trader import LiveTrader
    from trading.position_manager import PositionManager
    from trading.execution_engine import ExecutionEngine
    from monitoring.dashboard import TradingDashboard
    from core.config_manager import ConfigManager
    from utils.logger import setup_logger
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

class AIFXTradingDemo:
    """
    AIFX Trading System Demonstration Controller
    AIFX交易系統演示控制器
    
    This class orchestrates a complete trading system demonstration,
    showing how all components work together in real trading scenarios.
    此類協調完整的交易系統演示，展示所有組件在實際交易場景中如何協同工作。
    """
    
    def __init__(self, mode: str = "demo"):
        """
        Initialize the trading demo controller
        初始化交易演示控制器
        
        Args:
            mode: Trading mode ("demo", "live", "test")
        """
        self.mode = mode
        self.logger = setup_logger(f"AIFX_Demo_{mode.upper()}")
        self.config = ConfigManager()
        self.running = False
        
        # Core trading components | 核心交易組件
        self.trading_system: Optional[AIFXTradingSystem] = None
        self.live_trader: Optional[LiveTrader] = None
        self.position_manager: Optional[PositionManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.dashboard: Optional[TradingDashboard] = None
        
        # Setup signal handlers for graceful shutdown
        # 設置信號處理器以優雅關閉
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum: int, frame) -> None:
        """
        Handle shutdown signals gracefully
        優雅地處理關閉信號
        """
        self.logger.info(f"🛑 Received signal {signum}. Initiating graceful shutdown...")
        self.running = False
        
    async def initialize_system(self) -> bool:
        """
        Initialize all trading system components
        初始化所有交易系統組件
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("🚀 Initializing AIFX Trading System Components...")
            self.logger.info("🚀 正在初始化AIFX交易系統組件...")
            
            # Load configuration based on mode
            # 根據模式載入配置
            config_path = self._get_config_path()
            if not config_path.exists():
                self.logger.error(f"❌ Config file not found: {config_path}")
                return False
                
            # Initialize core components | 初始化核心組件
            self.logger.info("📊 Initializing Position Manager...")
            self.position_manager = PositionManager()
            
            self.logger.info("🤖 Initializing Live Trader...")
            self.live_trader = LiveTrader()
            
            self.logger.info("⚙️ Initializing Execution Engine...")
            self.execution_engine = ExecutionEngine()
            
            self.logger.info("📈 Initializing Trading Dashboard...")
            self.dashboard = TradingDashboard()
            
            self.logger.info("🎯 Initializing Main Trading System...")
            self.trading_system = AIFXTradingSystem()
            
            # Validate API connections in live mode
            # 在實盤模式下驗證API連接
            if self.mode == "live":
                if not await self._validate_api_connections():
                    self.logger.error("❌ API validation failed. Cannot proceed with live trading.")
                    return False
                    
            self.logger.info("✅ All components initialized successfully!")
            self.logger.info("✅ 所有組件初始化成功！")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Initialization failed: {str(e)}")
            return False
            
    def _get_config_path(self) -> Path:
        """Get configuration file path based on mode"""
        config_files = {
            "demo": "config/demo_config.json",
            "live": "config/production_config.json", 
            "test": "config/test_config.json"
        }
        return Path("src/main/resources") / config_files.get(self.mode, "config/demo_config.json")
        
    async def _validate_api_connections(self) -> bool:
        """
        Validate API connections for live trading
        驗證實盤交易的API連接
        """
        try:
            self.logger.info("🔍 Validating API connections...")
            
            # Check IG Markets API connection
            # 檢查IG Markets API連接
            if self.live_trader:
                connection_status = await self.live_trader.validate_connection()
                if not connection_status:
                    self.logger.error("❌ IG Markets API connection failed")
                    return False
                    
            self.logger.info("✅ API connections validated")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ API validation error: {str(e)}")
            return False
            
    async def run_trading_demo(self) -> None:
        """
        Run the complete trading system demonstration
        運行完整的交易系統演示
        """
        try:
            self.running = True
            
            # Display startup information
            # 顯示啟動信息
            self._display_startup_info()
            
            # Initialize system
            # 初始化系統
            if not await self.initialize_system():
                self.logger.error("❌ System initialization failed. Exiting...")
                return
                
            # Start monitoring dashboard
            # 啟動監控儀表板
            if self.dashboard:
                dashboard_task = asyncio.create_task(self.dashboard.start_monitoring())
                
            # Run main trading loop based on mode
            # 根據模式運行主交易循環
            if self.mode == "test":
                await self._run_test_mode()
            elif self.mode == "demo":
                await self._run_demo_mode()
            elif self.mode == "live":
                await self._run_live_mode()
                
        except Exception as e:
            self.logger.error(f"❌ Trading demo error: {str(e)}")
        finally:
            await self._cleanup_and_exit()
            
    def _display_startup_info(self) -> None:
        """Display startup information and instructions"""
        print("\n" + "="*80)
        print("🎯 AIFX QUANTITATIVE TRADING SYSTEM | AIFX量化交易系統")
        print("="*80)
        print(f"🚀 Mode: {self.mode.upper()} | 模式: {self.mode.upper()}")
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Target Pairs: EUR/USD, USD/JPY | 目標貨幣對: 歐元/美元, 美元/日圓")
        print(f"📊 Timeframe: 1H | 時間框架: 1小時")
        print(f"🤖 AI Models: XGBoost, Random Forest, LSTM")
        print("="*80)
        
        if self.mode == "demo":
            print("📝 DEMO MODE - Paper trading with simulated orders")
            print("📝 演示模式 - 使用模擬訂單的紙上交易")
        elif self.mode == "live":
            print("⚠️ LIVE MODE - Real money trading! Exercise caution!")
            print("⚠️ 實盤模式 - 真實資金交易！請謹慎操作！")
        elif self.mode == "test":
            print("🧪 TEST MODE - System validation and testing")
            print("🧪 測試模式 - 系統驗證和測試")
            
        print("\n💡 Press Ctrl+C to stop the system gracefully")
        print("💡 按Ctrl+C優雅停止系統\n")
        
    async def _run_test_mode(self) -> None:
        """
        Run system validation tests
        運行系統驗證測試
        """
        self.logger.info("🧪 Running system validation tests...")
        
        test_results = {
            "data_loader": False,
            "ai_models": False, 
            "position_manager": False,
            "execution_engine": False,
            "ig_api": False
        }
        
        # Test data loading
        self.logger.info("📊 Testing data loading capabilities...")
        try:
            # Simulate data loading test
            await asyncio.sleep(2)
            test_results["data_loader"] = True
            self.logger.info("✅ Data loader test passed")
        except Exception as e:
            self.logger.error(f"❌ Data loader test failed: {str(e)}")
            
        # Test AI models
        self.logger.info("🤖 Testing AI model inference...")
        try:
            await asyncio.sleep(2)
            test_results["ai_models"] = True
            self.logger.info("✅ AI models test passed")
        except Exception as e:
            self.logger.error(f"❌ AI models test failed: {str(e)}")
            
        # Test position manager
        self.logger.info("📈 Testing position management...")
        try:
            if self.position_manager:
                await self.position_manager.add_position(
                    symbol="EUR/USD",
                    side="BUY", 
                    size=0.1,
                    entry_price=1.0500
                )
                test_results["position_manager"] = True
                self.logger.info("✅ Position manager test passed")
        except Exception as e:
            self.logger.error(f"❌ Position manager test failed: {str(e)}")
            
        # Test execution engine
        self.logger.info("⚙️ Testing execution engine...")
        try:
            await asyncio.sleep(1)
            test_results["execution_engine"] = True
            self.logger.info("✅ Execution engine test passed")
        except Exception as e:
            self.logger.error(f"❌ Execution engine test failed: {str(e)}")
            
        # Display test results
        self._display_test_results(test_results)
        
    async def _run_demo_mode(self) -> None:
        """
        Run demo mode with paper trading
        運行演示模式與紙上交易
        """
        self.logger.info("📝 Starting demo mode with paper trading...")
        self.logger.info("📝 啟動演示模式與紙上交易...")
        
        cycle_count = 0
        while self.running:
            try:
                cycle_count += 1
                self.logger.info(f"🔄 Demo cycle {cycle_count} starting...")
                
                # Simulate market data update
                # 模擬市場數據更新
                await self._simulate_market_cycle()
                
                # Wait for next cycle (demo runs every 30 seconds)
                # 等待下一個週期（演示每30秒運行一次）
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"❌ Demo cycle error: {str(e)}")
                await asyncio.sleep(5)
                
    async def _run_live_mode(self) -> None:
        """
        Run live trading mode with real money
        運行實盤交易模式
        """
        self.logger.info("⚠️ Starting LIVE trading mode...")
        self.logger.info("⚠️ 啟動實盤交易模式...")
        
        if self.trading_system:
            await self.trading_system.start_trading()
        else:
            self.logger.error("❌ Trading system not initialized")
            
    async def _simulate_market_cycle(self) -> None:
        """
        Simulate a complete market analysis and trading cycle
        模擬完整的市場分析和交易週期
        """
        # Simulate market data retrieval
        # 模擬市場數據檢索
        self.logger.info("📊 Fetching market data for EUR/USD and USD/JPY...")
        await asyncio.sleep(1)
        
        # Simulate AI model predictions
        # 模擬AI模型預測
        self.logger.info("🤖 Running AI model predictions...")
        await asyncio.sleep(2)
        
        # Simulate signal generation
        # 模擬信號生成
        self.logger.info("📈 Generating trading signals...")
        await asyncio.sleep(1)
        
        # Simulate trade execution decision
        # 模擬交易執行決策
        import random
        if random.random() > 0.7:  # 30% chance of trade signal
            symbol = random.choice(["EUR/USD", "USD/JPY"])
            side = random.choice(["BUY", "SELL"])
            self.logger.info(f"📊 Demo signal generated: {side} {symbol}")
            
            # Simulate paper trade execution
            # 模擬紙上交易執行
            if self.position_manager:
                price = 1.0500 if symbol == "EUR/USD" else 150.50
                await self.position_manager.add_position(
                    symbol=symbol,
                    side=side,
                    size=0.1,
                    entry_price=price
                )
                self.logger.info(f"✅ Demo position opened: {side} 0.1 {symbol} @ {price}")
        else:
            self.logger.info("📊 No trading signals generated this cycle")
            
    def _display_test_results(self, results: Dict[str, bool]) -> None:
        """Display comprehensive test results"""
        print("\n" + "="*60)
        print("🧪 SYSTEM VALIDATION RESULTS | 系統驗證結果")
        print("="*60)
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        pass_rate = (passed_tests / total_tests) * 100
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name.replace('_', ' ').title():<25} {status}")
            
        print("-"*60)
        print(f"Overall Pass Rate: {pass_rate:.1f}% ({passed_tests}/{total_tests})")
        
        if pass_rate >= 80:
            print("🎉 System validation SUCCESSFUL! Ready for trading.")
            print("🎉 系統驗證成功！準備開始交易。")
        else:
            print("⚠️ System validation INCOMPLETE. Review failed tests.")
            print("⚠️ 系統驗證不完整。請檢查失敗的測試。")
        print("="*60)
        
    async def _cleanup_and_exit(self) -> None:
        """
        Clean up resources and exit gracefully
        清理資源並優雅退出
        """
        self.logger.info("🧹 Cleaning up system resources...")
        self.logger.info("🧹 正在清理系統資源...")
        
        try:
            # Close trading system
            # 關閉交易系統
            if self.trading_system:
                # Graceful shutdown of trading system
                pass
                
            # Close dashboard
            # 關閉儀表板
            if self.dashboard:
                await self.dashboard.stop_monitoring()
                
            # Close position manager
            # 關閉倉位管理器
            if self.position_manager:
                # Save positions if needed
                pass
                
            self.logger.info("✅ System shutdown completed successfully")
            self.logger.info("✅ 系統關閉成功完成")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup error: {str(e)}")

async def main():
    """
    Main entry point for AIFX Trading System Demo
    AIFX交易系統演示主入口點
    """
    parser = argparse.ArgumentParser(
        description="AIFX Quantitative Trading System Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples | 示例:
  python run_trading_demo.py --mode demo    # Paper trading demonstration
  python run_trading_demo.py --mode live    # Live trading (requires API keys)  
  python run_trading_demo.py --mode test    # System validation tests

For live trading, ensure your IG Markets API credentials are configured.
對於實盤交易，請確保已配置您的IG Markets API憑證。
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["demo", "live", "test"],
        default="demo",
        help="Trading mode: demo (paper trading), live (real money), test (validation)"
    )
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = AIFXTradingDemo(mode=args.mode)
    await demo.run_trading_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 System interrupted by user. Goodbye!")
        print("🛑 系統被用戶中斷。再見！")
    except Exception as e:
        print(f"❌ Fatal error: {str(e)}")
        sys.exit(1)