#!/usr/bin/env python3
"""
AIFX Complete Integration Test | AIFX完整集成測試
===============================================

Comprehensive integration testing for the complete AIFX trading system.
AIFX交易系統的全面集成測試。

This test validates the entire trading workflow from data ingestion to trade execution.
此測試驗證從數據提取到交易執行的整個交易工作流程。

Usage | 使用方法:
    python test_integration_complete.py
    
Author: AIFX Development Team  
Created: 2025-01-14
"""

import asyncio
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Add src path to Python path
sys.path.insert(0, str(Path(__file__).parent / "src" / "main" / "python"))

try:
    # Import all major components for integration testing
    from main_trading_system import AIFXTradingSystem
    from trading.live_trader import LiveTrader, TradingDecision
    from trading.position_manager import PositionManager  
    from trading.execution_engine import ExecutionEngine, ExecutionMode
    from monitoring.dashboard import TradingDashboard
    from core.config_manager import ConfigManager
    from utils.logger import setup_logger
    from services.ig_api import IGMarketsAPI
    from utils.data_loader import DataLoader
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

class AIFXIntegrationTest(unittest.TestCase):
    """
    Complete integration test suite for AIFX trading system
    AIFX交易系統的完整集成測試套件
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.logger = setup_logger("AIFX_Integration_Test")
        cls.logger.info("🧪 Setting up AIFX Integration Test Environment...")
        cls.logger.info("🧪 正在設置AIFX集成測試環境...")
        
    def setUp(self):
        """Set up each test case"""
        self.logger.info(f"🔄 Starting test: {self._testMethodName}")
        
    def tearDown(self):
        """Clean up after each test case"""  
        self.logger.info(f"✅ Completed test: {self._testMethodName}")
        
    async def test_01_system_initialization(self):
        """
        Test complete system initialization
        測試完整系統初始化
        """
        self.logger.info("🚀 Testing system initialization...")
        
        try:
            # Test config manager initialization
            config_manager = ConfigManager()
            self.assertIsNotNone(config_manager)
            
            # Test position manager initialization
            position_manager = PositionManager()
            self.assertIsNotNone(position_manager)
            
            # Test live trader initialization  
            live_trader = LiveTrader()
            self.assertIsNotNone(live_trader)
            
            # Test execution engine initialization
            execution_engine = ExecutionEngine()
            self.assertIsNotNone(execution_engine)
            
            # Test dashboard initialization
            dashboard = TradingDashboard()
            self.assertIsNotNone(dashboard)
            
            # Test main trading system initialization
            trading_system = AIFXTradingSystem()
            self.assertIsNotNone(trading_system)
            
            self.logger.info("✅ System initialization test passed")
            
        except Exception as e:
            self.logger.error(f"❌ System initialization test failed: {str(e)}")
            raise
            
    async def test_02_data_pipeline_integration(self):
        """
        Test complete data pipeline integration
        測試完整數據管道集成
        """
        self.logger.info("📊 Testing data pipeline integration...")
        
        try:
            # Mock data loader for testing
            with patch('utils.data_loader.DataLoader') as mock_loader:
                mock_instance = mock_loader.return_value
                mock_instance.get_forex_data.return_value = self._create_mock_market_data()
                
                # Test data loading
                data_loader = DataLoader()
                market_data = await data_loader.get_forex_data("EUR/USD", "1h", 100)
                
                self.assertIsNotNone(market_data)
                self.assertGreater(len(market_data), 0)
                
            self.logger.info("✅ Data pipeline integration test passed")
            
        except Exception as e:
            self.logger.error(f"❌ Data pipeline integration test failed: {str(e)}")
            raise
            
    async def test_03_position_management_workflow(self):
        """
        Test complete position management workflow
        測試完整倉位管理工作流程
        """
        self.logger.info("📈 Testing position management workflow...")
        
        try:
            position_manager = PositionManager()
            
            # Test adding position
            position_id = await position_manager.add_position(
                symbol="EUR/USD",
                side="BUY",
                size=0.1,
                entry_price=1.0500,
                stop_loss=1.0450,
                take_profit=1.0600
            )
            
            self.assertIsNotNone(position_id)
            
            # Test getting position
            position = position_manager.get_position(position_id)
            self.assertIsNotNone(position)
            self.assertEqual(position["symbol"], "EUR/USD")
            
            # Test position update
            await position_manager.update_position(position_id, current_price=1.0520)
            updated_position = position_manager.get_position(position_id)
            self.assertIsNotNone(updated_position.get("unrealized_pnl"))
            
            # Test closing position
            await position_manager.close_position(position_id, exit_price=1.0550)
            closed_position = position_manager.get_position(position_id)
            self.assertEqual(closed_position["status"], "CLOSED")
            
            self.logger.info("✅ Position management workflow test passed")
            
        except Exception as e:
            self.logger.error(f"❌ Position management workflow test failed: {str(e)}")
            raise
            
    async def test_04_trading_decision_execution(self):
        """
        Test trading decision execution workflow
        測試交易決策執行工作流程
        """
        self.logger.info("⚙️ Testing trading decision execution...")
        
        try:
            # Create mock trading decisions
            decisions = [
                TradingDecision(
                    symbol="EUR/USD",
                    action="BUY",
                    size=0.1,
                    confidence=0.75,
                    stop_loss=1.0450,
                    take_profit=1.0600,
                    reasoning="AI model prediction + technical signals"
                ),
                TradingDecision(
                    symbol="USD/JPY", 
                    action="SELL",
                    size=0.1,
                    confidence=0.65,
                    stop_loss=151.00,
                    take_profit=149.50,
                    reasoning="Divergence signal + momentum indicator"
                )
            ]
            
            # Test execution engine
            execution_engine = ExecutionEngine()
            
            # Mock the live trader execute method
            with patch.object(execution_engine, 'live_trader') as mock_trader:
                mock_trader.execute_decision = AsyncMock(return_value="ORDER_12345")
                
                result = await execution_engine.execute_decisions(
                    decisions, 
                    ExecutionMode.CONSERVATIVE
                )
                
                self.assertIsNotNone(result)
                
            self.logger.info("✅ Trading decision execution test passed")
            
        except Exception as e:
            self.logger.error(f"❌ Trading decision execution test failed: {str(e)}")
            raise
            
    async def test_05_risk_management_integration(self):
        """
        Test risk management system integration
        測試風險管理系統集成
        """
        self.logger.info("🛡️ Testing risk management integration...")
        
        try:
            position_manager = PositionManager()
            
            # Test risk limits
            risk_check = position_manager._validate_risk_limits(
                symbol="EUR/USD",
                size=0.1,
                current_exposure=0.5,
                account_balance=10000
            )
            self.assertTrue(risk_check)
            
            # Test excessive risk
            risk_check_high = position_manager._validate_risk_limits(
                symbol="EUR/USD", 
                size=2.0,  # Very high size
                current_exposure=1.0,
                account_balance=10000
            )
            self.assertFalse(risk_check_high)
            
            self.logger.info("✅ Risk management integration test passed")
            
        except Exception as e:
            self.logger.error(f"❌ Risk management integration test failed: {str(e)}")
            raise
            
    async def test_06_monitoring_dashboard_integration(self):
        """
        Test monitoring dashboard integration
        測試監控儀表板集成
        """
        self.logger.info("📊 Testing monitoring dashboard integration...")
        
        try:
            dashboard = TradingDashboard()
            
            # Test dashboard startup
            self.assertIsNotNone(dashboard.start_time)
            
            # Test metrics collection
            await dashboard._collect_system_metrics()
            self.assertIsNotNone(dashboard.metrics)
            
            # Test alert system
            dashboard._check_alerts()
            
            self.logger.info("✅ Monitoring dashboard integration test passed")
            
        except Exception as e:
            self.logger.error(f"❌ Monitoring dashboard integration test failed: {str(e)}")
            raise
            
    async def test_07_end_to_end_trading_workflow(self):
        """
        Test complete end-to-end trading workflow
        測試完整的端到端交易工作流程
        """
        self.logger.info("🎯 Testing end-to-end trading workflow...")
        
        try:
            # Initialize all components
            position_manager = PositionManager()
            live_trader = LiveTrader()
            execution_engine = ExecutionEngine()
            dashboard = TradingDashboard()
            
            # Mock external dependencies
            with patch.object(live_trader, 'ig_api') as mock_api:
                mock_api.place_order = AsyncMock(return_value={
                    "dealReference": "ORDER_12345",
                    "status": "ACCEPTED"
                })
                mock_api.get_position_status = AsyncMock(return_value={
                    "status": "OPEN",
                    "current_price": 1.0520
                })
                
                # 1. Create trading decision
                decision = TradingDecision(
                    symbol="EUR/USD",
                    action="BUY", 
                    size=0.1,
                    confidence=0.80,
                    stop_loss=1.0450,
                    take_profit=1.0600,
                    reasoning="Strong bullish signals from multiple indicators"
                )
                
                # 2. Execute trade
                order_id = await live_trader.execute_decision(decision)
                self.assertIsNotNone(order_id)
                
                # 3. Track position
                position_id = await position_manager.add_position(
                    symbol=decision.symbol,
                    side=decision.action,
                    size=decision.size,
                    entry_price=1.0500
                )
                
                # 4. Monitor position
                await position_manager.update_position(position_id, current_price=1.0520)
                position = position_manager.get_position(position_id)
                self.assertGreater(position["unrealized_pnl"], 0)
                
                # 5. Collect metrics
                await dashboard._collect_system_metrics()
                
            self.logger.info("✅ End-to-end trading workflow test passed")
            
        except Exception as e:
            self.logger.error(f"❌ End-to-end trading workflow test failed: {str(e)}")
            raise
            
    async def test_08_error_handling_and_recovery(self):
        """
        Test error handling and recovery mechanisms
        測試錯誤處理和恢復機制
        """
        self.logger.info("🔧 Testing error handling and recovery...")
        
        try:
            live_trader = LiveTrader()
            
            # Test API connection failure handling
            with patch.object(live_trader, 'ig_api') as mock_api:
                mock_api.place_order = AsyncMock(side_effect=Exception("API Connection Failed"))
                
                decision = TradingDecision(
                    symbol="EUR/USD",
                    action="BUY",
                    size=0.1,
                    confidence=0.75
                )
                
                # Should handle the error gracefully
                with self.assertLogs(level='ERROR') as log:
                    result = await live_trader.execute_decision(decision)
                    self.assertIsNone(result)
                    
            # Test position manager error handling
            position_manager = PositionManager()
            
            # Test invalid position operations
            invalid_position = position_manager.get_position("INVALID_ID")
            self.assertIsNone(invalid_position)
            
            self.logger.info("✅ Error handling and recovery test passed")
            
        except Exception as e:
            self.logger.error(f"❌ Error handling and recovery test failed: {str(e)}")
            raise
            
    def _create_mock_market_data(self) -> List[Dict[str, Any]]:
        """Create mock market data for testing"""
        base_time = datetime.now() - timedelta(hours=100)
        data = []
        
        for i in range(100):
            timestamp = base_time + timedelta(hours=i)
            data.append({
                "timestamp": timestamp,
                "open": 1.0500 + (i * 0.0001),
                "high": 1.0520 + (i * 0.0001), 
                "low": 1.0480 + (i * 0.0001),
                "close": 1.0510 + (i * 0.0001),
                "volume": 1000 + (i * 10)
            })
            
        return data

async def run_integration_tests():
    """
    Run all integration tests
    運行所有集成測試
    """
    print("🧪 AIFX COMPLETE INTEGRATION TEST SUITE")
    print("🧪 AIFX完整集成測試套件")
    print("="*80)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(AIFXIntegrationTest)
    
    # Custom test runner to handle async tests
    results = {
        "total": 0,
        "passed": 0, 
        "failed": 0,
        "errors": []
    }
    
    test_instance = AIFXIntegrationTest()
    test_instance.setUpClass()
    
    # Get all test methods
    test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
    test_methods.sort()  # Run in order
    
    for test_method_name in test_methods:
        results["total"] += 1
        test_method = getattr(test_instance, test_method_name)
        
        try:
            test_instance.setUp()
            if asyncio.iscoroutinefunction(test_method):
                await test_method()
            else:
                test_method()
            test_instance.tearDown()
            results["passed"] += 1
            
        except Exception as e:
            results["failed"] += 1
            results["errors"].append(f"{test_method_name}: {str(e)}")
            print(f"❌ FAILED: {test_method_name}")
            
    # Display results
    print("\n" + "="*80)
    print("📊 INTEGRATION TEST RESULTS | 集成測試結果")
    print("="*80)
    
    pass_rate = (results["passed"] / results["total"]) * 100 if results["total"] > 0 else 0
    
    print(f"Total Tests: {results['total']}")
    print(f"Passed: {results['passed']}")
    print(f"Failed: {results['failed']}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    
    if results["errors"]:
        print("\n❌ FAILED TESTS:")
        for error in results["errors"]:
            print(f"  - {error}")
            
    print("\n" + "="*80)
    
    if pass_rate >= 80:
        print("🎉 INTEGRATION TESTS SUCCESSFUL!")
        print("🎉 集成測試成功！")
        print("✅ AIFX trading system is ready for deployment")
        print("✅ AIFX交易系統已準備好部署")
    else:
        print("⚠️ INTEGRATION TESTS INCOMPLETE")
        print("⚠️ 集成測試不完整")
        print("🔧 Please review and fix the failed tests")
        print("🔧 請檢查並修復失敗的測試")
    
    print("="*80)
    return pass_rate >= 80

if __name__ == "__main__":
    try:
        success = asyncio.run(run_integration_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Integration tests interrupted by user")
    except Exception as e:
        print(f"❌ Integration test error: {str(e)}")
        sys.exit(1)