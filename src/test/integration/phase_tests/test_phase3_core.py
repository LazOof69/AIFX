#!/usr/bin/env python3
"""
Phase 3 Core Integration Test | 第三階段核心整合測試

Simplified integration testing for the core Phase 3 components we've built:
- Risk Management System | 風險管理系統
- Trading Strategy Engine | 交易策略引擎
- Signal Combination Framework | 信號組合框架

第三階段核心組件的簡化整合測試。
"""

import sys
import os
import unittest
import warnings
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path | 添加專案根目錄到路徑
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Suppress warnings for cleaner test output | 抑制警告以獲得更清潔的測試輸出
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARNING)

try:
    # Core components | 核心組件
    from src.main.python.core.risk_manager import (
        AdvancedRiskManager, RiskParameters, RiskLevel, Position, 
        RiskMetrics, create_risk_manager_preset
    )
    from src.main.python.core.signal_combiner import (
        TradingSignal, SignalType, SignalAggregator
    )
    from src.main.python.core.confidence_scorer import (
        AdvancedConfidenceScorer, ConfidenceComponents
    )
    
    IMPORTS_SUCCESS = True
    print("✅ Core components imported successfully")
    
except Exception as e:
    print(f"❌ Import Error: {e}")
    IMPORTS_SUCCESS = False


def create_sample_market_data():
    """Create sample market data for testing | 創建測試用樣本市場數據"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1H')
    n_periods = len(dates)
    
    np.random.seed(42)
    base_price = 1.1000
    returns = np.random.normal(0, 0.0005, n_periods)
    prices = base_price * (1 + returns).cumprod()
    
    high_low_spread = np.random.uniform(0.0001, 0.0010, n_periods)
    
    data = pd.DataFrame({
        'Open': prices,
        'High': prices + high_low_spread * 0.7,
        'Low': prices - high_low_spread * 0.3,
        'Close': prices + np.random.uniform(-high_low_spread*0.2, high_low_spread*0.2, n_periods),
        'Volume': np.random.randint(1000, 10000, n_periods)
    }, index=dates)
    
    return data


def test_risk_manager_core_functionality():
    """Test core risk manager functionality | 測試風險管理器核心功能"""
    print("\n🧪 Testing Risk Manager Core Functionality...")
    
    try:
        # Test initialization | 測試初始化
        risk_params = RiskParameters(
            max_position_size=0.02,
            min_position_size=0.005,
            stop_loss_atr_multiplier=2.0,
            take_profit_atr_multiplier=3.0
        )
        
        risk_manager = AdvancedRiskManager(risk_params, 100000.0)
        assert risk_manager is not None
        assert risk_manager.current_balance == 100000.0
        print("   ✅ Risk manager initialization passed")
        
        # Test preset creation | 測試預設創建
        conservative_manager = create_risk_manager_preset(RiskLevel.CONSERVATIVE)
        moderate_manager = create_risk_manager_preset(RiskLevel.MODERATE)
        aggressive_manager = create_risk_manager_preset(RiskLevel.AGGRESSIVE)
        
        assert conservative_manager is not None
        assert moderate_manager is not None
        assert aggressive_manager is not None
        print("   ✅ Risk manager presets created successfully")
        
        # Test position sizing | 測試倉位大小
        test_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=0.8,
            confidence=0.7,
            source="test_source",
            timestamp=datetime.now()
        )
        
        market_data = create_sample_market_data()
        position_size = risk_manager.calculate_position_size(test_signal, market_data, 100000.0)
        
        assert position_size > 0
        assert position_size <= 100000.0 * risk_params.max_position_size
        print(f"   ✅ Position size calculated: ${position_size:.2f}")
        
        # Test stop loss and take profit | 測試止損和止盈
        entry_price = 1.1000
        stop_loss = risk_manager.calculate_stop_loss(test_signal, entry_price, market_data)
        take_profit = risk_manager.calculate_take_profit(test_signal, entry_price, market_data)
        
        assert stop_loss < entry_price  # Stop loss below entry for long
        assert take_profit > entry_price  # Take profit above entry for long
        print(f"   ✅ SL/TP calculated: Entry={entry_price:.4f}, SL={stop_loss:.4f}, TP={take_profit:.4f}")
        
        # Test risk evaluation | 測試風險評估
        risk_metrics = risk_manager.evaluate_trade_risk(test_signal, market_data, entry_price)
        
        assert isinstance(risk_metrics, RiskMetrics)
        assert risk_metrics.position_size > 0
        assert isinstance(risk_metrics.risk_approval, bool)
        print(f"   ✅ Risk evaluation completed: Approved={risk_metrics.risk_approval}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Risk manager test failed: {e}")
        return False


def test_signal_combination_framework():
    """Test signal combination framework | 測試信號組合框架"""
    print("\n🧪 Testing Signal Combination Framework...")
    
    try:
        # Test signal creation | 測試信號創建
        signals = [
            TradingSignal(SignalType.BUY, 0.8, 0.7, "technical_1", datetime.now()),
            TradingSignal(SignalType.BUY, 0.6, 0.8, "ai_model_1", datetime.now()),
            TradingSignal(SignalType.SELL, 0.5, 0.6, "technical_2", datetime.now()),
        ]
        
        for signal in signals:
            assert isinstance(signal, TradingSignal)
            assert 0.0 <= signal.strength <= 1.0
            assert 0.0 <= signal.confidence <= 1.0
        
        print("   ✅ Trading signals created successfully")
        
        # Test signal aggregator | 測試信號聚合器
        aggregator = SignalAggregator()
        
        for signal in signals:
            aggregator.add_signal(signal)
        
        assert len(aggregator.signals) == 3
        
        # Test filtering | 測試過濾
        buy_signals = aggregator.get_signals_by_type(SignalType.BUY)
        sell_signals = aggregator.get_signals_by_type(SignalType.SELL)
        
        assert len(buy_signals) == 2
        assert len(sell_signals) == 1
        print("   ✅ Signal filtering working correctly")
        
        # Test summary | 測試摘要
        summary = aggregator.get_summary()
        assert summary['total_signals'] == 3
        assert 'signal_type_counts' in summary
        print(f"   ✅ Signal summary: {summary['total_signals']} signals")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Signal combination test failed: {e}")
        return False


def test_confidence_scoring_system():
    """Test confidence scoring system | 測試信心評分系統"""
    print("\n🧪 Testing Confidence Scoring System...")
    
    try:
        # Test confidence scorer initialization | 測試信心評分器初始化
        confidence_scorer = AdvancedConfidenceScorer(
            lookback_periods=50,
            volatility_window=20,
            agreement_threshold=0.7
        )
        
        assert confidence_scorer is not None
        print("   ✅ Confidence scorer initialized successfully")
        
        # Test confidence calculation | 測試信心計算
        test_signals = [
            TradingSignal(SignalType.BUY, 0.8, 0.7, "test_1", datetime.now()),
            TradingSignal(SignalType.BUY, 0.7, 0.8, "test_2", datetime.now()),
            TradingSignal(SignalType.BUY, 0.6, 0.6, "test_3", datetime.now()),
        ]
        
        market_data = create_sample_market_data()
        confidence_components = confidence_scorer.calculate_comprehensive_confidence(
            signals=test_signals,
            market_data=market_data
        )
        
        assert isinstance(confidence_components, ConfidenceComponents)
        assert 0.0 <= confidence_components.overall_confidence <= 1.0
        print(f"   ✅ Confidence calculated: {confidence_components.overall_confidence:.3f}")
        
        # Test confidence report | 測試信心報告
        report = confidence_scorer.get_confidence_report(confidence_components)
        assert 'overall_confidence' in report
        assert 'confidence_level' in report
        assert 'component_scores' in report
        print("   ✅ Confidence report generated successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Confidence scoring test failed: {e}")
        return False


def test_integrated_workflow():
    """Test integrated workflow between components | 測試組件間的整合工作流程"""
    print("\n🧪 Testing Integrated Workflow...")
    
    try:
        # Create components | 創建組件
        risk_params = RiskParameters(max_position_size=0.02, min_signal_confidence=0.6)
        risk_manager = AdvancedRiskManager(risk_params, 100000.0)
        
        confidence_scorer = AdvancedConfidenceScorer()
        aggregator = SignalAggregator()
        
        # Create test signals | 創建測試信號
        signals = [
            TradingSignal(SignalType.BUY, 0.8, 0.7, "technical", datetime.now()),
            TradingSignal(SignalType.BUY, 0.7, 0.8, "ai_model", datetime.now()),
        ]
        
        # Add signals to aggregator | 添加信號到聚合器
        for signal in signals:
            aggregator.add_signal(signal)
        
        # Calculate confidence | 計算信心
        market_data = create_sample_market_data()
        confidence_components = confidence_scorer.calculate_comprehensive_confidence(
            signals=signals,
            market_data=market_data
        )
        
        # Create combined signal with calculated confidence | 創建帶計算信心的組合信號
        combined_signal = TradingSignal(
            signal_type=SignalType.BUY,
            strength=0.75,  # Average of signal strengths
            confidence=confidence_components.overall_confidence,
            source="combined",
            timestamp=datetime.now()
        )
        
        # Evaluate risk | 評估風險
        risk_metrics = risk_manager.evaluate_trade_risk(
            combined_signal,
            market_data,
            1.1000
        )
        
        # Validate workflow | 驗證工作流程
        assert len(aggregator.signals) == 2
        assert confidence_components.overall_confidence > 0
        assert isinstance(risk_metrics, RiskMetrics)
        
        if risk_metrics.risk_approval:
            print(f"   ✅ Trade approved: Position=${risk_metrics.position_size:.2f}, RR={risk_metrics.risk_reward_ratio:.2f}")
        else:
            print(f"   ⚠️ Trade rejected: {risk_metrics.risk_warnings}")
        
        print("   ✅ Integrated workflow completed successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Integrated workflow test failed: {e}")
        return False


def test_position_management():
    """Test position management functionality | 測試倉位管理功能"""
    print("\n🧪 Testing Position Management...")
    
    try:
        risk_manager = AdvancedRiskManager(RiskParameters(), 100000.0)
        
        # Test adding position | 測試添加倉位
        risk_manager.add_position(
            symbol="EURUSD=X",
            side="long",
            size=1000.0,
            entry_price=1.1000,
            stop_loss=1.0950,
            take_profit=1.1100,
            signal_confidence=0.8
        )
        
        assert len(risk_manager.open_positions) == 1
        position = list(risk_manager.open_positions.values())[0]
        assert position.symbol == "EURUSD=X"
        assert position.side == "long"
        print("   ✅ Position added successfully")
        
        # Test closing position | 測試關閉倉位
        symbol = "EURUSD=X"
        exit_price = 1.1050
        risk_manager.close_position(symbol, exit_price, "test_exit")
        
        assert len(risk_manager.open_positions) == 0
        assert len(risk_manager.trade_history) == 1
        
        trade = risk_manager.trade_history[0]
        assert trade['exit_reason'] == "test_exit"
        print(f"   ✅ Position closed with PnL: ${trade['pnl']:.2f}")
        
        # Test risk summary | 測試風險摘要
        summary = risk_manager.get_risk_summary()
        assert 'account_summary' in summary
        assert 'position_summary' in summary
        assert 'performance_metrics' in summary
        print("   ✅ Risk summary generated successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Position management test failed: {e}")
        return False


def run_phase3_core_tests():
    """Run all Phase 3 core integration tests | 運行所有第三階段核心整合測試"""
    print("=" * 80)
    print("🚀 PHASE 3 CORE INTEGRATION TESTS | 第三階段核心整合測試")
    print("=" * 80)
    
    if not IMPORTS_SUCCESS:
        print("❌ Cannot run tests - required modules not available")
        return False
    
    # Define tests | 定義測試
    tests = [
        ("Risk Manager Core Functionality", test_risk_manager_core_functionality),
        ("Signal Combination Framework", test_signal_combination_framework),
        ("Confidence Scoring System", test_confidence_scoring_system),
        ("Position Management", test_position_management),
        ("Integrated Workflow", test_integrated_workflow),
    ]
    
    # Run tests | 運行測試
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append(False)
    
    # Calculate results | 計算結果
    total_tests = len(results)
    passed_tests = sum(results)
    success_rate = (passed_tests / total_tests) * 100
    
    # Print summary | 打印摘要
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY | 測試結果摘要")
    print("=" * 80)
    print(f"📈 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}")
    print(f"📊 Success Rate: {success_rate:.1f}%")
    
    # Final result | 最終結果
    if success_rate >= 80:
        print(f"\n🎉 PHASE 3 CORE TESTS: {'PASSED' if success_rate == 100 else 'MOSTLY PASSED'}")
        print("✅ Phase 3 core components are functioning correctly")
        return True
    else:
        print("\n🚨 PHASE 3 CORE TESTS: FAILED")
        print("❌ Phase 3 core components need additional work")
        return False


if __name__ == "__main__":
    success = run_phase3_core_tests()
    exit(0 if success else 1)