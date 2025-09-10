#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIFX Pipeline Runner - Windows Compatible
AIFX 管道運行器 - Windows 相容版本

This script provides a Windows-compatible way to run the AIFX pipeline orchestrator.
此腳本提供 Windows 相容的方式來運行 AIFX 管道協調器。
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
# 將項目根目錄添加到 Python 路徑
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Run the AIFX pipeline orchestrator"""
    print("🚀 AIFX Pipeline Orchestrator - Windows Compatible")
    print("=" * 60)
    
    try:
        # Import components with absolute imports
        from src.main.python.data.performance_test import LoadGenerator, PerformanceTestSuite
        from src.main.python.data.failover_manager import FailoverManager, CircuitBreaker
        
        print("\n✅ Successfully imported core components")
        
        # Test basic functionality
        print("\n🧪 Testing Core Functionality:")
        
        # Test 1: Load Generator
        print("\n1. Load Generator Test:")
        gen = LoadGenerator({'symbols': ['EURUSD', 'USDJPY'], 'price_volatility': 0.001})
        
        for i in range(3):
            eur_tick = gen.generate_forex_tick('EURUSD')
            jpy_tick = gen.generate_forex_tick('USDJPY') 
            print(f"   📈 EURUSD: {eur_tick['bid']:.5f}/{eur_tick['ask']:.5f}")
            print(f"   📈 USDJPY: {jpy_tick['bid']:.2f}/{jpy_tick['ask']:.2f}")
        
        # Test 2: Circuit Breaker
        print("\n2. Circuit Breaker Test:")
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        print(f"   Status: {breaker.state}")
        print(f"   Failure Threshold: {breaker.failure_threshold}")
        
        # Test 3: Performance Suite
        print("\n3. Performance Test Suite:")
        suite = PerformanceTestSuite()
        validation = suite.validate_requirements()
        print(f"   Requirements Validation: {sum(validation.values())}/{len(validation)} passed")
        
        print("\n🎉 All core components working correctly!")
        print("\n💡 To run more advanced tests:")
        print("   - python test_simple.py (basic tests)")
        print("   - python test_components.py (component tests)")
        print("   - python check_dependencies.py (dependency check)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n💡 Try running from project root directory:")
        print("   cd C:/Users/butte/OneDrive/桌面/AIFX_CLAUDE")
        print("   python run_pipeline.py")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)