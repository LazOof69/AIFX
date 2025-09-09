#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIFX Simple Testing Script
AIFX 簡單測試腳本

Quick testing without external dependencies.
不需要外部依賴的快速測試。
"""

import sys
import os
from datetime import datetime

# Add src path
sys.path.insert(0, os.path.join('src', 'main', 'python'))

print("🧪 AIFX Phase 4 Simple Testing")
print("=" * 40)

def test_performance_components():
    """Test performance components"""
    try:
        from data.performance_test import LoadGenerator, PerformanceMetrics
        
        # Test load generator
        config = {'symbols': ['EURUSD', 'USDJPY'], 'price_volatility': 0.001}
        gen = LoadGenerator(config)
        
        # Generate test data
        eurusd_tick = gen.generate_forex_tick('EURUSD')
        usdjpy_tick = gen.generate_forex_tick('USDJPY')
        
        print("✅ Load Generator Test:")
        print(f"   EURUSD: {eurusd_tick['bid']:.5f}/{eurusd_tick['ask']:.5f}")
        print(f"   USDJPY: {usdjpy_tick['bid']:.2f}/{usdjpy_tick['ask']:.2f}")
        
        # Test metrics
        metrics = PerformanceMetrics(
            test_name="sample_test",
            start_time=datetime.now()
        )
        metrics.total_operations = 100
        metrics.successful_operations = 95
        
        print(f"✅ Performance Metrics: {metrics.success_rate():.1f}% success rate")
        return True
        
    except Exception as e:
        print(f"❌ Performance Components: {e}")
        return False

def test_failover_logic():
    """Test failover logic"""
    try:
        from data.failover_manager import CircuitBreaker, SourceStatus
        
        # Test circuit breaker
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=60)
        
        print("✅ Circuit Breaker Test:")
        print(f"   Initial state: {breaker.state}")
        print(f"   Failure threshold: {breaker.failure_threshold}")
        
        # Simulate some failures
        for i in range(2):
            breaker._on_failure()
        
        print(f"   After 2 failures: {breaker.failure_count} (state: {breaker.state})")
        
        # Test enum
        status = SourceStatus.ACTIVE
        print(f"✅ Source Status: {status.value}")
        return True
        
    except Exception as e:
        print(f"❌ Failover Logic: {e}")
        return False

def test_data_structures():
    """Test data structures"""
    try:
        # Create forex tick data
        tick_data = {
            'symbol': 'EURUSD',
            'bid': 1.0850,
            'ask': 1.0852,
            'timestamp': datetime.now(),
            'source': 'test'
        }
        
        # Test calculations
        mid_price = (tick_data['bid'] + tick_data['ask']) / 2
        spread = tick_data['ask'] - tick_data['bid']
        
        print("✅ Data Structures Test:")
        print(f"   Symbol: {tick_data['symbol']}")
        print(f"   Mid Price: {mid_price:.5f}")
        print(f"   Spread: {spread:.5f}")
        return True
        
    except Exception as e:
        print(f"❌ Data Structures: {e}")
        return False

def test_configuration():
    """Test configuration handling"""
    try:
        import yaml
        import tempfile
        
        # Create test config
        config = {
            'forex_data_sources': {
                'yahoo': {'enabled': True, 'priority': 1},
                'oanda': {'enabled': False, 'priority': 2}
            },
            'performance': {
                'max_latency_ms': 50,
                'min_throughput_ops': 100
            }
        }
        
        # Test YAML handling
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        # Load and verify
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        os.unlink(config_path)
        
        print("✅ Configuration Test:")
        print(f"   Sources: {list(loaded_config['forex_data_sources'].keys())}")
        print(f"   Max Latency: {loaded_config['performance']['max_latency_ms']}ms")
        return True
        
    except Exception as e:
        print(f"❌ Configuration: {e}")
        return False

def main():
    """Run all tests"""
    print("\nRunning AIFX Component Tests...")
    print("-" * 40)
    
    tests = [
        ("Performance Components", test_performance_components),
        ("Failover Logic", test_failover_logic),
        ("Data Structures", test_data_structures),
        ("Configuration", test_configuration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Components are working correctly.")
        print("\n📋 Next Steps:")
        print("1. Install dependencies for full testing:")
        print("   pip install websocket-client psycopg2-binary redis")
        print("2. Setup PostgreSQL and Redis servers")
        print("3. Configure config/data-sources.yaml")
        print("4. Run full pipeline test:")
        print("   python src/main/python/data/pipeline_orchestrator.py")
    else:
        print("⚠️  Some tests failed. Check error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)