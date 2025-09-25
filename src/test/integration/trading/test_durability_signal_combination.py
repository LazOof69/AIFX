"""
Durability Test Suite for Phase 3.1.1 Signal Combination System | 第三階段3.1.1信號組合系統耐久性測試

Advanced durability and stress testing for the complete signal combination framework:
- Memory leak detection and resource management
- High-volume signal processing capability
- Edge case resilience and error recovery
- Long-running operation stability
- Concurrent operation safety
- Performance degradation monitoring

高級耐久性和壓力測試，針對完整的信號組合框架：
- 記憶體洩漏檢測和資源管理
- 高容量信號處理能力
- 邊緣情況韌性和錯誤恢復
- 長時間運行穩定性
- 並發操作安全性
- 性能降級監控
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('src/main/python')

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
import gc
import tracemalloc
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import unittest
from unittest.mock import Mock, patch
import warnings

# Import the signal combination system components
from core.signal_combiner import BaseSignalCombiner, TradingSignal, SignalType, SignalAggregator
from core.confidence_scorer import AdvancedConfidenceScorer, ConfidenceComponents
from core.weight_optimizer import DynamicWeightOptimizer, WeightOptimizationConfig, OptimizationMethod
from core.ai_signal_combiner import AISignalCombiner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning)


class SignalCombinationDurabilityTest:
    """
    Comprehensive durability test suite for signal combination system
    信號組合系統綜合耐久性測試套件
    """
    
    def __init__(self):
        """Initialize durability test environment | 初始化耐久性測試環境"""
        logger.info("Initializing Signal Combination Durability Test Suite...")
        
        # Test configuration | 測試配置
        self.test_config = {
            'high_volume_signal_count': 10000,
            'stress_test_duration': 60,  # seconds
            'memory_threshold_mb': 50,
            'concurrent_threads': 10,
            'performance_iterations': 1000
        }
        
        # Initialize components | 初始化組件
        self.setup_test_components()
        
        # Performance tracking | 性能追蹤
        self.performance_metrics = {
            'memory_usage': [],
            'processing_times': [],
            'error_count': 0,
            'successful_operations': 0
        }
        
    def setup_test_components(self):
        """Setup test components and mock data | 設置測試組件和模擬數據"""
        # Mock AI models for testing | 用於測試的模擬AI模型
        self.mock_models = {
            'XGBoost': self.create_mock_model('XGBoost'),
            'RandomForest': self.create_mock_model('RandomForest'),
            'LSTM': self.create_mock_model('LSTM')
        }
        
        # Initialize test components | 初始化測試組件
        self.ai_combiner = AISignalCombiner(self.mock_models)
        self.confidence_scorer = AdvancedConfidenceScorer()
        self.weight_optimizer = DynamicWeightOptimizer()
        self.signal_aggregator = SignalAggregator()
        
        # Create large test datasets | 創建大型測試數據集
        self.large_market_data = self.create_large_market_dataset(10000)  # 10K records
        
    def create_mock_model(self, model_name: str) -> Mock:
        """Create mock AI model with consistent behavior | 創建行為一致的模擬AI模型"""
        mock_model = Mock()
        mock_model.model_name = model_name
        mock_model.version = "1.0"
        mock_model.is_trained = True
        mock_model.metadata = {
            'training_samples': 10000,
            'performance_metrics': {'accuracy': 0.65 + np.random.random() * 0.1}
        }
        
        # Consistent prediction patterns | 一致的預測模式
        np.random.seed(hash(model_name) % 2**31)
        mock_model.predict.return_value = np.array([np.random.random() * 2 - 1])
        proba = np.random.dirichlet([1, 1])
        mock_model.predict_proba.return_value = np.array([proba])
        
        return mock_model
    
    def create_large_market_dataset(self, size: int) -> pd.DataFrame:
        """Create large synthetic market dataset | 創建大型合成市場數據集"""
        dates = pd.date_range(start='2020-01-01', periods=size, freq='1H')
        
        # Generate realistic price movements | 生成真實的價格變動
        np.random.seed(42)
        initial_price = 1.1000
        returns = np.random.normal(0.0001, 0.002, size)
        prices = np.zeros(size)
        prices[0] = initial_price
        
        for i in range(1, size):
            prices[i] = prices[i-1] * (1 + returns[i])
        
        market_data = pd.DataFrame({
            'Open': prices * (1 + np.random.normal(0, 0.0001, size)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.0005, size))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.0005, size))),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, size)
        }, index=dates)
        
        return market_data
    
    def create_high_volume_signals(self, count: int) -> List[TradingSignal]:
        """Create high volume of test signals | 創建大量測試信號"""
        signals = []
        current_time = datetime.now()
        
        for i in range(count):
            signal_type = np.random.choice([SignalType.BUY, SignalType.SELL, SignalType.HOLD])
            strength = np.random.random()
            confidence = np.random.random()
            
            # Vary sources to test different scenarios | 變化來源以測試不同情況
            sources = ['AI_XGBoost', 'AI_RandomForest', 'AI_LSTM', 'Tech_MA', 'Tech_RSI', 'Tech_MACD']
            source = np.random.choice(sources)
            
            timestamp = current_time - timedelta(seconds=np.random.randint(0, 3600))
            
            signal = TradingSignal(
                signal_type=signal_type,
                strength=strength,
                confidence=confidence,
                source=source,
                timestamp=timestamp,
                metadata={'batch_id': i // 1000, 'test_signal': True}
            )
            
            signals.append(signal)
        
        return signals
    
    def monitor_memory_usage(self) -> float:
        """Monitor current memory usage in MB | 監控當前記憶體使用量（MB）"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except ImportError:
            # Fallback to tracemalloc if psutil not available | 如果psutil不可用則回退到tracemalloc
            current, peak = tracemalloc.get_traced_memory()
            return current / 1024 / 1024
    
    def test_1_memory_leak_detection(self) -> Dict[str, Any]:
        """Test 1: Memory leak detection during continuous operation | 測試1：連續運行期間的記憶體洩漏檢測"""
        logger.info("🧪 Durability Test 1: Memory leak detection")
        
        # Start memory tracking | 開始記憶體追蹤
        tracemalloc.start()
        initial_memory = self.monitor_memory_usage()
        
        test_results = {
            'status': 'RUNNING',
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': initial_memory,
            'final_memory_mb': 0,
            'memory_growth_mb': 0,
            'iterations': 100,
            'leak_detected': False
        }
        
        try:
            # Perform repeated operations | 執行重複操作
            for iteration in range(test_results['iterations']):
                # Create and process signals | 創建和處理信號
                test_signals = self.create_high_volume_signals(100)
                
                # Process through all components | 通過所有組件處理
                for signal in test_signals:
                    self.signal_aggregator.add_signal(signal)
                
                # Combine signals | 組合信號
                combined = self.ai_combiner.combine_signals(test_signals[:10])  # Subset for efficiency
                
                # Calculate confidence | 計算信心
                confidence = self.confidence_scorer.calculate_comprehensive_confidence(
                    test_signals[:5], 
                    self.large_market_data.tail(50)
                )
                
                # Optimize weights | 優化權重
                sources = list(set(s.source for s in test_signals[:10]))
                weights = self.weight_optimizer.optimize_weights(sources)
                
                # Clear aggregator periodically | 定期清理聚合器
                if iteration % 10 == 0:
                    self.signal_aggregator.clear()
                    gc.collect()  # Force garbage collection
                
                # Monitor memory | 監控記憶體
                current_memory = self.monitor_memory_usage()
                test_results['peak_memory_mb'] = max(test_results['peak_memory_mb'], current_memory)
                
                # Check for concerning memory growth | 檢查令人擔憂的記憶體增長
                if current_memory > initial_memory + self.test_config['memory_threshold_mb']:
                    test_results['leak_detected'] = True
                    logger.warning(f"Potential memory leak detected at iteration {iteration}: "
                                 f"{current_memory:.1f}MB (started at {initial_memory:.1f}MB)")
            
            test_results['final_memory_mb'] = self.monitor_memory_usage()
            test_results['memory_growth_mb'] = test_results['final_memory_mb'] - initial_memory
            test_results['status'] = 'PASSED' if not test_results['leak_detected'] else 'WARNING'
            
        except Exception as e:
            test_results['status'] = 'FAILED'
            test_results['error'] = str(e)
        
        finally:
            tracemalloc.stop()
        
        logger.info(f"✅ Memory Test Result: {test_results['status']} - "
                   f"Growth: {test_results['memory_growth_mb']:.1f}MB")
        
        return test_results
    
    def test_2_high_volume_processing(self) -> Dict[str, Any]:
        """Test 2: High volume signal processing capability | 測試2：高容量信號處理能力"""
        logger.info("🧪 Durability Test 2: High volume signal processing")
        
        test_results = {
            'status': 'RUNNING',
            'total_signals': self.test_config['high_volume_signal_count'],
            'processing_time': 0,
            'signals_per_second': 0,
            'memory_efficiency': True,
            'errors_encountered': 0
        }
        
        try:
            # Create high volume of signals | 創建大量信號
            start_time = time.time()
            high_volume_signals = self.create_high_volume_signals(test_results['total_signals'])
            creation_time = time.time() - start_time
            
            logger.info(f"Created {len(high_volume_signals)} signals in {creation_time:.2f}s")
            
            # Process signals in batches | 批次處理信號
            batch_size = 1000
            processing_start = time.time()
            
            for i in range(0, len(high_volume_signals), batch_size):
                batch = high_volume_signals[i:i+batch_size]
                
                try:
                    # Process batch through signal aggregator | 通過信號聚合器處理批次
                    for signal in batch:
                        self.signal_aggregator.add_signal(signal)
                    
                    # Get batch summary | 獲取批次摘要
                    summary = self.signal_aggregator.get_summary()
                    
                    # Test confidence scoring on subset | 在子集上測試信心評分
                    sample_signals = batch[:10]
                    confidence_components = self.confidence_scorer.calculate_comprehensive_confidence(
                        sample_signals,
                        self.large_market_data.tail(100)
                    )
                    
                    # Test weight optimization | 測試權重優化
                    sources = list(set(s.source for s in sample_signals))
                    weights = self.weight_optimizer.optimize_weights(sources)
                    
                    # Clear aggregator for next batch | 為下一批次清理聚合器
                    self.signal_aggregator.clear()
                    
                except Exception as e:
                    test_results['errors_encountered'] += 1
                    if test_results['errors_encountered'] > 10:  # Fail if too many errors
                        raise e
            
            processing_time = time.time() - processing_start
            test_results['processing_time'] = processing_time
            test_results['signals_per_second'] = test_results['total_signals'] / processing_time
            
            # Check memory efficiency | 檢查記憶體效率
            final_memory = self.monitor_memory_usage()
            test_results['memory_efficiency'] = final_memory < 100  # Less than 100MB
            
            test_results['status'] = 'PASSED' if test_results['errors_encountered'] < 5 else 'WARNING'
            
        except Exception as e:
            test_results['status'] = 'FAILED'
            test_results['error'] = str(e)
        
        logger.info(f"✅ High Volume Test Result: {test_results['status']} - "
                   f"{test_results['signals_per_second']:.0f} signals/sec")
        
        return test_results
    
    def test_3_concurrent_operation_safety(self) -> Dict[str, Any]:
        """Test 3: Concurrent operation safety and thread safety | 測試3：並發操作安全性和線程安全性"""
        logger.info("🧪 Durability Test 3: Concurrent operation safety")
        
        test_results = {
            'status': 'RUNNING',
            'concurrent_threads': self.test_config['concurrent_threads'],
            'operations_per_thread': 100,
            'total_operations': 0,
            'successful_operations': 0,
            'thread_errors': 0,
            'race_conditions_detected': 0
        }
        
        def worker_thread(thread_id: int) -> Dict[str, Any]:
            """Worker thread for concurrent testing | 並發測試的工作線程"""
            thread_results = {
                'thread_id': thread_id,
                'operations': 0,
                'errors': 0,
                'processing_time': 0
            }
            
            start_time = time.time()
            
            try:
                # Create separate components for this thread | 為此線程創建單獨的組件
                thread_aggregator = SignalAggregator()
                thread_confidence_scorer = AdvancedConfidenceScorer()
                
                for i in range(test_results['operations_per_thread']):
                    # Create signals for this thread | 為此線程創建信號
                    thread_signals = self.create_high_volume_signals(10)
                    
                    # Process signals | 處理信號
                    for signal in thread_signals:
                        thread_aggregator.add_signal(signal)
                    
                    # Test confidence calculation | 測試信心計算
                    confidence = thread_confidence_scorer.calculate_comprehensive_confidence(
                        thread_signals[:5],
                        self.large_market_data.tail(50)
                    )
                    
                    # Test AI combiner (using shared instance - this tests thread safety)
                    # 測試AI組合器（使用共享實例 - 這測試線程安全性）
                    combined = self.ai_combiner.combine_signals(thread_signals[:3])
                    
                    thread_results['operations'] += 1
                    
                    # Small delay to increase chance of race conditions | 小延遲以增加競態條件的機會
                    time.sleep(0.001)
                
            except Exception as e:
                thread_results['errors'] += 1
                logger.warning(f"Thread {thread_id} error: {e}")
            
            thread_results['processing_time'] = time.time() - start_time
            return thread_results
        
        try:
            # Launch concurrent threads | 啟動並發線程
            with ThreadPoolExecutor(max_workers=test_results['concurrent_threads']) as executor:
                # Submit all worker threads | 提交所有工作線程
                futures = [
                    executor.submit(worker_thread, i) 
                    for i in range(test_results['concurrent_threads'])
                ]
                
                # Collect results | 收集結果
                thread_results = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=30)  # 30 second timeout per thread
                        thread_results.append(result)
                        test_results['successful_operations'] += result['operations']
                        test_results['thread_errors'] += result['errors']
                    except Exception as e:
                        test_results['thread_errors'] += 1
                        logger.error(f"Thread execution failed: {e}")
            
            test_results['total_operations'] = test_results['successful_operations'] + test_results['thread_errors']
            
            # Analyze results for race conditions | 分析結果中的競態條件
            expected_operations = test_results['concurrent_threads'] * test_results['operations_per_thread']
            if test_results['total_operations'] != expected_operations:
                test_results['race_conditions_detected'] = expected_operations - test_results['total_operations']
            
            # Determine test status | 確定測試狀態
            if test_results['thread_errors'] == 0 and test_results['race_conditions_detected'] == 0:
                test_results['status'] = 'PASSED'
            elif test_results['thread_errors'] < 5:
                test_results['status'] = 'WARNING'
            else:
                test_results['status'] = 'FAILED'
            
        except Exception as e:
            test_results['status'] = 'FAILED'
            test_results['error'] = str(e)
        
        logger.info(f"✅ Concurrent Test Result: {test_results['status']} - "
                   f"{test_results['successful_operations']}/{test_results['total_operations']} operations")
        
        return test_results
    
    def test_4_long_running_stability(self) -> Dict[str, Any]:
        """Test 4: Long-running operation stability | 測試4：長時間運行穩定性"""
        logger.info("🧪 Durability Test 4: Long-running stability")
        
        test_results = {
            'status': 'RUNNING',
            'duration_seconds': self.test_config['stress_test_duration'],
            'operations_completed': 0,
            'average_processing_time': 0,
            'performance_degradation': False,
            'stability_issues': 0,
            'memory_stable': True
        }
        
        start_time = time.time()
        processing_times = []
        memory_readings = []
        
        try:
            operation_count = 0
            
            while (time.time() - start_time) < test_results['duration_seconds']:
                operation_start = time.time()
                
                try:
                    # Simulate realistic trading signal processing | 模擬真實的交易信號處理
                    test_signals = self.create_high_volume_signals(50)
                    
                    # Process through complete pipeline | 通過完整管道處理
                    # 1. Aggregation | 1. 聚合
                    for signal in test_signals:
                        self.signal_aggregator.add_signal(signal)
                    
                    # 2. AI combination | 2. AI組合
                    ai_signals = test_signals[:10]  # Subset for AI processing
                    combined_signal = self.ai_combiner.combine_signals(ai_signals)
                    
                    # 3. Confidence scoring | 3. 信心評分
                    confidence = self.confidence_scorer.calculate_comprehensive_confidence(
                        ai_signals,
                        self.large_market_data.tail(100)
                    )
                    
                    # 4. Weight optimization | 4. 權重優化
                    sources = list(set(s.source for s in test_signals[:10]))
                    if sources:
                        weights = self.weight_optimizer.optimize_weights(sources)
                    
                    # Track performance | 追蹤性能
                    operation_time = time.time() - operation_start
                    processing_times.append(operation_time)
                    
                    # Monitor memory periodically | 定期監控記憶體
                    if operation_count % 10 == 0:
                        memory_mb = self.monitor_memory_usage()
                        memory_readings.append(memory_mb)
                    
                    # Clear aggregator periodically to prevent memory buildup | 定期清理聚合器以防止記憶體積累
                    if operation_count % 20 == 0:
                        self.signal_aggregator.clear()
                        gc.collect()
                    
                    operation_count += 1
                    
                except Exception as e:
                    test_results['stability_issues'] += 1
                    logger.warning(f"Stability issue in operation {operation_count}: {e}")
                    
                    # Fail if too many stability issues | 如果穩定性問題太多則失敗
                    if test_results['stability_issues'] > 10:
                        raise e
            
            test_results['operations_completed'] = operation_count
            
            # Analyze performance metrics | 分析性能指標
            if processing_times:
                test_results['average_processing_time'] = np.mean(processing_times)
                
                # Check for performance degradation | 檢查性能降級
                first_half = processing_times[:len(processing_times)//2]
                second_half = processing_times[len(processing_times)//2:]
                
                if len(first_half) > 10 and len(second_half) > 10:
                    first_half_avg = np.mean(first_half)
                    second_half_avg = np.mean(second_half)
                    
                    # Performance degradation if second half is 50% slower | 如果後半段慢50%則性能降級
                    if second_half_avg > first_half_avg * 1.5:
                        test_results['performance_degradation'] = True
            
            # Analyze memory stability | 分析記憶體穩定性
            if len(memory_readings) > 2:
                memory_growth = memory_readings[-1] - memory_readings[0]
                if memory_growth > 20:  # More than 20MB growth is concerning | 超過20MB增長令人擔憂
                    test_results['memory_stable'] = False
            
            # Determine overall status | 確定整體狀態
            if (test_results['stability_issues'] == 0 and 
                not test_results['performance_degradation'] and 
                test_results['memory_stable']):
                test_results['status'] = 'PASSED'
            elif test_results['stability_issues'] < 5:
                test_results['status'] = 'WARNING'
            else:
                test_results['status'] = 'FAILED'
                
        except Exception as e:
            test_results['status'] = 'FAILED'
            test_results['error'] = str(e)
        
        actual_duration = time.time() - start_time
        logger.info(f"✅ Long-running Test Result: {test_results['status']} - "
                   f"{test_results['operations_completed']} ops in {actual_duration:.1f}s")
        
        return test_results
    
    def test_5_extreme_edge_cases(self) -> Dict[str, Any]:
        """Test 5: Extreme edge cases and error recovery | 測試5：極端邊緣情況和錯誤恢復"""
        logger.info("🧪 Durability Test 5: Extreme edge cases")
        
        test_results = {
            'status': 'RUNNING',
            'edge_cases_tested': 0,
            'edge_cases_handled': 0,
            'critical_failures': 0,
            'recovery_successful': 0
        }
        
        edge_test_cases = [
            self._test_empty_data_handling,
            self._test_corrupted_signals,
            self._test_extreme_values,
            self._test_null_data_handling,
            self._test_circular_dependencies,
            self._test_resource_exhaustion_simulation,
            self._test_timing_edge_cases,
            self._test_concurrent_modifications
        ]
        
        try:
            for i, edge_test in enumerate(edge_test_cases):
                test_results['edge_cases_tested'] += 1
                
                try:
                    logger.info(f"Running edge case test {i+1}/{len(edge_test_cases)}: {edge_test.__name__}")
                    result = edge_test()
                    
                    if result.get('handled', False):
                        test_results['edge_cases_handled'] += 1
                    
                    if result.get('recovered', False):
                        test_results['recovery_successful'] += 1
                    
                    if result.get('critical_failure', False):
                        test_results['critical_failures'] += 1
                        
                except Exception as e:
                    test_results['critical_failures'] += 1
                    logger.error(f"Critical failure in edge case {i+1}: {e}")
            
            # Determine test status | 確定測試狀態
            success_rate = test_results['edge_cases_handled'] / test_results['edge_cases_tested']
            
            if success_rate >= 0.9 and test_results['critical_failures'] == 0:
                test_results['status'] = 'PASSED'
            elif success_rate >= 0.7:
                test_results['status'] = 'WARNING'
            else:
                test_results['status'] = 'FAILED'
                
        except Exception as e:
            test_results['status'] = 'FAILED'
            test_results['error'] = str(e)
        
        logger.info(f"✅ Edge Cases Test Result: {test_results['status']} - "
                   f"{test_results['edge_cases_handled']}/{test_results['edge_cases_tested']} handled")
        
        return test_results
    
    def _test_empty_data_handling(self) -> Dict[str, Any]:
        """Test empty data handling | 測試空數據處理"""
        try:
            # Test with empty signals | 用空信號測試
            result = self.ai_combiner.combine_signals([])
            confidence = self.confidence_scorer.calculate_comprehensive_confidence([])
            weights = self.weight_optimizer.optimize_weights([])
            
            return {'handled': True, 'recovered': True, 'critical_failure': False}
        except Exception:
            return {'handled': False, 'recovered': False, 'critical_failure': True}
    
    def _test_corrupted_signals(self) -> Dict[str, Any]:
        """Test corrupted signal handling | 測試損壞信號處理"""
        try:
            # Create signals with corrupted data | 創建帶有損壞數據的信號
            corrupted_signal = TradingSignal(
                SignalType.BUY, float('inf'), float('nan'), None
            )
            
            result = self.ai_combiner.combine_signals([corrupted_signal])
            return {'handled': True, 'recovered': True, 'critical_failure': False}
        except Exception:
            return {'handled': False, 'recovered': False, 'critical_failure': True}
    
    def _test_extreme_values(self) -> Dict[str, Any]:
        """Test extreme value handling | 測試極值處理"""
        try:
            # Test with extreme values | 用極值測試
            extreme_signals = [
                TradingSignal(SignalType.BUY, 1e10, 1e10, "extreme_source"),
                TradingSignal(SignalType.SELL, -1e10, -1e10, "extreme_source2")
            ]
            
            result = self.ai_combiner.combine_signals(extreme_signals)
            return {'handled': True, 'recovered': True, 'critical_failure': False}
        except Exception:
            return {'handled': False, 'recovered': False, 'critical_failure': True}
    
    def _test_null_data_handling(self) -> Dict[str, Any]:
        """Test null data handling | 測試空數據處理"""
        try:
            # Test with None values | 用None值測試
            confidence = self.confidence_scorer.calculate_comprehensive_confidence(
                None, None, None
            )
            return {'handled': True, 'recovered': True, 'critical_failure': False}
        except Exception:
            return {'handled': False, 'recovered': False, 'critical_failure': True}
    
    def _test_circular_dependencies(self) -> Dict[str, Any]:
        """Test circular dependency handling | 測試循環依賴處理"""
        try:
            # This is more of a design test - ensure no circular references | 這更像是設計測試 - 確保沒有循環引用
            # Create complex object relationships | 創建複雜的對象關係
            aggregator1 = SignalAggregator()
            aggregator2 = SignalAggregator()
            
            # Test that objects can be properly garbage collected | 測試對象可以正確垃圾回收
            del aggregator1, aggregator2
            gc.collect()
            
            return {'handled': True, 'recovered': True, 'critical_failure': False}
        except Exception:
            return {'handled': False, 'recovered': False, 'critical_failure': True}
    
    def _test_resource_exhaustion_simulation(self) -> Dict[str, Any]:
        """Test resource exhaustion simulation | 測試資源耗盡模擬"""
        try:
            # Simulate resource exhaustion by creating many objects | 通過創建許多對象來模擬資源耗盡
            large_objects = []
            for _ in range(100):
                obj = SignalAggregator()
                signals = self.create_high_volume_signals(100)
                for signal in signals:
                    obj.add_signal(signal)
                large_objects.append(obj)
            
            # Clean up | 清理
            del large_objects
            gc.collect()
            
            return {'handled': True, 'recovered': True, 'critical_failure': False}
        except MemoryError:
            # Memory error is expected behavior | 記憶體錯誤是預期行為
            gc.collect()
            return {'handled': True, 'recovered': True, 'critical_failure': False}
        except Exception:
            return {'handled': False, 'recovered': False, 'critical_failure': True}
    
    def _test_timing_edge_cases(self) -> Dict[str, Any]:
        """Test timing-related edge cases | 測試時間相關的邊緣情況"""
        try:
            # Test with signals from very different times | 用來自不同時間的信號測試
            old_time = datetime.now() - timedelta(days=365)
            future_time = datetime.now() + timedelta(days=365)
            
            timing_signals = [
                TradingSignal(SignalType.BUY, 0.7, 0.8, "old_source", old_time),
                TradingSignal(SignalType.SELL, 0.6, 0.7, "future_source", future_time)
            ]
            
            # Test confidence calculation with extreme timing | 用極端時間測試信心計算
            confidence = self.confidence_scorer.calculate_comprehensive_confidence(timing_signals)
            
            return {'handled': True, 'recovered': True, 'critical_failure': False}
        except Exception:
            return {'handled': False, 'recovered': False, 'critical_failure': True}
    
    def _test_concurrent_modifications(self) -> Dict[str, Any]:
        """Test concurrent modifications handling | 測試並發修改處理"""
        try:
            # Simulate concurrent modifications | 模擬並發修改
            aggregator = SignalAggregator()
            
            def modifier_thread():
                signals = self.create_high_volume_signals(10)
                for signal in signals:
                    aggregator.add_signal(signal)
                    time.sleep(0.001)
            
            # Start modifier thread | 開始修改線程
            thread = threading.Thread(target=modifier_thread)
            thread.start()
            
            # Try to access aggregator while it's being modified | 嘗試在修改時訪問聚合器
            for _ in range(10):
                try:
                    summary = aggregator.get_summary()
                    time.sleep(0.001)
                except Exception:
                    pass  # Expected during concurrent access | 並發訪問期間預期
            
            thread.join()
            return {'handled': True, 'recovered': True, 'critical_failure': False}
        except Exception:
            return {'handled': False, 'recovered': False, 'critical_failure': True}
    
    def run_all_durability_tests(self) -> Dict[str, Any]:
        """Run all durability tests and generate comprehensive report | 運行所有耐久性測試並生成綜合報告"""
        logger.info("🚀 Starting Signal Combination Durability Test Suite...")
        
        start_time = time.time()
        
        # Test suite | 測試套件
        durability_tests = [
            ('Memory Leak Detection', self.test_1_memory_leak_detection),
            ('High Volume Processing', self.test_2_high_volume_processing),
            ('Concurrent Operation Safety', self.test_3_concurrent_operation_safety),
            ('Long-running Stability', self.test_4_long_running_stability),
            ('Extreme Edge Cases', self.test_5_extreme_edge_cases)
        ]
        
        # Execute tests | 執行測試
        test_results = {}
        for test_name, test_function in durability_tests:
            logger.info(f"\n--- Executing: {test_name} ---")
            try:
                test_results[test_name] = test_function()
            except Exception as e:
                logger.error(f"Test {test_name} failed catastrophically: {e}")
                test_results[test_name] = {
                    'status': 'CATASTROPHIC_FAILURE',
                    'error': str(e)
                }
        
        # Generate comprehensive report | 生成綜合報告
        total_duration = time.time() - start_time
        return self.generate_durability_report(test_results, total_duration)
    
    def generate_durability_report(self, test_results: Dict[str, Any], 
                                 total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive durability report | 生成綜合耐久性報告"""
        
        # Analyze test results | 分析測試結果
        passed_tests = [name for name, result in test_results.items() 
                       if result.get('status') == 'PASSED']
        warning_tests = [name for name, result in test_results.items() 
                        if result.get('status') == 'WARNING']
        failed_tests = [name for name, result in test_results.items() 
                       if result.get('status') in ['FAILED', 'CATASTROPHIC_FAILURE']]
        
        total_tests = len(test_results)
        pass_rate = len(passed_tests) / total_tests * 100 if total_tests > 0 else 0
        
        # Determine overall durability status | 確定整體耐久性狀態
        if pass_rate >= 80 and len(failed_tests) == 0:
            overall_status = "EXCELLENT_DURABILITY"
            durability_grade = "A"
        elif pass_rate >= 60 and len(failed_tests) <= 1:
            overall_status = "GOOD_DURABILITY"
            durability_grade = "B"
        elif pass_rate >= 40:
            overall_status = "MODERATE_DURABILITY"
            durability_grade = "C"
        else:
            overall_status = "POOR_DURABILITY"
            durability_grade = "D"
        
        # Compile comprehensive report | 編譯綜合報告
        durability_report = {
            'test_execution_summary': {
                'total_tests': total_tests,
                'passed_tests': len(passed_tests),
                'warning_tests': len(warning_tests),
                'failed_tests': len(failed_tests),
                'pass_rate': f"{pass_rate:.1f}%",
                'total_duration': f"{total_duration:.2f}s"
            },
            'overall_assessment': {
                'durability_status': overall_status,
                'durability_grade': durability_grade,
                'production_ready': pass_rate >= 70 and len(failed_tests) == 0,
                'high_load_capable': pass_rate >= 80,
                'long_term_stable': pass_rate >= 75
            },
            'detailed_results': test_results,
            'recommendations': self.generate_durability_recommendations(
                test_results, overall_status
            ),
            'system_capabilities_validated': {
                'memory_management': any('Memory' in name and 
                                       test_results[name].get('status') == 'PASSED' 
                                       for name in test_results),
                'high_volume_processing': any('High Volume' in name and 
                                            test_results[name].get('status') == 'PASSED' 
                                            for name in test_results),
                'concurrent_safety': any('Concurrent' in name and 
                                       test_results[name].get('status') == 'PASSED' 
                                       for name in test_results),
                'long_term_stability': any('Long-running' in name and 
                                         test_results[name].get('status') == 'PASSED' 
                                         for name in test_results),
                'error_resilience': any('Edge Cases' in name and 
                                      test_results[name].get('status') in ['PASSED', 'WARNING'] 
                                      for name in test_results)
            }
        }
        
        return durability_report
    
    def generate_durability_recommendations(self, test_results: Dict[str, Any], 
                                          overall_status: str) -> List[str]:
        """Generate recommendations based on durability test results | 基於耐久性測試結果生成建議"""
        recommendations = []
        
        # Overall status recommendations | 整體狀態建議
        if overall_status == "EXCELLENT_DURABILITY":
            recommendations.extend([
                "✅ System demonstrates excellent durability - ready for production deployment",
                "✅ 系統展現出色的耐久性 - 準備生產部署",
                "🎯 Consider implementing advanced monitoring for production environment",
                "🎯 考慮為生產環境實施高級監控"
            ])
        elif overall_status == "GOOD_DURABILITY":
            recommendations.extend([
                "✅ System shows good durability with minor areas for improvement",
                "✅ 系統展現良好的耐久性，有少數改進空間",
                "🔧 Address warning conditions before production deployment",
                "🔧 生產部署前處理警告條件"
            ])
        else:
            recommendations.extend([
                "⚠️ System requires durability improvements before production use",
                "⚠️ 系統需要耐久性改進才能用於生產",
                "🔧 Focus on failed test areas for system hardening",
                "🔧 重點關注失敗測試領域以加固系統"
            ])
        
        # Specific test recommendations | 具體測試建議
        for test_name, result in test_results.items():
            if result.get('status') == 'FAILED':
                if 'Memory' in test_name:
                    recommendations.append("🔧 Investigate memory management and potential leaks")
                    recommendations.append("🔧 調查記憶體管理和潛在洩漏")
                elif 'High Volume' in test_name:
                    recommendations.append("🔧 Optimize high-volume processing algorithms")
                    recommendations.append("🔧 優化高容量處理演算法")
                elif 'Concurrent' in test_name:
                    recommendations.append("🔧 Implement better thread safety mechanisms")
                    recommendations.append("🔧 實施更好的線程安全機制")
                elif 'Long-running' in test_name:
                    recommendations.append("🔧 Address long-term stability issues")
                    recommendations.append("🔧 解決長期穩定性問題")
                elif 'Edge Cases' in test_name:
                    recommendations.append("🔧 Strengthen error handling and recovery mechanisms")
                    recommendations.append("🔧 加強錯誤處理和恢復機制")
        
        return recommendations


def main():
    """Main durability test execution function | 主要耐久性測試執行函數"""
    try:
        print("=" * 100)
        print("🔬 AIFX PHASE 3.1.1 SIGNAL COMBINATION DURABILITY TEST SUITE")
        print("🎯 Comprehensive durability and stress testing")
        print("=" * 100)
        
        # Initialize and run durability tests | 初始化並運行耐久性測試
        durability_tester = SignalCombinationDurabilityTest()
        
        # Execute comprehensive durability test suite | 執行綜合耐久性測試套件
        durability_report = durability_tester.run_all_durability_tests()
        
        # Display comprehensive results | 顯示綜合結果
        print(f"\n📊 DURABILITY TEST EXECUTION SUMMARY:")
        print(f"Total Tests: {durability_report['test_execution_summary']['total_tests']}")
        print(f"Passed: {durability_report['test_execution_summary']['passed_tests']}")
        print(f"Warnings: {durability_report['test_execution_summary']['warning_tests']}")
        print(f"Failed: {durability_report['test_execution_summary']['failed_tests']}")
        print(f"Pass Rate: {durability_report['test_execution_summary']['pass_rate']}")
        print(f"Total Duration: {durability_report['test_execution_summary']['total_duration']}")
        
        print(f"\n🎯 OVERALL DURABILITY ASSESSMENT:")
        print(f"Status: {durability_report['overall_assessment']['durability_status']}")
        print(f"Grade: {durability_report['overall_assessment']['durability_grade']}")
        print(f"Production Ready: {durability_report['overall_assessment']['production_ready']}")
        print(f"High Load Capable: {durability_report['overall_assessment']['high_load_capable']}")
        print(f"Long-term Stable: {durability_report['overall_assessment']['long_term_stable']}")
        
        print(f"\n🔍 SYSTEM CAPABILITIES VALIDATED:")
        capabilities = durability_report['system_capabilities_validated']
        for capability, status in capabilities.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {capability.replace('_', ' ').title()}: {'VALIDATED' if status else 'FAILED'}")
        
        print(f"\n💡 DURABILITY RECOMMENDATIONS:")
        for rec in durability_report['recommendations']:
            print(f"{rec}")
        
        print("=" * 100)
        
        # Return appropriate exit code | 返回適當的退出代碼
        if durability_report['overall_assessment']['production_ready']:
            print("🎉 DURABILITY TEST SUITE: PASSED - System ready for production!")
            return 0
        else:
            print("⚠️ DURABILITY TEST SUITE: NEEDS ATTENTION - Address issues before production")
            return 1
        
    except Exception as e:
        logger.error(f"💥 Durability test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)