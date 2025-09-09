# -*- coding: utf-8 -*-
"""
AIFX Real-time Data Pipeline - Performance Testing
AIFX 實時數據管道 - 性能測試

This module tests performance and validates <50ms latency requirement.
該模組測試性能並驗證<50ms延遲要求。

Author: AIFX Development Team
Created: 2025-01-14
Version: 1.0.0
"""

import asyncio
import logging
import time
import statistics
import json
import threading
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class PerformanceMetrics:
    """Performance test metrics"""
    test_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    throughput_ops_sec: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    
    def calculate_stats(self) -> None:
        """Calculate performance statistics"""
        if not self.latencies_ms:
            return
        
        self.avg_latency_ms = statistics.mean(self.latencies_ms)
        self.min_latency_ms = min(self.latencies_ms)
        self.max_latency_ms = max(self.latencies_ms)
        
        # Percentiles
        self.p50_latency_ms = np.percentile(self.latencies_ms, 50)
        self.p95_latency_ms = np.percentile(self.latencies_ms, 95)
        self.p99_latency_ms = np.percentile(self.latencies_ms, 99)
        
        # Throughput
        if self.end_time and self.start_time:
            duration_sec = (self.end_time - self.start_time).total_seconds()
            if duration_sec > 0:
                self.throughput_ops_sec = self.successful_operations / duration_sec
    
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_operations == 0:
            return 0.0
        return (self.successful_operations / self.total_operations) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'test_name': self.test_name,
            'duration_sec': (self.end_time - self.start_time).total_seconds() if self.end_time else 0,
            'total_operations': self.total_operations,
            'successful_operations': self.successful_operations,
            'failed_operations': self.failed_operations,
            'success_rate_pct': self.success_rate(),
            'throughput_ops_sec': self.throughput_ops_sec,
            'latency_ms': {
                'avg': round(self.avg_latency_ms, 2),
                'min': round(self.min_latency_ms, 2),
                'max': round(self.max_latency_ms, 2),
                'p50': round(self.p50_latency_ms, 2),
                'p95': round(self.p95_latency_ms, 2),
                'p99': round(self.p99_latency_ms, 2)
            }
        }


class LoadGenerator:
    """Generates load for performance testing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load generation settings
        self.symbols = config.get('symbols', ['EURUSD', 'USDJPY'])
        self.base_rates = {
            'EURUSD': 1.0850,
            'USDJPY': 148.50,
            'GBPUSD': 1.2650,
            'USDCHF': 0.9150,
            'AUDUSD': 0.6750
        }
        
        # Test configuration
        self.test_duration = config.get('test_duration', 60)  # seconds
        self.target_rps = config.get('target_rps', 100)  # requests per second
        self.concurrent_threads = config.get('concurrent_threads', 10)
        
        # Data generation
        self.price_volatility = config.get('price_volatility', 0.001)  # 0.1% volatility
        
    def generate_forex_tick(self, symbol: str) -> Dict[str, Any]:
        """Generate realistic forex tick data"""
        base_rate = self.base_rates.get(symbol, 1.0000)
        
        # Add random price movement (normal distribution)
        price_change = np.random.normal(0, self.price_volatility)
        new_rate = base_rate * (1 + price_change)
        
        # Calculate bid/ask with typical spread
        spreads = {
            'EURUSD': 0.00015,
            'USDJPY': 0.015,
            'GBPUSD': 0.0002,
            'USDCHF': 0.0002,
            'AUDUSD': 0.0002
        }
        
        spread = spreads.get(symbol, 0.0002)
        bid = new_rate - (spread / 2)
        ask = new_rate + (spread / 2)
        
        return {
            'symbol': symbol,
            'bid': round(bid, 5),
            'ask': round(ask, 5),
            'spread': round(spread, 5),
            'timestamp': datetime.now(),
            'source': 'test_generator',
            'volume': np.random.uniform(1000, 10000)
        }
    
    async def generate_continuous_load(self, callback, duration: int) -> List[float]:
        """Generate continuous load for specified duration"""
        latencies = []
        end_time = time.time() + duration
        
        while time.time() < end_time:
            for symbol in self.symbols:
                start_time = time.time()
                
                try:
                    # Generate tick data
                    tick_data = self.generate_forex_tick(symbol)
                    
                    # Call the callback (e.g., data processing function)
                    await callback(tick_data)
                    
                    # Record latency
                    latency_ms = (time.time() - start_time) * 1000
                    latencies.append(latency_ms)
                    
                except Exception as e:
                    self.logger.error(f"Error generating load: {e}")
                    latencies.append(float('inf'))  # Mark as failed
            
            # Control rate
            await asyncio.sleep(1.0 / self.target_rps)
        
        return latencies


class LatencyTester:
    """Tests latency of various system components"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results: Dict[str, PerformanceMetrics] = {}
    
    async def test_data_processing_latency(self, data_processor, test_data: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Test data processing latency"""
        metrics = PerformanceMetrics(
            test_name="data_processing_latency",
            start_time=datetime.now()
        )
        
        self.logger.info(f"Testing data processing latency with {len(test_data)} samples")
        
        for data in test_data:
            start_time = time.time()
            
            try:
                # Process data
                await data_processor(data)
                
                latency_ms = (time.time() - start_time) * 1000
                metrics.latencies_ms.append(latency_ms)
                metrics.successful_operations += 1
                
            except Exception as e:
                self.logger.error(f"Data processing error: {e}")
                metrics.failed_operations += 1
            
            metrics.total_operations += 1
        
        metrics.end_time = datetime.now()
        metrics.calculate_stats()
        
        self.results[metrics.test_name] = metrics
        return metrics
    
    async def test_database_insert_latency(self, db_manager, test_ticks: List) -> PerformanceMetrics:
        """Test database insert latency"""
        metrics = PerformanceMetrics(
            test_name="database_insert_latency",
            start_time=datetime.now()
        )
        
        self.logger.info(f"Testing database insert latency with {len(test_ticks)} ticks")
        
        for tick in test_ticks:
            start_time = time.time()
            
            try:
                success = await db_manager.insert_tick(tick)
                
                latency_ms = (time.time() - start_time) * 1000
                metrics.latencies_ms.append(latency_ms)
                
                if success:
                    metrics.successful_operations += 1
                else:
                    metrics.failed_operations += 1
                    
            except Exception as e:
                self.logger.error(f"Database insert error: {e}")
                metrics.failed_operations += 1
            
            metrics.total_operations += 1
        
        metrics.end_time = datetime.now()
        metrics.calculate_stats()
        
        self.results[metrics.test_name] = metrics
        return metrics
    
    async def test_cache_latency(self, redis_manager, test_ticks: List) -> PerformanceMetrics:
        """Test Redis cache latency"""
        metrics = PerformanceMetrics(
            test_name="cache_latency",
            start_time=datetime.now()
        )
        
        self.logger.info(f"Testing cache latency with {len(test_ticks)} ticks")
        
        for tick in test_ticks:
            start_time = time.time()
            
            try:
                success = await redis_manager.cache_tick(tick)
                
                latency_ms = (time.time() - start_time) * 1000
                metrics.latencies_ms.append(latency_ms)
                
                if success:
                    metrics.successful_operations += 1
                else:
                    metrics.failed_operations += 1
                    
            except Exception as e:
                self.logger.error(f"Cache error: {e}")
                metrics.failed_operations += 1
            
            metrics.total_operations += 1
        
        metrics.end_time = datetime.now()
        metrics.calculate_stats()
        
        self.results[metrics.test_name] = metrics
        return metrics
    
    async def test_end_to_end_latency(self, pipeline_processor, test_data: List[Dict[str, Any]]) -> PerformanceMetrics:
        """Test end-to-end pipeline latency"""
        metrics = PerformanceMetrics(
            test_name="end_to_end_latency",
            start_time=datetime.now()
        )
        
        self.logger.info(f"Testing end-to-end latency with {len(test_data)} samples")
        
        for data in test_data:
            start_time = time.time()
            
            try:
                # Process through entire pipeline
                success = await pipeline_processor(data)
                
                latency_ms = (time.time() - start_time) * 1000
                metrics.latencies_ms.append(latency_ms)
                
                if success:
                    metrics.successful_operations += 1
                else:
                    metrics.failed_operations += 1
                    
            except Exception as e:
                self.logger.error(f"Pipeline processing error: {e}")
                metrics.failed_operations += 1
            
            metrics.total_operations += 1
        
        metrics.end_time = datetime.now()
        metrics.calculate_stats()
        
        self.results[metrics.test_name] = metrics
        return metrics


class ThroughputTester:
    """Tests system throughput under load"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results: Dict[str, PerformanceMetrics] = {}
    
    async def test_concurrent_processing(self, processor_func, test_data: List[Dict[str, Any]], 
                                       concurrent_workers: int = 10) -> PerformanceMetrics:
        """Test concurrent processing throughput"""
        metrics = PerformanceMetrics(
            test_name=f"concurrent_processing_{concurrent_workers}_workers",
            start_time=datetime.now()
        )
        
        self.logger.info(f"Testing concurrent processing with {concurrent_workers} workers")
        
        # Divide data among workers
        chunk_size = len(test_data) // concurrent_workers
        chunks = [test_data[i:i + chunk_size] for i in range(0, len(test_data), chunk_size)]
        
        async def process_chunk(chunk_data: List[Dict[str, Any]]) -> Tuple[int, int, List[float]]:
            """Process a chunk of data"""
            successful = 0
            failed = 0
            latencies = []
            
            for data in chunk_data:
                start_time = time.time()
                
                try:
                    await processor_func(data)
                    latency_ms = (time.time() - start_time) * 1000
                    latencies.append(latency_ms)
                    successful += 1
                except Exception as e:
                    failed += 1
                    
            return successful, failed, latencies
        
        # Run chunks concurrently
        tasks = [process_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for result in results:
            if isinstance(result, tuple):
                successful, failed, latencies = result
                metrics.successful_operations += successful
                metrics.failed_operations += failed
                metrics.latencies_ms.extend(latencies)
        
        metrics.total_operations = metrics.successful_operations + metrics.failed_operations
        metrics.end_time = datetime.now()
        metrics.calculate_stats()
        
        self.results[metrics.test_name] = metrics
        return metrics
    
    async def test_sustained_throughput(self, processor_func, load_generator: LoadGenerator,
                                      duration_seconds: int = 60) -> PerformanceMetrics:
        """Test sustained throughput over time"""
        metrics = PerformanceMetrics(
            test_name=f"sustained_throughput_{duration_seconds}s",
            start_time=datetime.now()
        )
        
        self.logger.info(f"Testing sustained throughput for {duration_seconds} seconds")
        
        async def process_callback(data: Dict[str, Any]) -> None:
            """Callback for processing data"""
            await processor_func(data)
        
        # Generate continuous load
        latencies = await load_generator.generate_continuous_load(process_callback, duration_seconds)
        
        # Process results
        metrics.latencies_ms = [lat for lat in latencies if lat != float('inf')]
        metrics.successful_operations = len(metrics.latencies_ms)
        metrics.failed_operations = len(latencies) - metrics.successful_operations
        metrics.total_operations = len(latencies)
        
        metrics.end_time = datetime.now()
        metrics.calculate_stats()
        
        self.results[metrics.test_name] = metrics
        return metrics


class PerformanceReporter:
    """Generates performance test reports"""
    
    def __init__(self, output_dir: str = "output/performance_reports"):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        import os
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_report(self, metrics: Dict[str, PerformanceMetrics], 
                       report_name: str = "performance_report") -> str:
        """Generate comprehensive performance report"""
        
        # Create summary data
        summary_data = []
        for test_name, metric in metrics.items():
            summary_data.append({
                'Test': metric.test_name,
                'Operations': metric.total_operations,
                'Success Rate %': round(metric.success_rate(), 2),
                'Throughput ops/sec': round(metric.throughput_ops_sec, 2),
                'Avg Latency ms': round(metric.avg_latency_ms, 2),
                'P95 Latency ms': round(metric.p95_latency_ms, 2),
                'P99 Latency ms': round(metric.p99_latency_ms, 2),
                'Max Latency ms': round(metric.max_latency_ms, 2)
            })
        
        # Create DataFrame for better formatting
        df = pd.DataFrame(summary_data)
        
        # Generate report
        report_lines = [
            "=" * 80,
            "AIFX REAL-TIME DATA PIPELINE - PERFORMANCE TEST REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "EXECUTIVE SUMMARY:",
            "================",
        ]
        
        # Add latency requirement check
        requirement_met = True
        for metric in metrics.values():
            if metric.p95_latency_ms > 50:  # 50ms requirement
                requirement_met = False
                break
        
        report_lines.extend([
            f"✅ Latency Requirement (<50ms): {'PASSED' if requirement_met else '❌ FAILED'}",
            f"Total Test Cases: {len(metrics)}",
            f"Overall Success Rate: {np.mean([m.success_rate() for m in metrics.values()]):.1f}%",
            "",
        ])
        
        # Add detailed results table
        report_lines.extend([
            "DETAILED RESULTS:",
            "================",
            df.to_string(index=False),
            "",
        ])
        
        # Add individual test analysis
        report_lines.append("INDIVIDUAL TEST ANALYSIS:")
        report_lines.append("========================")
        
        for test_name, metric in metrics.items():
            report_lines.extend([
                f"\n{test_name.upper()}:",
                f"  Duration: {(metric.end_time - metric.start_time).total_seconds():.1f}s",
                f"  Total Operations: {metric.total_operations:,}",
                f"  Successful: {metric.successful_operations:,} ({metric.success_rate():.1f}%)",
                f"  Failed: {metric.failed_operations:,}",
                f"  Throughput: {metric.throughput_ops_sec:.1f} ops/sec",
                f"  Latency Statistics (ms):",
                f"    Average: {metric.avg_latency_ms:.2f}",
                f"    Min: {metric.min_latency_ms:.2f}",
                f"    Max: {metric.max_latency_ms:.2f}",
                f"    P50: {metric.p50_latency_ms:.2f}",
                f"    P95: {metric.p95_latency_ms:.2f} {'✅' if metric.p95_latency_ms <= 50 else '❌'}",
                f"    P99: {metric.p99_latency_ms:.2f}",
            ])
        
        # Add recommendations
        report_lines.extend([
            "",
            "RECOMMENDATIONS:",
            "===============",
        ])
        
        if not requirement_met:
            report_lines.append("❌ CRITICAL: Latency requirement not met!")
            for test_name, metric in metrics.items():
                if metric.p95_latency_ms > 50:
                    report_lines.append(f"  - {test_name}: {metric.p95_latency_ms:.2f}ms (exceeds 50ms)")
        else:
            report_lines.append("✅ All latency requirements met!")
        
        # Performance recommendations
        avg_throughput = np.mean([m.throughput_ops_sec for m in metrics.values()])
        if avg_throughput < 100:
            report_lines.append("⚠️  Consider optimizing throughput (current: {:.1f} ops/sec)".format(avg_throughput))
        
        report_lines.append("\n" + "=" * 80)
        
        # Write report to file
        report_content = "\n".join(report_lines)
        report_file = f"{self.output_dir}/{report_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        self.logger.info(f"Performance report generated: {report_file}")
        return report_file
    
    def generate_charts(self, metrics: Dict[str, PerformanceMetrics], 
                       chart_name: str = "performance_charts") -> str:
        """Generate performance charts"""
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('AIFX Real-time Data Pipeline - Performance Analysis', fontsize=16)
        
        # Chart 1: Latency comparison
        test_names = list(metrics.keys())
        avg_latencies = [metrics[name].avg_latency_ms for name in test_names]
        p95_latencies = [metrics[name].p95_latency_ms for name in test_names]
        
        x = np.arange(len(test_names))
        width = 0.35
        
        ax1.bar(x - width/2, avg_latencies, width, label='Average Latency', alpha=0.8)
        ax1.bar(x + width/2, p95_latencies, width, label='P95 Latency', alpha=0.8)
        ax1.axhline(y=50, color='r', linestyle='--', label='50ms Requirement')
        ax1.set_xlabel('Test Cases')
        ax1.set_ylabel('Latency (ms)')
        ax1.set_title('Latency Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(test_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Chart 2: Throughput comparison
        throughputs = [metrics[name].throughput_ops_sec for name in test_names]
        ax2.bar(test_names, throughputs, color='green', alpha=0.7)
        ax2.set_xlabel('Test Cases')
        ax2.set_ylabel('Throughput (ops/sec)')
        ax2.set_title('Throughput Comparison')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Chart 3: Success rates
        success_rates = [metrics[name].success_rate() for name in test_names]
        colors = ['green' if rate >= 99 else 'orange' if rate >= 95 else 'red' for rate in success_rates]
        ax3.bar(test_names, success_rates, color=colors, alpha=0.7)
        ax3.set_xlabel('Test Cases')
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_title('Success Rate by Test')
        ax3.set_ylim(0, 100)
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Chart 4: Latency distribution (for first test with data)
        for test_name, metric in metrics.items():
            if metric.latencies_ms:
                ax4.hist(metric.latencies_ms, bins=50, alpha=0.7, label=test_name)
                break
        ax4.axvline(x=50, color='r', linestyle='--', label='50ms Requirement')
        ax4.set_xlabel('Latency (ms)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Latency Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save chart
        chart_file = f"{self.output_dir}/{chart_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Performance charts generated: {chart_file}")
        return chart_file


class PerformanceTestSuite:
    """Main performance test suite"""
    
    def __init__(self, config_path: str = "config/data-sources.yaml"):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.load_generator = LoadGenerator({
            'symbols': ['EURUSD', 'USDJPY'],
            'test_duration': 60,
            'target_rps': 100,
            'concurrent_threads': 10
        })
        
        self.latency_tester = LatencyTester()
        self.throughput_tester = ThroughputTester()
        self.reporter = PerformanceReporter()
        
        self.all_results: Dict[str, PerformanceMetrics] = {}
    
    async def run_full_performance_test(self) -> Dict[str, PerformanceMetrics]:
        """Run complete performance test suite"""
        self.logger.info("Starting full performance test suite")
        
        try:
            # Create mock implementations for testing
            async def mock_data_processor(data: Dict[str, Any]) -> bool:
                """Mock data processor for testing"""
                await asyncio.sleep(0.001)  # Simulate 1ms processing
                return True
            
            async def mock_pipeline_processor(data: Dict[str, Any]) -> bool:
                """Mock pipeline processor for testing"""
                await asyncio.sleep(0.005)  # Simulate 5ms processing
                return True
            
            # Generate test data
            test_data = []
            for i in range(1000):
                test_data.append(self.load_generator.generate_forex_tick('EURUSD'))
            
            self.logger.info(f"Generated {len(test_data)} test samples")
            
            # Test 1: Data processing latency
            self.logger.info("Testing data processing latency...")
            metrics1 = await self.latency_tester.test_data_processing_latency(
                mock_data_processor, test_data[:100]
            )
            self.all_results[metrics1.test_name] = metrics1
            
            # Test 2: End-to-end latency
            self.logger.info("Testing end-to-end latency...")
            metrics2 = await self.latency_tester.test_end_to_end_latency(
                mock_pipeline_processor, test_data[:100]
            )
            self.all_results[metrics2.test_name] = metrics2
            
            # Test 3: Concurrent processing
            self.logger.info("Testing concurrent processing...")
            metrics3 = await self.throughput_tester.test_concurrent_processing(
                mock_data_processor, test_data[:500], concurrent_workers=5
            )
            self.all_results[metrics3.test_name] = metrics3
            
            # Test 4: Sustained throughput (shorter duration for demo)
            self.logger.info("Testing sustained throughput...")
            load_gen = LoadGenerator({
                'symbols': ['EURUSD'],
                'target_rps': 50,
                'test_duration': 30
            })
            
            metrics4 = await self.throughput_tester.test_sustained_throughput(
                mock_data_processor, load_gen, duration_seconds=30
            )
            self.all_results[metrics4.test_name] = metrics4
            
            self.logger.info("Performance tests completed successfully")
            return self.all_results
            
        except Exception as e:
            self.logger.error(f"Performance test failed: {e}")
            raise
    
    def generate_reports(self) -> Tuple[str, str]:
        """Generate performance reports and charts"""
        if not self.all_results:
            raise ValueError("No test results available. Run tests first.")
        
        # Generate text report
        report_file = self.reporter.generate_report(self.all_results)
        
        # Generate charts
        chart_file = self.reporter.generate_charts(self.all_results)
        
        return report_file, chart_file
    
    def validate_requirements(self) -> Dict[str, bool]:
        """Validate performance requirements"""
        results = {
            'latency_50ms': True,
            'throughput_100ops': True,
            'success_rate_95pct': True,
            'overall_pass': False
        }
        
        for test_name, metrics in self.all_results.items():
            # Check latency requirement
            if metrics.p95_latency_ms > 50:
                results['latency_50ms'] = False
                self.logger.warning(f"{test_name}: P95 latency {metrics.p95_latency_ms:.2f}ms exceeds 50ms")
            
            # Check throughput requirement
            if metrics.throughput_ops_sec < 100 and 'throughput' in test_name:
                results['throughput_100ops'] = False
                self.logger.warning(f"{test_name}: Throughput {metrics.throughput_ops_sec:.1f} ops/sec below 100")
            
            # Check success rate requirement
            if metrics.success_rate() < 95:
                results['success_rate_95pct'] = False
                self.logger.warning(f"{test_name}: Success rate {metrics.success_rate():.1f}% below 95%")
        
        # Overall pass/fail
        results['overall_pass'] = all([
            results['latency_50ms'],
            results['throughput_100ops'],
            results['success_rate_95pct']
        ])
        
        return results


# Factory function
def create_performance_test_suite() -> PerformanceTestSuite:
    """Create performance test suite"""
    return PerformanceTestSuite()


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    async def main():
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        try:
            # Create test suite
            test_suite = create_performance_test_suite()
            
            print("Starting AIFX Real-time Data Pipeline Performance Tests...")
            print("=" * 60)
            
            # Run tests
            results = await test_suite.run_full_performance_test()
            
            print(f"\nCompleted {len(results)} test cases")
            
            # Validate requirements
            validation = test_suite.validate_requirements()
            print(f"\nRequirement Validation:")
            print(f"  Latency <50ms: {'✅ PASS' if validation['latency_50ms'] else '❌ FAIL'}")
            print(f"  Throughput >100 ops/sec: {'✅ PASS' if validation['throughput_100ops'] else '❌ FAIL'}")
            print(f"  Success Rate >95%: {'✅ PASS' if validation['success_rate_95pct'] else '❌ FAIL'}")
            print(f"  Overall: {'✅ PASS' if validation['overall_pass'] else '❌ FAIL'}")
            
            # Generate reports
            print("\nGenerating reports...")
            report_file, chart_file = test_suite.generate_reports()
            print(f"  Report: {report_file}")
            print(f"  Charts: {chart_file}")
            
            print("\nPerformance test suite completed successfully!")
            
            # Return appropriate exit code
            sys.exit(0 if validation['overall_pass'] else 1)
            
        except Exception as e:
            print(f"Performance test failed: {e}")
            sys.exit(1)
    
    # Run test suite
    asyncio.run(main())