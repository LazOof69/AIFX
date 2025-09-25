"""
AIFX Performance Monitoring and Optimization | AIFX性能監控和優化

Tools for monitoring and optimizing system performance, memory usage,
and resource utilization in the AIFX trading system.

AIFX交易系統中用於監控和優化系統性能、內存使用和資源利用率的工具。

Features | 功能:
- Real-time performance monitoring | 實時性能監控
- Memory optimization utilities | 內存優化工具
- Caching and data structure optimization | 快取和數據結構優化
- Profiling and benchmarking tools | 分析和基準測試工具
"""

import time
import psutil
import threading
import functools
import gc
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio
import weakref
from memory_profiler import profile as memory_profile


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure | 性能指標數據結構"""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    function_name: str
    execution_time: float
    call_count: int = 1
    error_count: int = 0


class PerformanceMonitor:
    """
    Real-time performance monitoring system | 實時性能監控系統
    """

    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history: deque = deque(maxlen=history_size)
        self.function_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_time': 0.0,
            'call_count': 0,
            'avg_time': 0.0,
            'min_time': float('inf'),
            'max_time': 0.0,
            'error_count': 0
        })
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

    def start_monitoring(self, interval: float = 1.0):
        """Start continuous performance monitoring | 開始持續性能監控"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop performance monitoring | 停止性能監控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

    def _monitor_loop(self, interval: float):
        """Performance monitoring loop | 性能監控循環"""
        process = psutil.Process()

        while self.monitoring_active:
            try:
                # Get system metrics
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                memory_percent = process.memory_percent()

                # Create metrics record
                metrics = PerformanceMetrics(
                    timestamp=datetime.now(),
                    cpu_percent=cpu_percent,
                    memory_mb=memory_mb,
                    memory_percent=memory_percent,
                    function_name="system",
                    execution_time=0.0
                )

                with self.lock:
                    self.metrics_history.append(metrics)

            except Exception as e:
                print(f"Performance monitoring error: {e}")

            time.sleep(interval)

    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics | 獲取當前性能指標"""
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else None

    def get_average_metrics(self, duration_minutes: int = 5) -> Dict[str, float]:
        """Get average metrics over time period | 獲取時間段內的平均指標"""
        cutoff_time = datetime.now() - timedelta(minutes=duration_minutes)

        with self.lock:
            recent_metrics = [
                m for m in self.metrics_history
                if m.timestamp > cutoff_time
            ]

        if not recent_metrics:
            return {}

        return {
            'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'avg_memory_mb': sum(m.memory_mb for m in recent_metrics) / len(recent_metrics),
            'avg_memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            'sample_count': len(recent_metrics)
        }

    def get_function_stats(self) -> Dict[str, Dict]:
        """Get function performance statistics | 獲取函數性能統計"""
        with self.lock:
            return dict(self.function_stats)


class PerformanceProfiler:
    """
    Function performance profiling decorator | 函數性能分析裝飾器
    """

    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor

    def __call__(self, func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            error_occurred = False

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_occurred = True
                raise
            finally:
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                execution_time = end_time - start_time
                memory_delta = end_memory - start_memory

                # Update function statistics
                func_name = f"{func.__module__}.{func.__name__}"
                with self.monitor.lock:
                    stats = self.monitor.function_stats[func_name]
                    stats['total_time'] += execution_time
                    stats['call_count'] += 1
                    stats['avg_time'] = stats['total_time'] / stats['call_count']
                    stats['min_time'] = min(stats['min_time'], execution_time)
                    stats['max_time'] = max(stats['max_time'], execution_time)
                    if error_occurred:
                        stats['error_count'] += 1

                    # Add to metrics history
                    metrics = PerformanceMetrics(
                        timestamp=datetime.now(),
                        cpu_percent=psutil.Process().cpu_percent(),
                        memory_mb=end_memory,
                        memory_percent=psutil.Process().memory_percent(),
                        function_name=func_name,
                        execution_time=execution_time,
                        error_count=1 if error_occurred else 0
                    )
                    self.monitor.metrics_history.append(metrics)

        return wrapper


class CacheManager:
    """
    Intelligent caching system with TTL and memory management | 智能快取系統配合TTL和內存管理
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.Lock()

    def get(self, key: str) -> Any:
        """Get item from cache | 從快取中獲取項目"""
        with self.lock:
            if key not in self.cache:
                return None

            item = self.cache[key]

            # Check TTL
            if time.time() > item['expires_at']:
                del self.cache[key]
                del self.access_times[key]
                return None

            # Update access time
            self.access_times[key] = time.time()
            return item['value']

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache | 設置快取項目"""
        if ttl is None:
            ttl = self.default_ttl

        expires_at = time.time() + ttl

        with self.lock:
            # Check if we need to evict items
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            self.cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'created_at': time.time()
            }
            self.access_times[key] = time.time()

    def _evict_lru(self) -> None:
        """Evict least recently used items | 驅逐最近最少使用的項目"""
        if not self.access_times:
            return

        # Find oldest access time
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        del self.cache[oldest_key]
        del self.access_times[oldest_key]

    def clear_expired(self) -> int:
        """Clear expired items | 清除過期項目"""
        current_time = time.time()
        expired_keys = []

        with self.lock:
            for key, item in self.cache.items():
                if current_time > item['expires_at']:
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]
                del self.access_times[key]

        return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics | 獲取快取統計"""
        with self.lock:
            return {
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'hit_rate': 0.0,  # Would need hit/miss tracking
                'expired_count': self.clear_expired()
            }


class MemoryOptimizer:
    """
    Memory usage optimization utilities | 內存使用優化工具
    """

    @staticmethod
    def force_garbage_collection() -> Dict[str, int]:
        """Force garbage collection and return statistics | 強制垃圾回收並返回統計"""
        collected = gc.collect()
        return {
            'objects_collected': collected,
            'garbage_count': len(gc.garbage),
            'reference_count': sys.getrefcount
        }

    @staticmethod
    def get_memory_usage() -> Dict[str, float]:
        """Get current memory usage statistics | 獲取當前內存使用統計"""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024
        }

    @staticmethod
    @contextmanager
    def memory_limit(max_memory_mb: int):
        """Context manager to enforce memory limits | 上下文管理器以強制執行內存限制"""
        import resource

        # Set memory limit
        resource.setrlimit(
            resource.RLIMIT_AS,
            (max_memory_mb * 1024 * 1024, max_memory_mb * 1024 * 1024)
        )

        try:
            yield
        finally:
            # Reset memory limit
            resource.setrlimit(resource.RLIMIT_AS, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    @staticmethod
    def optimize_dataframe_memory(df) -> tuple:
        """Optimize pandas DataFrame memory usage | 優化pandas DataFrame內存使用"""
        original_memory = df.memory_usage(deep=True).sum()

        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = df[col].astype('int32')

        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = df[col].astype('float32')

        # Optimize object columns with few unique values
        for col in df.select_dtypes(include=['object']).columns:
            unique_count = df[col].nunique()
            total_count = len(df[col])
            if unique_count / total_count < 0.5:  # Less than 50% unique
                df[col] = df[col].astype('category')

        optimized_memory = df.memory_usage(deep=True).sum()
        memory_reduction = (original_memory - optimized_memory) / original_memory * 100

        return df, memory_reduction


class AsyncPerformanceOptimizer:
    """
    Asynchronous performance optimization utilities | 異步性能優化工具
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    async def run_in_executor(self, func: Callable, *args, **kwargs) -> Any:
        """Run CPU-intensive function in thread executor | 在線程執行器中運行CPU密集型函數"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            functools.partial(func, **kwargs),
            *args
        )

    async def batch_process(self, items: List[Any], func: Callable, batch_size: int = 100) -> List[Any]:
        """Process items in batches for better performance | 批量處理項目以提高性能"""
        results = []

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_tasks = [self.run_in_executor(func, item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

            # Allow other tasks to run
            await asyncio.sleep(0)

        return results

    def close(self):
        """Close the thread executor | 關閉線程執行器"""
        self.executor.shutdown(wait=True)


# Global instances | 全局實例
performance_monitor = PerformanceMonitor()
cache_manager = CacheManager()
memory_optimizer = MemoryOptimizer()
async_optimizer = AsyncPerformanceOptimizer()

# Convenience decorators | 便利裝飾器
def profile_performance(func: Callable) -> Callable:
    """Decorator to profile function performance | 分析函數性能的裝飾器"""
    profiler = PerformanceProfiler(performance_monitor)
    return profiler(func)


def cached(ttl: int = 300):
    """Decorator to cache function results | 快取函數結果的裝飾器"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            # Try to get from cache
            cached_result = cache_manager.get(key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(key, result, ttl)
            return result

        return wrapper
    return decorator


@contextmanager
def performance_context(operation_name: str):
    """Context manager for performance measurement | 性能測量的上下文管理器"""
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024

    try:
        yield
    finally:
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024

        print(f"Performance Report - {operation_name}:")
        print(f"  Execution time: {end_time - start_time:.4f} seconds")
        print(f"  Memory change: {end_memory - start_memory:.2f} MB")


# Initialize global monitoring
def start_global_monitoring():
    """Start global performance monitoring | 開始全局性能監控"""
    performance_monitor.start_monitoring()


def stop_global_monitoring():
    """Stop global performance monitoring | 停止全局性能監控"""
    performance_monitor.stop_monitoring()
    async_optimizer.close()