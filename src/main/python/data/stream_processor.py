"""
AIFX Real-time Stream Processor | AIFX 實時流處理器
High-performance streaming data processor with validation and quality monitoring
高性能流式數據處理器，具有驗證和品質監控功能

Phase 4.2 Real-time Data Pipeline Component
第四階段 4.2 實時數據管道組件
"""

import asyncio
import time
import logging
import statistics
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import threading
import json
import math

import numpy as np
from scipy import stats
import redis

from .realtime_feed import ForexTick, DataSource
from .data_ingestion import MultiSourceDataIngestion, create_multi_source_ingestion

# Logger setup | 日誌設置
logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Data validation results | 數據驗證結果"""
    VALID = "VALID"
    INVALID_PRICE = "INVALID_PRICE"
    INVALID_SPREAD = "INVALID_SPREAD"
    STALE_DATA = "STALE_DATA"
    DUPLICATE_DATA = "DUPLICATE_DATA"
    ANOMALY_DETECTED = "ANOMALY_DETECTED"


class QualityScore(Enum):
    """Data quality score levels | 數據品質分數級別"""
    EXCELLENT = "EXCELLENT"  # 95-100%
    GOOD = "GOOD"           # 85-95%
    FAIR = "FAIR"           # 70-85%
    POOR = "POOR"           # 50-70%
    CRITICAL = "CRITICAL"   # <50%


@dataclass
class ValidationMetrics:
    """
    Data validation metrics | 數據驗證指標
    """
    total_validations: int = 0
    valid_count: int = 0
    invalid_price_count: int = 0
    invalid_spread_count: int = 0
    stale_data_count: int = 0
    duplicate_count: int = 0
    anomaly_count: int = 0
    
    @property
    def validation_rate(self) -> float:
        """Calculate validation success rate | 計算驗證成功率"""
        if self.total_validations == 0:
            return 1.0
        return self.valid_count / self.total_validations
    
    @property
    def quality_score(self) -> QualityScore:
        """Calculate overall quality score | 計算整體品質分數"""
        rate = self.validation_rate
        if rate >= 0.95:
            return QualityScore.EXCELLENT
        elif rate >= 0.85:
            return QualityScore.GOOD
        elif rate >= 0.70:
            return QualityScore.FAIR
        elif rate >= 0.50:
            return QualityScore.POOR
        else:
            return QualityScore.CRITICAL


@dataclass
class StreamMetrics:
    """
    Stream processing metrics | 流處理指標
    """
    ticks_processed: int = 0
    processing_rate: float = 0.0
    average_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    last_processed_time: Optional[datetime] = None
    
    # Latency tracking | 延遲追蹤
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def update_latency(self, latency_ms: float):
        """Update latency statistics | 更新延遲統計"""
        self.latency_samples.append(latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        
        if self.latency_samples:
            self.average_latency_ms = sum(self.latency_samples) / len(self.latency_samples)


@dataclass
class PriceHistory:
    """
    Price history for validation and anomaly detection | 用於驗證和異常檢測的價格歷史
    """
    symbol: str
    prices: deque = field(default_factory=lambda: deque(maxlen=100))  # Last 100 prices
    timestamps: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def add_price(self, price: float, timestamp: datetime):
        """Add new price to history | 添加新價格到歷史"""
        self.prices.append(price)
        self.timestamps.append(timestamp)
    
    @property
    def mean_price(self) -> float:
        """Calculate mean price | 計算平均價格"""
        return statistics.mean(self.prices) if self.prices else 0.0
    
    @property
    def std_deviation(self) -> float:
        """Calculate price standard deviation | 計算價格標準差"""
        return statistics.stdev(self.prices) if len(self.prices) > 1 else 0.0
    
    @property
    def latest_price(self) -> Optional[float]:
        """Get latest price | 獲取最新價格"""
        return self.prices[-1] if self.prices else None
    
    @property
    def price_change_rate(self) -> float:
        """Calculate recent price change rate | 計算近期價格變化率"""
        if len(self.prices) < 2:
            return 0.0
        return (self.prices[-1] - self.prices[-2]) / self.prices[-2]


class DataValidator:
    """
    Real-time data validator with anomaly detection | 實時數據驗證器，具有異常檢測功能
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data validator | 初始化數據驗證器
        """
        self.config = config.get('data_quality', {})
        self.metrics = ValidationMetrics()
        
        # Price history for validation | 用於驗證的價格歷史
        self.price_histories: Dict[str, PriceHistory] = {}
        
        # Validation thresholds | 驗證閾值
        self.max_price_change = self.config.get('price_change_threshold', 0.02)  # 2%
        self.max_spread_ratio = 0.01  # 1% of mid price
        self.max_data_age_seconds = self.config.get('min_data_freshness', 10)
        
        # Recent data tracking for duplicate detection | 重複檢測的最近數據追蹤
        self.recent_ticks: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        
        logger.info("Data validator initialized")
    
    def validate_tick(self, tick: ForexTick) -> Tuple[ValidationResult, Optional[str]]:
        """
        Validate incoming tick data | 驗證傳入的跳動數據
        
        Returns:
            Tuple of (ValidationResult, error_message)
        """
        self.metrics.total_validations += 1
        
        try:
            # Check for basic price validity | 檢查基本價格有效性
            if not self._is_valid_price(tick):
                self.metrics.invalid_price_count += 1
                return ValidationResult.INVALID_PRICE, f"Invalid price: bid={tick.bid}, ask={tick.ask}"
            
            # Check spread validity | 檢查點差有效性
            if not self._is_valid_spread(tick):
                self.metrics.invalid_spread_count += 1
                return ValidationResult.INVALID_SPREAD, f"Invalid spread: {tick.spread:.5f}"
            
            # Check data freshness | 檢查數據新鮮度
            if not self._is_fresh_data(tick):
                self.metrics.stale_data_count += 1
                return ValidationResult.STALE_DATA, f"Stale data: {tick.timestamp}"
            
            # Check for duplicates | 檢查重複數據
            if self._is_duplicate_tick(tick):
                self.metrics.duplicate_count += 1
                return ValidationResult.DUPLICATE_DATA, "Duplicate tick detected"
            
            # Check for price anomalies | 檢查價格異常
            if self._is_price_anomaly(tick):
                self.metrics.anomaly_count += 1
                return ValidationResult.ANOMALY_DETECTED, f"Price anomaly detected: {tick.mid_price}"
            
            # Update price history | 更新價格歷史
            self._update_price_history(tick)
            
            self.metrics.valid_count += 1
            return ValidationResult.VALID, None
            
        except Exception as e:
            logger.error(f"Error validating tick: {e}")
            return ValidationResult.INVALID_PRICE, f"Validation error: {str(e)}"
    
    def _is_valid_price(self, tick: ForexTick) -> bool:
        """Check if price values are valid | 檢查價格值是否有效"""
        # Check for positive values | 檢查正值
        if tick.bid <= 0 or tick.ask <= 0:
            return False
        
        # Check reasonable range for forex prices | 檢查外匯價格的合理範圍
        if tick.bid > 1000 or tick.ask > 1000:  # Most forex pairs are below 1000
            return False
        
        # Bid should be less than ask | 買價應低於賣價
        if tick.bid >= tick.ask:
            return False
        
        return True
    
    def _is_valid_spread(self, tick: ForexTick) -> bool:
        """Check if spread is within acceptable range | 檢查點差是否在可接受範圍內"""
        spread_ratio = tick.spread / tick.mid_price
        return spread_ratio <= self.max_spread_ratio
    
    def _is_fresh_data(self, tick: ForexTick) -> bool:
        """Check if data is fresh (not too old) | 檢查數據是否新鮮（不太舊）"""
        age_seconds = (datetime.now() - tick.timestamp).total_seconds()
        return age_seconds <= self.max_data_age_seconds
    
    def _is_duplicate_tick(self, tick: ForexTick) -> bool:
        """Check for duplicate tick data | 檢查重複跳動數據"""
        key = f"{tick.symbol}:{tick.source.value}"
        recent_ticks = self.recent_ticks[key]
        
        # Check if identical tick exists in recent history | 檢查最近歷史中是否存在相同跳動
        for recent_tick in recent_ticks:
            if (recent_tick['bid'] == tick.bid and 
                recent_tick['ask'] == tick.ask and 
                abs((recent_tick['timestamp'] - tick.timestamp).total_seconds()) < 1):
                return True
        
        # Add current tick to recent history | 將當前跳動添加到最近歷史
        recent_ticks.append({
            'bid': tick.bid,
            'ask': tick.ask,
            'timestamp': tick.timestamp
        })
        
        return False
    
    def _is_price_anomaly(self, tick: ForexTick) -> bool:
        """Detect price anomalies using statistical methods | 使用統計方法檢測價格異常"""
        if tick.symbol not in self.price_histories:
            return False  # Not enough history for anomaly detection
        
        history = self.price_histories[tick.symbol]
        
        if len(history.prices) < 10:  # Need at least 10 samples
            return False
        
        # Check for sudden price jumps | 檢查價格突然跳躍
        price_change = abs(history.price_change_rate)
        if price_change > self.max_price_change:
            return True
        
        # Statistical anomaly detection using z-score | 使用z分數進行統計異常檢測
        if history.std_deviation > 0:
            z_score = abs(tick.mid_price - history.mean_price) / history.std_deviation
            if z_score > 3:  # 3 standard deviations
                return True
        
        return False
    
    def _update_price_history(self, tick: ForexTick):
        """Update price history for the symbol | 更新品種的價格歷史"""
        if tick.symbol not in self.price_histories:
            self.price_histories[tick.symbol] = PriceHistory(tick.symbol)
        
        self.price_histories[tick.symbol].add_price(tick.mid_price, tick.timestamp)


class StreamProcessor:
    """
    High-performance real-time stream processor | 高性能實時流處理器
    """
    
    def __init__(self, config_path: str = "config/data-sources.yaml"):
        """
        Initialize stream processor | 初始化流處理器
        """
        # Load configuration | 載入配置
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Core components | 核心組件
        self.ingestion_system = create_multi_source_ingestion(config_path)
        self.data_validator = DataValidator(self.config)
        
        # Processing metrics | 處理指標
        self.stream_metrics = StreamMetrics()
        self.start_time = datetime.now()
        
        # Processing pipeline | 處理管道
        self.processors: List[Callable[[ForexTick], None]] = []
        
        # Quality monitoring | 品質監控
        self.quality_alerts: List[Callable[[str, Any], None]] = []
        
        # Redis connection for streaming | 用於流式處理的Redis連接
        self.redis_client: Optional[redis.Redis] = None
        self._init_redis_connection()
        
        # Threading | 線程控制
        self.processing_threads: List[threading.Thread] = []
        self.shutdown_event = threading.Event()
        
        # Initialize ingestion callbacks | 初始化攝取回調
        self.ingestion_system.add_data_processor(self._process_tick_batch)
        
        logger.info("Stream processor initialized")
    
    def _init_redis_connection(self):
        """Initialize Redis connection for streaming | 初始化用於流式處理的Redis連接"""
        try:
            cache_config = self.config.get('caching', {}).get('redis', {})
            
            self.redis_client = redis.Redis(
                host=cache_config.get('host', 'localhost'),
                port=cache_config.get('port', 6379),
                db=cache_config.get('db', 1),
                password=cache_config.get('password'),
                decode_responses=True
            )
            
            self.redis_client.ping()
            logger.info("Redis connection established for stream processing")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def add_processor(self, processor: Callable[[ForexTick], None]):
        """Add tick processor to pipeline | 添加跳動處理器到管道"""
        self.processors.append(processor)
        logger.info(f"Added processor: {processor.__name__}")
    
    def add_quality_alert_handler(self, handler: Callable[[str, Any], None]):
        """Add quality alert handler | 添加品質警報處理器"""
        self.quality_alerts.append(handler)
        logger.info(f"Added quality alert handler: {handler.__name__}")
    
    def start(self, sources: Optional[List[DataSource]] = None):
        """Start stream processing | 啟動流處理"""
        logger.info("Starting stream processor...")
        
        # Start data ingestion | 啟動數據攝取
        self.ingestion_system.start(sources)
        
        # Start quality monitoring | 啟動品質監控
        self._start_quality_monitoring()
        
        logger.info("Stream processor started successfully")
    
    def stop(self):
        """Stop stream processing | 停止流處理"""
        logger.info("Stopping stream processor...")
        
        self.shutdown_event.set()
        
        # Stop ingestion system | 停止攝取系統
        self.ingestion_system.stop()
        
        # Wait for processing threads | 等待處理線程
        for thread in self.processing_threads:
            thread.join(timeout=10)
        
        # Close Redis connection | 關閉Redis連接
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("Stream processor stopped")
    
    def _process_tick_batch(self, ticks: List[ForexTick]):
        """Process batch of ticks | 處理跳動批次"""
        for tick in ticks:
            self._process_single_tick(tick)
    
    def _process_single_tick(self, tick: ForexTick):
        """Process individual tick | 處理單個跳動"""
        process_start_time = time.time()
        
        try:
            # Validate tick data | 驗證跳動數據
            validation_result, error_message = self.data_validator.validate_tick(tick)
            
            if validation_result == ValidationResult.VALID:
                # Process through pipeline | 通過管道處理
                for processor in self.processors:
                    try:
                        processor(tick)
                    except Exception as e:
                        logger.error(f"Error in processor {processor.__name__}: {e}")
                
                # Update to Redis streams | 更新到Redis流
                self._update_redis_stream(tick)
                
            else:
                # Handle validation errors | 處理驗證錯誤
                self._handle_validation_error(tick, validation_result, error_message)
            
            # Update metrics | 更新指標
            self.stream_metrics.ticks_processed += 1
            self.stream_metrics.last_processed_time = datetime.now()
            
            # Update latency | 更新延遲
            processing_time = (time.time() - process_start_time) * 1000  # Convert to ms
            self.stream_metrics.update_latency(processing_time)
            
            # Calculate processing rate | 計算處理速率
            uptime = (datetime.now() - self.start_time).total_seconds()
            if uptime > 0:
                self.stream_metrics.processing_rate = self.stream_metrics.ticks_processed / uptime
            
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
    
    def _update_redis_stream(self, tick: ForexTick):
        """Update Redis stream with processed tick | 使用處理過的跳動更新Redis流"""
        if not self.redis_client:
            return
        
        try:
            stream_key = f"forex:processed:{tick.symbol}"
            
            # Add to stream with processing metadata | 添加到流中並包含處理元數據
            stream_data = tick.to_dict()
            stream_data.update({
                'processed_at': datetime.now().isoformat(),
                'processing_latency_ms': self.stream_metrics.latency_samples[-1] if self.stream_metrics.latency_samples else 0,
                'validation_status': 'valid'
            })
            
            self.redis_client.xadd(stream_key, stream_data, maxlen=10000)
            
        except Exception as e:
            logger.error(f"Failed to update Redis stream: {e}")
    
    def _handle_validation_error(self, tick: ForexTick, result: ValidationResult, error_message: Optional[str]):
        """Handle validation errors | 處理驗證錯誤"""
        logger.warning(f"Validation failed for {tick.symbol}: {result.value} - {error_message}")
        
        # Send quality alert if configured | 如果配置了，發送品質警報
        for alert_handler in self.quality_alerts:
            try:
                alert_handler(f"validation_error_{result.value.lower()}", {
                    'symbol': tick.symbol,
                    'source': tick.source.value,
                    'result': result.value,
                    'error_message': error_message,
                    'timestamp': tick.timestamp.isoformat()
                })
            except Exception as e:
                logger.error(f"Error in quality alert handler: {e}")
    
    def _start_quality_monitoring(self):
        """Start quality monitoring thread | 啟動品質監控線程"""
        def quality_monitor():
            """Quality monitoring worker | 品質監控工作器"""
            while not self.shutdown_event.is_set():
                try:
                    # Check data quality metrics | 檢查數據品質指標
                    self._check_quality_metrics()
                    
                    # Check processing performance | 檢查處理性能
                    self._check_processing_performance()
                    
                    # Sleep for monitoring interval | 按監控間隔睡眠
                    self.shutdown_event.wait(30)  # 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in quality monitoring: {e}")
                    self.shutdown_event.wait(5)
        
        quality_thread = threading.Thread(target=quality_monitor, name="QualityMonitor", daemon=True)
        quality_thread.start()
        self.processing_threads.append(quality_thread)
        
        logger.info("Quality monitoring started")
    
    def _check_quality_metrics(self):
        """Check data quality metrics and send alerts | 檢查數據品質指標並發送警報"""
        validation_metrics = self.data_validator.metrics
        quality_score = validation_metrics.quality_score
        
        # Send alert if quality is poor | 如果品質差則發送警報
        if quality_score in [QualityScore.POOR, QualityScore.CRITICAL]:
            for alert_handler in self.quality_alerts:
                try:
                    alert_handler("poor_data_quality", {
                        'quality_score': quality_score.value,
                        'validation_rate': validation_metrics.validation_rate,
                        'total_validations': validation_metrics.total_validations,
                        'error_breakdown': {
                            'invalid_price': validation_metrics.invalid_price_count,
                            'invalid_spread': validation_metrics.invalid_spread_count,
                            'stale_data': validation_metrics.stale_data_count,
                            'duplicates': validation_metrics.duplicate_count,
                            'anomalies': validation_metrics.anomaly_count
                        }
                    })
                except Exception as e:
                    logger.error(f"Error sending quality alert: {e}")
    
    def _check_processing_performance(self):
        """Check processing performance and send alerts | 檢查處理性能並發送警報"""
        # Check average latency | 檢查平均延遲
        if self.stream_metrics.average_latency_ms > 100:  # 100ms threshold
            for alert_handler in self.quality_alerts:
                try:
                    alert_handler("high_processing_latency", {
                        'average_latency_ms': self.stream_metrics.average_latency_ms,
                        'max_latency_ms': self.stream_metrics.max_latency_ms,
                        'processing_rate': self.stream_metrics.processing_rate
                    })
                except Exception as e:
                    logger.error(f"Error sending performance alert: {e}")
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status | 獲取當前處理狀態"""
        validation_metrics = self.data_validator.metrics
        ingestion_status = self.ingestion_system.get_ingestion_status()
        
        return {
            'stream_processor': {
                'ticks_processed': self.stream_metrics.ticks_processed,
                'processing_rate': self.stream_metrics.processing_rate,
                'average_latency_ms': self.stream_metrics.average_latency_ms,
                'max_latency_ms': self.stream_metrics.max_latency_ms,
                'last_processed_time': self.stream_metrics.last_processed_time.isoformat() if self.stream_metrics.last_processed_time else None
            },
            'data_validation': {
                'total_validations': validation_metrics.total_validations,
                'validation_rate': validation_metrics.validation_rate,
                'quality_score': validation_metrics.quality_score.value,
                'error_breakdown': {
                    'invalid_price': validation_metrics.invalid_price_count,
                    'invalid_spread': validation_metrics.invalid_spread_count,
                    'stale_data': validation_metrics.stale_data_count,
                    'duplicates': validation_metrics.duplicate_count,
                    'anomalies': validation_metrics.anomaly_count
                }
            },
            'ingestion_system': ingestion_status
        }
    
    def get_symbol_quality_report(self, symbol: str) -> Dict[str, Any]:
        """Get quality report for specific symbol | 獲取特定品種的品質報告"""
        if symbol not in self.data_validator.price_histories:
            return {'error': f'No data available for symbol {symbol}'}
        
        history = self.data_validator.price_histories[symbol]
        
        return {
            'symbol': symbol,
            'price_samples': len(history.prices),
            'latest_price': history.latest_price,
            'mean_price': history.mean_price,
            'price_volatility': history.std_deviation,
            'recent_change_rate': history.price_change_rate,
            'data_timestamps': [ts.isoformat() for ts in list(history.timestamps)[-5:]]  # Last 5 timestamps
        }


# Factory function | 工廠函數
def create_stream_processor(config_path: str = "config/data-sources.yaml") -> StreamProcessor:
    """
    Create and configure stream processor | 創建並配置流處理器
    """
    return StreamProcessor(config_path)


if __name__ == "__main__":
    # Example usage | 使用示例
    def example_tick_processor(tick: ForexTick):
        """Example tick processor | 示例跳動處理器"""
        print(f"Processing: {tick.symbol} | {tick.mid_price:.5f} | {tick.source.value}")
    
    def example_quality_alert(alert_type: str, data: Any):
        """Example quality alert handler | 示例品質警報處理器"""
        print(f"QUALITY ALERT [{alert_type}]: {data}")
    
    # Create and start processor | 創建並啟動處理器
    processor = create_stream_processor()
    processor.add_processor(example_tick_processor)
    processor.add_quality_alert_handler(example_quality_alert)
    
    try:
        processor.start([DataSource.YAHOO])
        
        # Monitor processing | 監控處理
        while True:
            time.sleep(30)
            status = processor.get_processing_status()
            
            print(f"\n=== Stream Processing Status ===")
            print(f"Ticks processed: {status['stream_processor']['ticks_processed']}")
            print(f"Processing rate: {status['stream_processor']['processing_rate']:.2f} ticks/sec")
            print(f"Average latency: {status['stream_processor']['average_latency_ms']:.2f}ms")
            print(f"Data quality: {status['data_validation']['quality_score']}")
            print(f"Validation rate: {status['data_validation']['validation_rate']:.2%}")
            
    except KeyboardInterrupt:
        print("\nStopping processor...")
        processor.stop()