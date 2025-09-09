"""
AIFX Multi-Source Data Ingestion System | AIFX 多源數據攝取系統
Advanced data ingestion pipeline with intelligent routing and fallback
高級數據攝取管道，具有智能路由和備援機制

Phase 4.2 Real-time Data Pipeline Component
第四階段 4.2 實時數據管道組件
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import requests
import aiohttp
import yaml
import redis
from contextlib import asynccontextmanager

from .realtime_feed import (
    RealTimeForexFeed, ForexTick, DataSource, ConnectionStatus,
    create_realtime_forex_feed
)
from ..database.connection_pool import get_pool_manager
from ..monitoring.database_monitor import create_database_monitor

# Logger setup | 日誌設置
logger = logging.getLogger(__name__)


class IngestionStatus(Enum):
    """Data ingestion status | 數據攝取狀態"""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    PAUSED = "PAUSED"


class DataPriority(Enum):
    """Data priority levels | 數據優先級"""
    CRITICAL = "CRITICAL"    # Real-time trading data
    HIGH = "HIGH"           # Market indicators
    MEDIUM = "MEDIUM"       # Analytics data
    LOW = "LOW"            # Historical data


@dataclass
class IngestionMetrics:
    """
    Data ingestion performance metrics | 數據攝取性能指標
    """
    total_records_ingested: int = 0
    records_per_second: float = 0.0
    error_count: int = 0
    last_update_time: Optional[datetime] = None
    average_processing_time_ms: float = 0.0
    buffer_utilization: float = 0.0
    source_health_scores: Dict[str, float] = field(default_factory=dict)
    
    # Performance tracking | 性能追蹤
    processing_times: List[float] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate ingestion uptime | 計算攝取正常運行時間"""
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate | 計算成功率"""
        total_attempts = self.total_records_ingested + self.error_count
        if total_attempts == 0:
            return 1.0
        return self.total_records_ingested / total_attempts
    
    def update_processing_time(self, processing_time: float):
        """Update processing time statistics | 更新處理時間統計"""
        self.processing_times.append(processing_time)
        
        # Keep only last 1000 measurements | 只保留最後1000個測量值
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]
        
        # Update average | 更新平均值
        self.average_processing_time_ms = sum(self.processing_times) / len(self.processing_times)


@dataclass
class IngestionBuffer:
    """
    Data ingestion buffer for batching | 數據攝取緩衝區用於批處理
    """
    max_size: int
    flush_interval: float
    priority: DataPriority
    
    # Buffer state | 緩衝區狀態
    data: List[ForexTick] = field(default_factory=list)
    last_flush_time: datetime = field(default_factory=datetime.now)
    total_flushes: int = 0
    
    @property
    def is_full(self) -> bool:
        """Check if buffer is full | 檢查緩衝區是否已滿"""
        return len(self.data) >= self.max_size
    
    @property
    def should_flush(self) -> bool:
        """Check if buffer should be flushed | 檢查是否應該刷新緩衝區"""
        time_elapsed = (datetime.now() - self.last_flush_time).total_seconds()
        return self.is_full or time_elapsed >= self.flush_interval
    
    @property
    def utilization(self) -> float:
        """Calculate buffer utilization | 計算緩衝區利用率"""
        return len(self.data) / self.max_size if self.max_size > 0 else 0.0
    
    def add_tick(self, tick: ForexTick) -> bool:
        """
        Add tick to buffer | 添加跳動到緩衝區
        
        Returns:
            True if added successfully, False if buffer is full
        """
        if len(self.data) < self.max_size:
            self.data.append(tick)
            return True
        return False
    
    def flush(self) -> List[ForexTick]:
        """
        Flush buffer and return data | 刷新緩衝區並返回數據
        """
        data_to_return = self.data.copy()
        self.data.clear()
        self.last_flush_time = datetime.now()
        self.total_flushes += 1
        return data_to_return


class DataRouter:
    """
    Intelligent data routing system | 智能數據路由系統
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data router | 初始化數據路由器
        """
        self.config = config
        self.source_priorities: Dict[DataSource, int] = {}
        self.source_health_scores: Dict[DataSource, float] = {}
        self.active_sources: Set[DataSource] = set()
        
        # Load source priorities | 載入數據源優先級
        self._load_source_priorities()
        
        logger.info("Data router initialized")
    
    def _load_source_priorities(self):
        """
        Load source priorities from configuration | 從配置載入數據源優先級
        """
        for source_name, source_config in self.config.get('forex_data_sources', {}).items():
            try:
                source = DataSource(source_name)
                priority = source_config.get('priority', 99)
                self.source_priorities[source] = priority
                self.source_health_scores[source] = 0.0  # Initialize health score
                
                if source_config.get('enabled', False):
                    self.active_sources.add(source)
                    
            except ValueError:
                logger.warning(f"Unknown data source in config: {source_name}")
    
    def get_best_source(self, symbol: str) -> Optional[DataSource]:
        """
        Get the best available data source for a symbol | 獲取品種的最佳可用數據源
        """
        # Filter active sources | 過濾活躍數據源
        available_sources = [
            (source, self.source_priorities.get(source, 99), self.source_health_scores.get(source, 0.0))
            for source in self.active_sources
        ]
        
        if not available_sources:
            return None
        
        # Sort by priority (lower number = higher priority) and health score | 按優先級和健康分數排序
        available_sources.sort(key=lambda x: (x[1], -x[2]))
        
        return available_sources[0][0]
    
    def update_source_health(self, source: DataSource, health_score: float):
        """
        Update source health score | 更新數據源健康分數
        """
        self.source_health_scores[source] = max(0.0, min(100.0, health_score))
        
        # Remove source if health is too low | 如果健康狀況太差則移除數據源
        if health_score < 20.0 and source in self.active_sources:
            self.active_sources.remove(source)
            logger.warning(f"Removed unhealthy source: {source.value} (health: {health_score})")
        
        # Re-add source if health improves | 如果健康狀況改善則重新添加數據源
        elif health_score >= 50.0 and source not in self.active_sources:
            self.active_sources.add(source)
            logger.info(f"Re-added healthy source: {source.value} (health: {health_score})")
    
    def should_use_fallback(self, primary_source: DataSource) -> bool:
        """
        Determine if fallback source should be used | 判斷是否應該使用備援數據源
        """
        health_score = self.source_health_scores.get(primary_source, 0.0)
        return health_score < 30.0  # Use fallback if health is below 30%


class MultiSourceDataIngestion:
    """
    Multi-source data ingestion system with intelligent routing and quality control
    多源數據攝取系統，具有智能路由和品質控制
    """
    
    def __init__(self, config_path: str = "config/data-sources.yaml"):
        """
        Initialize multi-source data ingestion | 初始化多源數據攝取
        """
        # Load configuration | 載入配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Core components | 核心組件
        self.realtime_feed = create_realtime_forex_feed(config_path)
        self.data_router = DataRouter(self.config)
        
        # Database connections | 資料庫連接
        self.db_pool_manager = get_pool_manager()
        self.redis_client: Optional[redis.Redis] = None
        
        # Ingestion state | 攝取狀態
        self.status = IngestionStatus.STOPPED
        self.metrics = IngestionMetrics()
        
        # Data buffers | 數據緩衝區
        self.buffers: Dict[DataPriority, IngestionBuffer] = {}
        self._init_buffers()
        
        # Data processors | 數據處理器
        self.data_processors: List[Callable[[List[ForexTick]], None]] = []
        
        # Threading | 線程控制
        self.worker_threads: List[threading.Thread] = []
        self.shutdown_event = threading.Event()
        
        # Initialize components | 初始化組件
        self._init_redis_connection()
        self._init_realtime_feed()
        
        logger.info("Multi-source data ingestion system initialized")
    
    def _init_buffers(self):
        """
        Initialize data buffers for different priorities | 為不同優先級初始化數據緩衝區
        """
        buffer_config = self.config.get('data_ingestion', {})
        
        # Critical data buffer (real-time trading) | 關鍵數據緩衝區（實時交易）
        self.buffers[DataPriority.CRITICAL] = IngestionBuffer(
            max_size=100,  # Small buffer for immediate processing
            flush_interval=1.0,  # 1 second
            priority=DataPriority.CRITICAL
        )
        
        # High priority buffer (market data) | 高優先級緩衝區（市場數據）
        self.buffers[DataPriority.HIGH] = IngestionBuffer(
            max_size=500,
            flush_interval=5.0,  # 5 seconds
            priority=DataPriority.HIGH
        )
        
        # Medium priority buffer (analytics) | 中等優先級緩衝區（分析數據）
        self.buffers[DataPriority.MEDIUM] = IngestionBuffer(
            max_size=buffer_config.get('buffer_size', 1000),
            flush_interval=buffer_config.get('flush_interval', 10.0),
            priority=DataPriority.MEDIUM
        )
    
    def _init_redis_connection(self):
        """
        Initialize Redis connection | 初始化Redis連接
        """
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
            logger.info("Redis connection established for data ingestion")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _init_realtime_feed(self):
        """
        Initialize real-time data feed | 初始化實時數據源
        """
        # Add tick handler | 添加跳動處理器
        self.realtime_feed.add_data_subscriber(self._handle_incoming_tick)
    
    def add_data_processor(self, processor: Callable[[List[ForexTick]], None]):
        """
        Add data processor callback | 添加數據處理器回調
        """
        self.data_processors.append(processor)
        logger.info(f"Added data processor: {processor.__name__}")
    
    def start(self, sources: Optional[List[DataSource]] = None):
        """
        Start data ingestion | 啟動數據攝取
        """
        if self.status == IngestionStatus.RUNNING:
            logger.warning("Data ingestion is already running")
            return
        
        logger.info("Starting multi-source data ingestion...")
        self.status = IngestionStatus.STARTING
        
        try:
            # Start real-time feed | 啟動實時數據源
            self.realtime_feed.start(sources)
            
            # Start buffer flush workers | 啟動緩衝區刷新工作器
            self._start_buffer_workers()
            
            # Start health monitoring | 啟動健康監控
            self._start_health_monitoring()
            
            self.status = IngestionStatus.RUNNING
            self.metrics.start_time = datetime.now()
            
            logger.info("Multi-source data ingestion started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start data ingestion: {e}")
            self.status = IngestionStatus.ERROR
            raise
    
    def stop(self):
        """
        Stop data ingestion | 停止數據攝取
        """
        logger.info("Stopping multi-source data ingestion...")
        
        self.status = IngestionStatus.STOPPED
        self.shutdown_event.set()
        
        # Stop real-time feed | 停止實時數據源
        self.realtime_feed.stop()
        
        # Wait for worker threads to finish | 等待工作線程完成
        for thread in self.worker_threads:
            thread.join(timeout=10)
        
        # Flush remaining data | 刷新剩餘數據
        self._flush_all_buffers()
        
        # Close connections | 關閉連接
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("Multi-source data ingestion stopped")
    
    def _handle_incoming_tick(self, tick: ForexTick):
        """
        Handle incoming tick data from real-time feed | 處理來自實時數據源的跳動數據
        """
        start_time = time.time()
        
        try:
            # Determine data priority | 確定數據優先級
            priority = self._get_data_priority(tick)
            
            # Add to appropriate buffer | 添加到適當的緩衝區
            buffer = self.buffers.get(priority, self.buffers[DataPriority.MEDIUM])
            
            if buffer.add_tick(tick):
                self.metrics.total_records_ingested += 1
                self.metrics.last_update_time = datetime.now()
                
                # Update router health scores | 更新路由器健康分數
                self._update_source_health(tick.source, True)
                
            else:
                logger.warning(f"Buffer full for priority {priority.value}, dropping tick")
                
        except Exception as e:
            logger.error(f"Error handling incoming tick: {e}")
            self.metrics.error_count += 1
            self._update_source_health(tick.source, False)
        
        finally:
            # Update processing time metrics | 更新處理時間指標
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.metrics.update_processing_time(processing_time)
    
    def _get_data_priority(self, tick: ForexTick) -> DataPriority:
        """
        Determine data priority based on tick characteristics | 根據跳動特徵確定數據優先級
        """
        # Critical for primary trading pairs | 主要交易對為關鍵優先級
        primary_pairs = self.config.get('data_ingestion', {}).get('primary_pairs', [])
        if tick.symbol in primary_pairs:
            return DataPriority.CRITICAL
        
        # High for secondary pairs | 次要貨幣對為高優先級
        secondary_pairs = self.config.get('data_ingestion', {}).get('secondary_pairs', [])
        if tick.symbol in secondary_pairs:
            return DataPriority.HIGH
        
        return DataPriority.MEDIUM
    
    def _update_source_health(self, source: DataSource, success: bool):
        """
        Update source health score | 更新數據源健康分數
        """
        current_score = self.data_router.source_health_scores.get(source, 50.0)
        
        # Simple exponential moving average for health score | 健康分數的簡單指數移動平均
        if success:
            new_score = current_score + (100.0 - current_score) * 0.1
        else:
            new_score = current_score * 0.9
        
        self.data_router.update_source_health(source, new_score)
        self.metrics.source_health_scores[source.value] = new_score
    
    def _start_buffer_workers(self):
        """
        Start buffer flush worker threads | 啟動緩衝區刷新工作線程
        """
        for priority, buffer in self.buffers.items():
            worker_thread = threading.Thread(
                target=self._buffer_worker,
                args=(priority, buffer),
                name=f"BufferWorker-{priority.value}",
                daemon=True
            )
            worker_thread.start()
            self.worker_threads.append(worker_thread)
            
        logger.info(f"Started {len(self.buffers)} buffer worker threads")
    
    def _buffer_worker(self, priority: DataPriority, buffer: IngestionBuffer):
        """
        Buffer worker thread for flushing data | 緩衝區工作線程用於刷新數據
        """
        logger.info(f"Buffer worker started for priority: {priority.value}")
        
        while not self.shutdown_event.is_set():
            try:
                if buffer.should_flush:
                    data = buffer.flush()
                    
                    if data:
                        # Process data | 處理數據
                        self._process_buffer_data(priority, data)
                        
                        # Update metrics | 更新指標
                        self.metrics.buffer_utilization = sum(
                            buf.utilization for buf in self.buffers.values()
                        ) / len(self.buffers)
                
                # Sleep based on priority | 根據優先級睡眠
                sleep_time = {
                    DataPriority.CRITICAL: 0.1,  # 100ms
                    DataPriority.HIGH: 0.5,      # 500ms
                    DataPriority.MEDIUM: 1.0,    # 1s
                    DataPriority.LOW: 5.0        # 5s
                }.get(priority, 1.0)
                
                self.shutdown_event.wait(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in buffer worker for {priority.value}: {e}")
                self.shutdown_event.wait(1.0)  # Wait 1 second on error
        
        logger.info(f"Buffer worker stopped for priority: {priority.value}")
    
    def _process_buffer_data(self, priority: DataPriority, data: List[ForexTick]):
        """
        Process buffered data | 處理緩衝的數據
        """
        try:
            # Store in database | 存儲到資料庫
            self._store_ticks_in_database(data)
            
            # Update Redis cache | 更新Redis緩存
            self._update_redis_cache(data)
            
            # Notify data processors | 通知數據處理器
            for processor in self.data_processors:
                try:
                    processor(data)
                except Exception as e:
                    logger.error(f"Error in data processor {processor.__name__}: {e}")
            
            logger.debug(f"Processed {len(data)} ticks for priority {priority.value}")
            
        except Exception as e:
            logger.error(f"Failed to process buffer data: {e}")
            self.metrics.error_count += len(data)
    
    def _store_ticks_in_database(self, ticks: List[ForexTick]):
        """
        Store ticks in database | 在資料庫中存儲跳動數據
        """
        try:
            with self.db_pool_manager.get_session() as session:
                # Group ticks by symbol for efficient storage | 按品種分組跳動數據以高效存儲
                ticks_by_symbol = {}
                for tick in ticks:
                    if tick.symbol not in ticks_by_symbol:
                        ticks_by_symbol[tick.symbol] = []
                    ticks_by_symbol[tick.symbol].append(tick)
                
                # Store each symbol's data | 存儲每個品種的數據
                for symbol, symbol_ticks in ticks_by_symbol.items():
                    self._store_symbol_ticks(session, symbol, symbol_ticks)
                    
        except Exception as e:
            logger.error(f"Failed to store ticks in database: {e}")
            raise
    
    def _store_symbol_ticks(self, session, symbol: str, ticks: List[ForexTick]):
        """
        Store ticks for a specific symbol | 存儲特定品種的跳動數據
        """
        # This would integrate with the existing database schema
        # For now, we'll use a simplified approach
        # 這將與現有的資料庫架構整合，現在我們使用簡化的方法
        
        table_name = f"realtime_ticks_{symbol.lower()}"
        
        # Create table if not exists (simplified) | 如果不存在則創建表（簡化）
        session.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                timestamp TIMESTAMP PRIMARY KEY,
                symbol VARCHAR(10) NOT NULL,
                bid DECIMAL(10, 5) NOT NULL,
                ask DECIMAL(10, 5) NOT NULL,
                source VARCHAR(20) NOT NULL,
                volume INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert ticks | 插入跳動數據
        for tick in ticks:
            session.execute(f"""
                INSERT INTO {table_name} 
                (timestamp, symbol, bid, ask, source, volume)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (timestamp) DO UPDATE SET
                bid = EXCLUDED.bid,
                ask = EXCLUDED.ask,
                source = EXCLUDED.source,
                volume = EXCLUDED.volume
            """, (
                tick.timestamp,
                tick.symbol,
                tick.bid,
                tick.ask,
                tick.source.value,
                tick.volume
            ))
    
    def _update_redis_cache(self, ticks: List[ForexTick]):
        """
        Update Redis cache with latest tick data | 使用最新跳動數據更新Redis緩存
        """
        if not self.redis_client:
            return
        
        try:
            pipe = self.redis_client.pipeline()
            
            for tick in ticks:
                # Update latest tick cache | 更新最新跳動緩存
                cache_key = f"forex:latest:{tick.symbol}:{tick.source.value}"
                pipe.hset(cache_key, mapping=tick.to_dict())
                pipe.expire(cache_key, 300)  # 5 minutes TTL
                
                # Add to time series if enabled | 如果啟用，添加到時間序列
                if self.config.get('caching', {}).get('redis', {}).get('use_streams', False):
                    stream_key = f"forex:stream:{tick.symbol}"
                    pipe.xadd(stream_key, tick.to_dict(), maxlen=10000)
            
            pipe.execute()
            
        except Exception as e:
            logger.error(f"Failed to update Redis cache: {e}")
    
    def _start_health_monitoring(self):
        """
        Start health monitoring thread | 啟動健康監控線程
        """
        def health_monitor():
            """Health monitoring worker | 健康監控工作器"""
            while not self.shutdown_event.is_set():
                try:
                    # Update metrics | 更新指標
                    self._update_ingestion_metrics()
                    
                    # Check source health | 檢查數據源健康狀況
                    self._check_source_health()
                    
                    # Sleep for monitoring interval | 按監控間隔睡眠
                    self.shutdown_event.wait(30)  # 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in health monitoring: {e}")
                    self.shutdown_event.wait(5)
        
        health_thread = threading.Thread(target=health_monitor, name="HealthMonitor", daemon=True)
        health_thread.start()
        self.worker_threads.append(health_thread)
    
    def _update_ingestion_metrics(self):
        """
        Update ingestion performance metrics | 更新攝取性能指標
        """
        # Calculate records per second | 計算每秒記錄數
        uptime = self.metrics.uptime_seconds
        if uptime > 0:
            self.metrics.records_per_second = self.metrics.total_records_ingested / uptime
    
    def _check_source_health(self):
        """
        Check and update source health scores | 檢查並更新數據源健康分數
        """
        feed_status = self.realtime_feed.get_connection_status()
        
        for source_name, source_status in feed_status.get('sources', {}).items():
            try:
                source = DataSource(source_name)
                
                # Calculate health score based on connection status | 根據連接狀態計算健康分數
                if source_status['is_healthy']:
                    health_score = 100.0 - (source_status['error_count'] * 5.0)
                else:
                    health_score = max(0.0, 50.0 - (source_status['error_count'] * 10.0))
                
                self.data_router.update_source_health(source, health_score)
                
            except ValueError:
                continue  # Skip unknown sources
    
    def _flush_all_buffers(self):
        """
        Flush all buffers during shutdown | 關閉期間刷新所有緩衝區
        """
        logger.info("Flushing all buffers...")
        
        for priority, buffer in self.buffers.items():
            try:
                data = buffer.flush()
                if data:
                    self._process_buffer_data(priority, data)
                    logger.info(f"Flushed {len(data)} ticks from {priority.value} buffer")
            except Exception as e:
                logger.error(f"Error flushing {priority.value} buffer: {e}")
    
    def get_ingestion_status(self) -> Dict[str, Any]:
        """
        Get current ingestion status and metrics | 獲取當前攝取狀態和指標
        """
        feed_status = self.realtime_feed.get_connection_status()
        
        return {
            'status': self.status.value,
            'uptime_seconds': self.metrics.uptime_seconds,
            'total_records_ingested': self.metrics.total_records_ingested,
            'records_per_second': self.metrics.records_per_second,
            'error_count': self.metrics.error_count,
            'success_rate': self.metrics.success_rate,
            'average_processing_time_ms': self.metrics.average_processing_time_ms,
            'buffer_utilization': self.metrics.buffer_utilization,
            'last_update_time': self.metrics.last_update_time.isoformat() if self.metrics.last_update_time else None,
            'feed_status': feed_status,
            'source_health_scores': self.metrics.source_health_scores,
            'buffer_stats': {
                priority.value: {
                    'size': len(buffer.data),
                    'utilization': buffer.utilization,
                    'total_flushes': buffer.total_flushes
                }
                for priority, buffer in self.buffers.items()
            }
        }


# Factory function | 工廠函數
def create_multi_source_ingestion(config_path: str = "config/data-sources.yaml") -> MultiSourceDataIngestion:
    """
    Create and configure multi-source data ingestion system
    創建並配置多源數據攝取系統
    """
    return MultiSourceDataIngestion(config_path)


if __name__ == "__main__":
    # Example usage | 使用示例
    def tick_processor(ticks: List[ForexTick]):
        """Example tick processor | 示例跳動處理器"""
        print(f"Processing batch of {len(ticks)} ticks")
        for tick in ticks[:3]:  # Show first 3 ticks
            print(f"  {tick.symbol}: {tick.bid}/{tick.ask} from {tick.source.value}")
    
    # Create and start ingestion | 創建並啟動攝取
    ingestion = create_multi_source_ingestion()
    ingestion.add_data_processor(tick_processor)
    
    try:
        ingestion.start([DataSource.YAHOO])
        
        # Monitor status | 監控狀態
        while True:
            time.sleep(30)
            status = ingestion.get_ingestion_status()
            print(f"Ingestion status: {status['status']}")
            print(f"Records ingested: {status['total_records_ingested']}")
            print(f"Records/sec: {status['records_per_second']:.2f}")
            print(f"Success rate: {status['success_rate']:.2%}")
            
    except KeyboardInterrupt:
        print("Stopping ingestion...")
        ingestion.stop()