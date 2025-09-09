# -*- coding: utf-8 -*-
"""
AIFX Real-time Data Pipeline - Database Integration
AIFX 實時數據管道 - 數據庫整合

This module handles integration with existing database and monitoring systems.
該模組處理與現有數據庫和監控系統的整合。

Author: AIFX Development Team
Created: 2025-01-14
Version: 1.0.0
"""

import asyncio
import logging
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, AsyncGenerator
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
import redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import yaml


# Prometheus metrics
FOREX_DATA_UPDATES = Counter('forex_data_updates_total', 'Total forex data updates', ['source', 'symbol'])
FOREX_DATA_LATENCY = Histogram('forex_data_latency_seconds', 'Forex data update latency', ['source'])
FOREX_CONNECTION_STATUS = Gauge('forex_connection_status', 'Forex connection status', ['source'])
DATABASE_OPERATIONS = Counter('database_operations_total', 'Database operations', ['operation', 'status'])
DATABASE_LATENCY = Histogram('database_latency_seconds', 'Database operation latency', ['operation'])


@dataclass
class ForexTick:
    """Forex tick data structure"""
    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    source: str
    spread: Optional[float] = None
    volume: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'symbol': self.symbol,
            'bid': self.bid,
            'ask': self.ask,
            'timestamp': self.timestamp,
            'source': self.source,
            'spread': self.spread,
            'volume': self.volume
        }


@dataclass
class DatabaseMetrics:
    """Database performance metrics"""
    total_inserts: int = 0
    total_queries: int = 0
    avg_insert_time_ms: float = 0.0
    avg_query_time_ms: float = 0.0
    connection_pool_size: int = 0
    active_connections: int = 0
    failed_operations: int = 0


class DatabaseManager:
    """Manages database connections and operations for forex data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database configuration
        db_config = config.get('database', {})
        self.host = db_config.get('host', 'localhost')
        self.port = db_config.get('port', 5432)
        self.database = db_config.get('database', 'aifx')
        self.username = db_config.get('username', 'aifx_user')
        self.password = db_config.get('password', 'password')
        
        # Connection pool settings
        self.min_connections = db_config.get('min_connections', 2)
        self.max_connections = db_config.get('max_connections', 20)
        
        # Initialize connection pool
        self.connection_pool = None
        self.metrics = DatabaseMetrics()
        
        # Performance settings
        self.batch_size = db_config.get('batch_size', 1000)
        self.batch_timeout = db_config.get('batch_timeout', 5.0)  # seconds
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize database connection pool and tables"""
        try:
            # Create connection pool
            self.connection_pool = ThreadedConnectionPool(
                minconn=self.min_connections,
                maxconn=self.max_connections,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password
            )
            
            # Create tables if they don't exist
            self._create_tables()
            
            self.logger.info("Database manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_tables(self) -> None:
        """Create necessary database tables"""
        connection = None
        try:
            connection = self.connection_pool.getconn()
            cursor = connection.cursor()
            
            # Create forex_ticks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS forex_ticks (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    bid DECIMAL(10, 5) NOT NULL,
                    ask DECIMAL(10, 5) NOT NULL,
                    spread DECIMAL(10, 5),
                    volume DECIMAL(15, 2),
                    source VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            
            # Create index for fast queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_forex_ticks_symbol_timestamp 
                ON forex_ticks(symbol, timestamp DESC);
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_forex_ticks_source_timestamp 
                ON forex_ticks(source, timestamp DESC);
            """)
            
            # Create forex_ohlc table for aggregated data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS forex_ohlc (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    open_price DECIMAL(10, 5) NOT NULL,
                    high_price DECIMAL(10, 5) NOT NULL,
                    low_price DECIMAL(10, 5) NOT NULL,
                    close_price DECIMAL(10, 5) NOT NULL,
                    volume DECIMAL(15, 2),
                    tick_count INTEGER,
                    source VARCHAR(50) NOT NULL,
                    period_start TIMESTAMP WITH TIME ZONE NOT NULL,
                    period_end TIMESTAMP WITH TIME ZONE NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    UNIQUE(symbol, timeframe, period_start, source)
                );
            """)
            
            # Create data quality monitoring table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_metrics (
                    id SERIAL PRIMARY KEY,
                    source VARCHAR(50) NOT NULL,
                    symbol VARCHAR(10) NOT NULL,
                    latency_ms DECIMAL(10, 2),
                    quality_score DECIMAL(5, 4),
                    anomaly_detected BOOLEAN DEFAULT FALSE,
                    anomaly_type VARCHAR(50),
                    anomaly_description TEXT,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );
            """)
            
            connection.commit()
            self.logger.info("Database tables created/verified successfully")
            
        except Exception as e:
            if connection:
                connection.rollback()
            self.logger.error(f"Failed to create database tables: {e}")
            raise
        finally:
            if connection:
                self.connection_pool.putconn(connection)
    
    @asynccontextmanager
    async def get_connection(self):
        """Async context manager for database connections"""
        connection = None
        try:
            connection = self.connection_pool.getconn()
            self.metrics.active_connections += 1
            yield connection
        except Exception as e:
            self.metrics.failed_operations += 1
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            if connection:
                self.connection_pool.putconn(connection)
                self.metrics.active_connections -= 1
    
    async def insert_tick(self, tick: ForexTick) -> bool:
        """Insert a single forex tick into database"""
        start_time = time.time()
        
        try:
            async with self.get_connection() as connection:
                cursor = connection.cursor()
                
                cursor.execute("""
                    INSERT INTO forex_ticks (symbol, bid, ask, spread, volume, source, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    tick.symbol,
                    tick.bid,
                    tick.ask,
                    tick.spread,
                    tick.volume,
                    tick.source,
                    tick.timestamp
                ))
                
                connection.commit()
                self.metrics.total_inserts += 1
                
                # Update metrics
                insert_time_ms = (time.time() - start_time) * 1000
                self.metrics.avg_insert_time_ms = (
                    (self.metrics.avg_insert_time_ms * (self.metrics.total_inserts - 1) + insert_time_ms)
                    / self.metrics.total_inserts
                )
                
                # Update Prometheus metrics
                FOREX_DATA_UPDATES.labels(source=tick.source, symbol=tick.symbol).inc()
                DATABASE_OPERATIONS.labels(operation='insert', status='success').inc()
                DATABASE_LATENCY.labels(operation='insert').observe(time.time() - start_time)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to insert tick: {e}")
            DATABASE_OPERATIONS.labels(operation='insert', status='error').inc()
            return False
    
    async def insert_ticks_batch(self, ticks: List[ForexTick]) -> int:
        """Insert multiple forex ticks in batch"""
        if not ticks:
            return 0
        
        start_time = time.time()
        inserted_count = 0
        
        try:
            async with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # Prepare batch data
                batch_data = []
                for tick in ticks:
                    batch_data.append((
                        tick.symbol,
                        tick.bid,
                        tick.ask,
                        tick.spread,
                        tick.volume,
                        tick.source,
                        tick.timestamp
                    ))
                
                # Execute batch insert
                cursor.executemany("""
                    INSERT INTO forex_ticks (symbol, bid, ask, spread, volume, source, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, batch_data)
                
                connection.commit()
                inserted_count = len(ticks)
                self.metrics.total_inserts += inserted_count
                
                # Update metrics
                insert_time_ms = (time.time() - start_time) * 1000
                avg_per_record = insert_time_ms / len(ticks)
                self.metrics.avg_insert_time_ms = (
                    (self.metrics.avg_insert_time_ms * (self.metrics.total_inserts - inserted_count) + 
                     avg_per_record * inserted_count) / self.metrics.total_inserts
                )
                
                # Update Prometheus metrics
                for tick in ticks:
                    FOREX_DATA_UPDATES.labels(source=tick.source, symbol=tick.symbol).inc()
                
                DATABASE_OPERATIONS.labels(operation='batch_insert', status='success').inc()
                DATABASE_LATENCY.labels(operation='batch_insert').observe(time.time() - start_time)
                
                self.logger.debug(f"Batch inserted {inserted_count} ticks in {insert_time_ms:.2f}ms")
                
        except Exception as e:
            self.logger.error(f"Failed to batch insert ticks: {e}")
            DATABASE_OPERATIONS.labels(operation='batch_insert', status='error').inc()
        
        return inserted_count
    
    async def get_latest_ticks(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get latest ticks for a symbol"""
        start_time = time.time()
        
        try:
            async with self.get_connection() as connection:
                cursor = connection.cursor()
                
                cursor.execute("""
                    SELECT symbol, bid, ask, spread, volume, source, timestamp
                    FROM forex_ticks
                    WHERE symbol = %s
                    ORDER BY timestamp DESC
                    LIMIT %s
                """, (symbol, limit))
                
                results = cursor.fetchall()
                
                ticks = []
                for row in results:
                    ticks.append({
                        'symbol': row[0],
                        'bid': float(row[1]),
                        'ask': float(row[2]),
                        'spread': float(row[3]) if row[3] else None,
                        'volume': float(row[4]) if row[4] else None,
                        'source': row[5],
                        'timestamp': row[6]
                    })
                
                self.metrics.total_queries += 1
                query_time_ms = (time.time() - start_time) * 1000
                self.metrics.avg_query_time_ms = (
                    (self.metrics.avg_query_time_ms * (self.metrics.total_queries - 1) + query_time_ms)
                    / self.metrics.total_queries
                )
                
                DATABASE_OPERATIONS.labels(operation='query', status='success').inc()
                DATABASE_LATENCY.labels(operation='query').observe(time.time() - start_time)
                
                return ticks
                
        except Exception as e:
            self.logger.error(f"Failed to query latest ticks: {e}")
            DATABASE_OPERATIONS.labels(operation='query', status='error').inc()
            return []
    
    async def insert_quality_metric(self, source: str, symbol: str, 
                                   latency_ms: float, quality_score: float,
                                   anomaly_detected: bool = False,
                                   anomaly_type: str = None,
                                   anomaly_description: str = None) -> bool:
        """Insert data quality metrics"""
        try:
            async with self.get_connection() as connection:
                cursor = connection.cursor()
                
                cursor.execute("""
                    INSERT INTO data_quality_metrics 
                    (source, symbol, latency_ms, quality_score, anomaly_detected, 
                     anomaly_type, anomaly_description, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    source, symbol, latency_ms, quality_score,
                    anomaly_detected, anomaly_type, anomaly_description,
                    datetime.now()
                ))
                
                connection.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to insert quality metric: {e}")
            return False
    
    def get_metrics(self) -> DatabaseMetrics:
        """Get database performance metrics"""
        # Update connection pool metrics
        if self.connection_pool:
            self.metrics.connection_pool_size = self.max_connections
        
        return self.metrics
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """Clean up old data beyond retention period"""
        try:
            with self.connection_pool.getconn() as connection:
                cursor = connection.cursor()
                
                cutoff_date = datetime.now() - timedelta(days=days_to_keep)
                
                # Delete old tick data
                cursor.execute("""
                    DELETE FROM forex_ticks 
                    WHERE timestamp < %s
                """, (cutoff_date,))
                
                deleted_ticks = cursor.rowcount
                
                # Delete old quality metrics
                cursor.execute("""
                    DELETE FROM data_quality_metrics 
                    WHERE timestamp < %s
                """, (cutoff_date,))
                
                deleted_metrics = cursor.rowcount
                
                connection.commit()
                
                total_deleted = deleted_ticks + deleted_metrics
                self.logger.info(f"Cleaned up {total_deleted} old records ({deleted_ticks} ticks, {deleted_metrics} metrics)")
                
                return total_deleted
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
            return 0
    
    def shutdown(self) -> None:
        """Shutdown database manager"""
        if self.connection_pool:
            self.connection_pool.closeall()
        self.logger.info("Database manager shutdown complete")


class RedisManager:
    """Manages Redis connections for caching and real-time data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Redis configuration
        redis_config = config.get('caching', {}).get('redis', {})
        self.host = redis_config.get('host', 'localhost')
        self.port = redis_config.get('port', 6379)
        self.db = redis_config.get('db', 1)
        self.password = redis_config.get('password')
        
        # Connection settings
        self.max_connections = redis_config.get('max_connections', 20)
        self.key_prefix = redis_config.get('key_prefix', 'forex:realtime')
        self.default_ttl = redis_config.get('default_ttl', 300)
        
        # Stream settings
        self.use_streams = redis_config.get('use_streams', True)
        self.stream_maxlen = redis_config.get('stream_maxlen', 10000)
        
        # Initialize Redis connection
        self.redis_client = None
        self._initialize_redis()
    
    def _initialize_redis(self) -> None:
        """Initialize Redis connection pool"""
        try:
            self.redis_client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            
            self.logger.info("Redis manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def cache_tick(self, tick: ForexTick, ttl: Optional[int] = None) -> bool:
        """Cache forex tick data"""
        try:
            key = f"{self.key_prefix}:tick:{tick.symbol}:latest"
            value = json.dumps(tick.to_dict(), default=str)
            
            ttl = ttl or self.default_ttl
            
            self.redis_client.setex(key, ttl, value)
            
            # Also add to stream if enabled
            if self.use_streams:
                stream_key = f"{self.key_prefix}:stream:{tick.symbol}"
                self.redis_client.xadd(
                    stream_key,
                    tick.to_dict(),
                    maxlen=self.stream_maxlen,
                    approximate=True
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cache tick: {e}")
            return False
    
    async def get_cached_tick(self, symbol: str) -> Optional[ForexTick]:
        """Get cached forex tick"""
        try:
            key = f"{self.key_prefix}:tick:{symbol}:latest"
            cached_data = self.redis_client.get(key)
            
            if cached_data:
                data = json.loads(cached_data)
                return ForexTick(
                    symbol=data['symbol'],
                    bid=data['bid'],
                    ask=data['ask'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    source=data['source'],
                    spread=data.get('spread'),
                    volume=data.get('volume')
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get cached tick: {e}")
            return None
    
    async def get_stream_data(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent data from Redis stream"""
        try:
            stream_key = f"{self.key_prefix}:stream:{symbol}"
            
            # Read from stream (latest entries first)
            entries = self.redis_client.xrevrange(stream_key, count=count)
            
            stream_data = []
            for entry_id, fields in entries:
                stream_data.append({
                    'id': entry_id,
                    'data': fields
                })
            
            return stream_data
            
        except Exception as e:
            self.logger.error(f"Failed to get stream data: {e}")
            return []
    
    def shutdown(self) -> None:
        """Shutdown Redis manager"""
        if self.redis_client:
            self.redis_client.close()
        self.logger.info("Redis manager shutdown complete")


class MonitoringIntegration:
    """Integrates with monitoring systems (Prometheus, alerts)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Monitoring configuration
        monitoring_config = config.get('monitoring', {})
        self.prometheus_enabled = monitoring_config.get('prometheus', {}).get('enabled', True)
        self.prometheus_port = monitoring_config.get('prometheus', {}).get('port', 8002)
        
        # Initialize Prometheus metrics server
        if self.prometheus_enabled:
            self._start_prometheus_server()
    
    def _start_prometheus_server(self) -> None:
        """Start Prometheus metrics server"""
        try:
            start_http_server(self.prometheus_port)
            self.logger.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        except Exception as e:
            self.logger.error(f"Failed to start Prometheus server: {e}")
    
    def update_connection_status(self, source: str, status: int) -> None:
        """Update connection status metric (1=connected, 0=disconnected)"""
        FOREX_CONNECTION_STATUS.labels(source=source).set(status)
    
    def record_data_latency(self, source: str, latency_seconds: float) -> None:
        """Record data latency metric"""
        FOREX_DATA_LATENCY.labels(source=source).observe(latency_seconds)


class DatabaseIntegrationManager:
    """Main manager for database and monitoring integration"""
    
    def __init__(self, config_path: str = "config/data-sources.yaml"):
        """Initialize database integration manager"""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.db_manager = DatabaseManager(self.config)
        self.redis_manager = RedisManager(self.config)
        self.monitoring = MonitoringIntegration(self.config)
        
        self.logger.info("Database integration manager initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    async def process_tick(self, tick: ForexTick) -> bool:
        """Process a forex tick through all systems"""
        try:
            # Cache in Redis
            await self.redis_manager.cache_tick(tick)
            
            # Store in database
            await self.db_manager.insert_tick(tick)
            
            # Update monitoring metrics
            self.monitoring.update_connection_status(tick.source, 1)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process tick: {e}")
            return False
    
    async def process_tick_batch(self, ticks: List[ForexTick]) -> int:
        """Process batch of forex ticks"""
        if not ticks:
            return 0
        
        try:
            # Cache in Redis (process in parallel)
            cache_tasks = [self.redis_manager.cache_tick(tick) for tick in ticks]
            await asyncio.gather(*cache_tasks, return_exceptions=True)
            
            # Batch insert to database
            inserted_count = await self.db_manager.insert_ticks_batch(ticks)
            
            # Update monitoring metrics
            for tick in ticks:
                self.monitoring.update_connection_status(tick.source, 1)
            
            return inserted_count
            
        except Exception as e:
            self.logger.error(f"Failed to process tick batch: {e}")
            return 0
    
    async def get_latest_data(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get latest data for symbol (from cache first, then database)"""
        try:
            # Try cache first
            cached_tick = await self.redis_manager.get_cached_tick(symbol)
            if cached_tick and limit == 1:
                return [cached_tick.to_dict()]
            
            # Get from database
            return await self.db_manager.get_latest_ticks(symbol, limit)
            
        except Exception as e:
            self.logger.error(f"Failed to get latest data: {e}")
            return []
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        return {
            'database': asdict(self.db_manager.get_metrics()),
            'redis': {
                'host': self.redis_manager.host,
                'port': self.redis_manager.port,
                'connected': True  # Simplified for now
            },
            'monitoring': {
                'prometheus_enabled': self.monitoring.prometheus_enabled,
                'prometheus_port': self.monitoring.prometheus_port
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown all components"""
        self.db_manager.shutdown()
        self.redis_manager.shutdown()
        self.logger.info("Database integration manager shutdown complete")


# Factory function
def create_database_integration_manager(config_path: str = "config/data-sources.yaml") -> DatabaseIntegrationManager:
    """Create database integration manager"""
    return DatabaseIntegrationManager(config_path)


if __name__ == "__main__":
    # Example usage and testing
    import asyncio
    
    async def main():
        try:
            # Create manager
            manager = create_database_integration_manager()
            
            # Create sample tick
            tick = ForexTick(
                symbol="EURUSD",
                bid=1.0850,
                ask=1.0852,
                timestamp=datetime.now(),
                source="yahoo",
                spread=0.0002
            )
            
            # Process tick
            success = await manager.process_tick(tick)
            print(f"Tick processed: {success}")
            
            # Get latest data
            latest_data = await manager.get_latest_data("EURUSD", 5)
            print(f"Latest data points: {len(latest_data)}")
            
            # Get system metrics
            metrics = manager.get_system_metrics()
            print(f"System metrics: {json.dumps(metrics, indent=2)}")
            
            manager.shutdown()
            print("Database integration test completed successfully")
            
        except Exception as e:
            print(f"Error testing database integration: {e}")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run test
    asyncio.run(main())