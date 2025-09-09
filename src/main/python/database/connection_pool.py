"""
AIFX Database Connection Pool Manager | AIFX 資料庫連接池管理器
High-performance database connection pooling for production trading operations
用於生產交易操作的高性能資料庫連接池

Phase 4.1.4 Database Optimization Component
第四階段 4.1.4 資料庫優化組件
"""

import os
import time
import logging
import threading
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Lock, RLock

from sqlalchemy import create_engine, text, event, pool
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, DisconnectionError
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool, NullPool
import psycopg2
import redis
from redis.connection import ConnectionPool
from redis.sentinel import Sentinel
import pymongo
from pymongo import MongoClient

# Logger setup | 日誌設置
logger = logging.getLogger(__name__)


@dataclass
class ConnectionPoolConfig:
    """
    Database connection pool configuration | 資料庫連接池配置
    """
    # Basic pool settings | 基本池設置
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    
    # Performance settings | 性能設置
    echo: bool = False
    echo_pool: bool = False
    isolation_level: str = "READ_COMMITTED"
    
    # Connection settings | 連接設置
    connect_timeout: int = 10
    query_timeout: int = 30
    statement_timeout: int = 60
    
    # Health check settings | 健康檢查設置
    health_check_interval: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Monitoring settings | 監控設置
    enable_monitoring: bool = True
    log_slow_queries: bool = True
    slow_query_threshold: float = 1.0


@dataclass
class PoolStats:
    """
    Connection pool statistics | 連接池統計信息
    """
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    checked_out: int = 0
    overflow: int = 0
    failed_connections: int = 0
    
    # Performance metrics | 性能指標
    avg_connection_time: float = 0.0
    max_connection_time: float = 0.0
    query_count: int = 0
    slow_query_count: int = 0
    error_count: int = 0
    
    # Timestamps | 時間戳
    last_health_check: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)


class DatabaseConnectionPool:
    """
    Advanced database connection pool manager | 高級資料庫連接池管理器
    """
    
    def __init__(self, config: ConnectionPoolConfig):
        """
        Initialize connection pool | 初始化連接池
        
        Args:
            config: Pool configuration | 池配置
        """
        self.config = config
        self.stats = PoolStats()
        self._lock = RLock()
        self._engines: Dict[str, Engine] = {}
        self._session_factories: Dict[str, sessionmaker] = {}
        self._health_check_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        
        logger.info("Database connection pool initialized")
    
    def create_postgresql_engine(self, connection_string: str, name: str = "postgres") -> Engine:
        """
        Create PostgreSQL engine with optimized settings | 創建優化設置的PostgreSQL引擎
        """
        connect_args = {
            "connect_timeout": self.config.connect_timeout,
            "application_name": f"AIFX-{name}",
            "options": f"-c statement_timeout={self.config.statement_timeout * 1000}ms"
        }
        
        # SSL settings for production | 生產環境SSL設置
        if "sslmode" not in connection_string:
            if os.getenv("POSTGRES_SSL_MODE"):
                connect_args["sslmode"] = os.getenv("POSTGRES_SSL_MODE", "prefer")
        
        engine = create_engine(
            connection_string,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            pool_pre_ping=self.config.pool_pre_ping,
            echo=self.config.echo,
            echo_pool=self.config.echo_pool,
            isolation_level=self.config.isolation_level,
            connect_args=connect_args,
            execution_options={
                "autocommit": False,
                "isolation_level": self.config.isolation_level
            }
        )
        
        # Add event listeners for monitoring | 添加監控事件監聽器
        if self.config.enable_monitoring:
            self._add_monitoring_events(engine, name)
        
        self._engines[name] = engine
        self._session_factories[name] = sessionmaker(bind=engine, autoflush=False)
        
        logger.info(f"PostgreSQL engine created: {name}")
        return engine
    
    def create_mongodb_connection(self, connection_string: str, name: str = "mongo") -> MongoClient:
        """
        Create MongoDB connection with connection pooling | 創建帶連接池的MongoDB連接
        """
        client = MongoClient(
            connection_string,
            maxPoolSize=self.config.pool_size,
            minPoolSize=5,
            maxIdleTimeMS=self.config.pool_recycle * 1000,
            connectTimeoutMS=self.config.connect_timeout * 1000,
            serverSelectionTimeoutMS=self.config.query_timeout * 1000,
            waitQueueTimeoutMS=self.config.pool_timeout * 1000,
            retryWrites=True,
            retryReads=True,
            appname=f"AIFX-{name}"
        )
        
        # Test connection | 測試連接
        try:
            client.admin.command('ping')
            logger.info(f"MongoDB connection created: {name}")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
        
        return client
    
    def _add_monitoring_events(self, engine: Engine, name: str):
        """
        Add SQLAlchemy event listeners for monitoring | 添加SQLAlchemy監控事件監聽器
        """
        
        @event.listens_for(engine, "connect")
        def on_connect(dbapi_conn, connection_record):
            """Connection established event | 連接建立事件"""
            with self._lock:
                self.stats.total_connections += 1
                self.stats.active_connections += 1
        
        @event.listens_for(engine, "checkout")
        def on_checkout(dbapi_conn, connection_record, connection_proxy):
            """Connection checked out event | 連接檢出事件"""
            with self._lock:
                self.stats.checked_out += 1
                connection_record.checkout_time = time.time()
        
        @event.listens_for(engine, "checkin")
        def on_checkin(dbapi_conn, connection_record):
            """Connection checked in event | 連接檢入事件"""
            with self._lock:
                self.stats.checked_out = max(0, self.stats.checked_out - 1)
                
                # Calculate connection time | 計算連接時間
                if hasattr(connection_record, 'checkout_time'):
                    connection_time = time.time() - connection_record.checkout_time
                    self.stats.avg_connection_time = (
                        (self.stats.avg_connection_time * self.stats.query_count + connection_time) /
                        (self.stats.query_count + 1)
                    )
                    self.stats.max_connection_time = max(
                        self.stats.max_connection_time, connection_time
                    )
        
        @event.listens_for(engine, "before_cursor_execute")
        def on_before_execute(conn, cursor, statement, parameters, context, executemany):
            """Before query execution event | 查詢執行前事件"""
            context.query_start_time = time.time()
        
        @event.listens_for(engine, "after_cursor_execute")
        def on_after_execute(conn, cursor, statement, parameters, context, executemany):
            """After query execution event | 查詢執行後事件"""
            with self._lock:
                self.stats.query_count += 1
                
                # Check for slow queries | 檢查慢查詢
                if hasattr(context, 'query_start_time'):
                    query_time = time.time() - context.query_start_time
                    if query_time > self.config.slow_query_threshold:
                        self.stats.slow_query_count += 1
                        if self.config.log_slow_queries:
                            logger.warning(f"Slow query detected ({query_time:.3f}s): {statement[:100]}...")
    
    @contextmanager
    def get_session(self, name: str = "postgres"):
        """
        Get database session with automatic cleanup | 獲取帶自動清理的資料庫會話
        """
        if name not in self._session_factories:
            raise ValueError(f"No session factory found for: {name}")
        
        session = self._session_factories[name]()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            with self._lock:
                self.stats.error_count += 1
            logger.error(f"Database session error ({name}): {e}")
            raise
        finally:
            session.close()
    
    @contextmanager
    def get_connection(self, name: str = "postgres"):
        """
        Get raw database connection | 獲取原始資料庫連接
        """
        if name not in self._engines:
            raise ValueError(f"No engine found for: {name}")
        
        connection = self._engines[name].connect()
        try:
            yield connection
            connection.commit()
        except Exception as e:
            connection.rollback()
            with self._lock:
                self.stats.error_count += 1
            logger.error(f"Database connection error ({name}): {e}")
            raise
        finally:
            connection.close()
    
    def execute_query(self, query: str, params: Optional[Dict] = None, 
                     name: str = "postgres") -> List[Tuple]:
        """
        Execute query and return results | 執行查詢並返回結果
        """
        with self.get_connection(name) as conn:
            result = conn.execute(text(query), params or {})
            return result.fetchall()
    
    def health_check(self, name: str = "postgres") -> Dict[str, Any]:
        """
        Perform health check on connection pool | 執行連接池健康檢查
        """
        health_info = {
            "name": name,
            "status": "unknown",
            "timestamp": datetime.now(),
            "response_time": None,
            "error": None
        }
        
        try:
            start_time = time.time()
            
            if name in self._engines:
                # Test SQL connection | 測試SQL連接
                with self.get_connection(name) as conn:
                    conn.execute(text("SELECT 1"))
                health_info["status"] = "healthy"
            else:
                health_info["status"] = "engine_not_found"
                health_info["error"] = f"No engine found for: {name}"
            
            health_info["response_time"] = time.time() - start_time
            
        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["error"] = str(e)
            logger.error(f"Health check failed for {name}: {e}")
        
        self.stats.last_health_check = health_info["timestamp"]
        return health_info
    
    def get_pool_status(self, name: str = "postgres") -> Dict[str, Any]:
        """
        Get detailed pool status information | 獲取詳細的池狀態信息
        """
        if name not in self._engines:
            return {"error": f"No engine found for: {name}"}
        
        engine = self._engines[name]
        pool = engine.pool
        
        return {
            "name": name,
            "pool_size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "checked_in": pool.checkedin(),
            "invalid": pool.invalid(),
            "config": {
                "pool_size": self.config.pool_size,
                "max_overflow": self.config.max_overflow,
                "pool_timeout": self.config.pool_timeout,
                "pool_recycle": self.config.pool_recycle
            },
            "stats": self.stats.__dict__
        }
    
    def start_monitoring(self):
        """
        Start background health check monitoring | 啟動後台健康檢查監控
        """
        if self._health_check_thread and self._health_check_thread.is_alive():
            logger.warning("Monitoring already started")
            return
        
        def health_check_loop():
            """Background health check loop | 後台健康檢查循環"""
            while not self._shutdown_event.is_set():
                try:
                    for name in self._engines:
                        self.health_check(name)
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    logger.error(f"Health check monitoring error: {e}")
                    time.sleep(5)  # Brief pause on error
        
        self._health_check_thread = threading.Thread(target=health_check_loop, daemon=True)
        self._health_check_thread.start()
        logger.info("Database health monitoring started")
    
    def stop_monitoring(self):
        """
        Stop background monitoring | 停止後台監控
        """
        self._shutdown_event.set()
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)
        logger.info("Database monitoring stopped")
    
    def dispose_all(self):
        """
        Dispose all engines and connections | 釋放所有引擎和連接
        """
        self.stop_monitoring()
        
        for name, engine in self._engines.items():
            try:
                engine.dispose()
                logger.info(f"Engine disposed: {name}")
            except Exception as e:
                logger.error(f"Error disposing engine {name}: {e}")
        
        self._engines.clear()
        self._session_factories.clear()


class RedisConnectionPool:
    """
    Redis connection pool manager | Redis 連接池管理器
    """
    
    def __init__(self, config: ConnectionPoolConfig):
        """
        Initialize Redis connection pool | 初始化Redis連接池
        """
        self.config = config
        self.pools: Dict[str, ConnectionPool] = {}
        self.clients: Dict[str, redis.Redis] = {}
        
        logger.info("Redis connection pool initialized")
    
    def create_redis_pool(self, connection_string: str, name: str = "redis") -> redis.Redis:
        """
        Create Redis connection pool | 創建Redis連接池
        """
        # Parse connection string | 解析連接字符串
        if connection_string.startswith("redis://"):
            pool = ConnectionPool.from_url(
                connection_string,
                max_connections=self.config.pool_size,
                socket_timeout=self.config.connect_timeout,
                socket_connect_timeout=self.config.connect_timeout,
                socket_keepalive=True,
                health_check_interval=self.config.health_check_interval,
                retry_on_timeout=True
            )
        else:
            # Parse host, port, db from string | 從字符串解析主機、端口、數據庫
            parts = connection_string.split(":")
            host = parts[0] if len(parts) > 0 else "localhost"
            port = int(parts[1]) if len(parts) > 1 else 6379
            db = int(parts[2]) if len(parts) > 2 else 0
            
            pool = ConnectionPool(
                host=host,
                port=port,
                db=db,
                max_connections=self.config.pool_size,
                socket_timeout=self.config.connect_timeout,
                socket_connect_timeout=self.config.connect_timeout,
                socket_keepalive=True,
                health_check_interval=self.config.health_check_interval,
                retry_on_timeout=True
            )
        
        client = redis.Redis(connection_pool=pool, decode_responses=True)
        
        # Test connection | 測試連接
        try:
            client.ping()
            logger.info(f"Redis connection pool created: {name}")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
        
        self.pools[name] = pool
        self.clients[name] = client
        return client
    
    def get_client(self, name: str = "redis") -> redis.Redis:
        """
        Get Redis client | 獲取Redis客戶端
        """
        if name not in self.clients:
            raise ValueError(f"No Redis client found for: {name}")
        return self.clients[name]
    
    def health_check(self, name: str = "redis") -> Dict[str, Any]:
        """
        Redis health check | Redis健康檢查
        """
        health_info = {
            "name": name,
            "status": "unknown",
            "timestamp": datetime.now(),
            "response_time": None,
            "error": None,
            "info": {}
        }
        
        try:
            start_time = time.time()
            client = self.get_client(name)
            
            # Test ping | 測試ping
            client.ping()
            
            # Get Redis info | 獲取Redis信息
            info = client.info()
            health_info["info"] = {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "N/A"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0)
            }
            
            health_info["status"] = "healthy"
            health_info["response_time"] = time.time() - start_time
            
        except Exception as e:
            health_info["status"] = "unhealthy"
            health_info["error"] = str(e)
            logger.error(f"Redis health check failed for {name}: {e}")
        
        return health_info


# Global connection pool manager | 全局連接池管理器
_pool_manager: Optional[DatabaseConnectionPool] = None
_redis_pool_manager: Optional[RedisConnectionPool] = None


def get_pool_manager() -> DatabaseConnectionPool:
    """
    Get or create global pool manager | 獲取或創建全局池管理器
    """
    global _pool_manager
    if _pool_manager is None:
        config = ConnectionPoolConfig()
        _pool_manager = DatabaseConnectionPool(config)
    return _pool_manager


def get_redis_pool_manager() -> RedisConnectionPool:
    """
    Get or create global Redis pool manager | 獲取或創建全局Redis池管理器
    """
    global _redis_pool_manager
    if _redis_pool_manager is None:
        config = ConnectionPoolConfig()
        _redis_pool_manager = RedisConnectionPool(config)
    return _redis_pool_manager


def initialize_production_pools():
    """
    Initialize production database pools | 初始化生產資料庫池
    """
    logger.info("Initializing production database pools...")
    
    pool_manager = get_pool_manager()
    redis_manager = get_redis_pool_manager()
    
    # PostgreSQL setup | PostgreSQL設置
    postgres_conn = (
        f"postgresql://{os.getenv('POSTGRES_USER', 'aifx')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'password')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'aifx_production')}"
    )
    
    pool_manager.create_postgresql_engine(postgres_conn, "postgres_primary")
    
    # Redis setup | Redis設置
    redis_conn = (
        f"redis://:{os.getenv('REDIS_PASSWORD', '')}@"
        f"{os.getenv('REDIS_HOST', 'localhost')}:"
        f"{os.getenv('REDIS_PORT', '6379')}/0"
    )
    
    redis_manager.create_redis_pool(redis_conn, "redis_primary")
    
    # Start monitoring | 啟動監控
    pool_manager.start_monitoring()
    
    logger.info("Production database pools initialized successfully")
    
    return pool_manager, redis_manager