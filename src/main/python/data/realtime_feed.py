"""
AIFX Real-time Forex Data Feed | AIFX 實時外匯數據源
WebSocket-based real-time forex data streaming with multi-source support
基於WebSocket的實時外匯數據流，支持多數據源

Phase 4.2 Real-time Data Pipeline Component
第四階段 4.2 實時數據管道組件
"""

import asyncio
import json
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import websocket
import requests
from contextlib import asynccontextmanager

import redis
import yaml

# Logger setup | 日誌設置
logger = logging.getLogger(__name__)


class ConnectionStatus(Enum):
    """Connection status enumeration | 連接狀態枚舉"""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING" 
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    ERROR = "ERROR"


class DataSource(Enum):
    """Data source types | 數據源類型"""
    YAHOO = "yahoo"
    OANDA = "oanda"
    FXCM = "fxcm"


@dataclass
class ForexTick:
    """
    Forex tick data structure | 外匯跳動數據結構
    """
    symbol: str
    bid: float
    ask: float
    timestamp: datetime
    source: DataSource
    volume: Optional[int] = None
    
    # Derived properties | 衍生屬性
    @property
    def mid_price(self) -> float:
        """Calculate mid price | 計算中間價"""
        return (self.bid + self.ask) / 2
    
    @property
    def spread(self) -> float:
        """Calculate spread in pips | 計算點差（以點為單位）"""
        return self.ask - self.bid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization | 轉換為字典用於序列化"""
        return {
            "symbol": self.symbol,
            "bid": self.bid,
            "ask": self.ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source.value,
            "volume": self.volume
        }


@dataclass
class ConnectionMetrics:
    """
    Connection performance metrics | 連接性能指標
    """
    source: DataSource
    status: ConnectionStatus
    connected_at: Optional[datetime] = None
    disconnected_at: Optional[datetime] = None
    reconnect_count: int = 0
    last_data_received: Optional[datetime] = None
    total_messages: int = 0
    error_count: int = 0
    average_latency_ms: float = 0.0
    
    @property
    def uptime_seconds(self) -> float:
        """Calculate connection uptime | 計算連接正常運行時間"""
        if self.connected_at and self.status == ConnectionStatus.CONNECTED:
            return (datetime.now() - self.connected_at).total_seconds()
        return 0.0
    
    @property
    def is_healthy(self) -> bool:
        """Check if connection is healthy | 檢查連接是否健康"""
        if self.status != ConnectionStatus.CONNECTED:
            return False
        
        if self.last_data_received:
            seconds_since_data = (datetime.now() - self.last_data_received).total_seconds()
            return seconds_since_data < 60  # Data within last minute
        
        return False


class WebSocketManager:
    """
    WebSocket connection manager with automatic reconnection
    WebSocket連接管理器，具有自動重連功能
    """
    
    def __init__(self, source: DataSource, config: Dict[str, Any], 
                 data_handler: Callable[[ForexTick], None]):
        """
        Initialize WebSocket manager | 初始化WebSocket管理器
        
        Args:
            source: Data source type | 數據源類型
            config: Source configuration | 源配置
            data_handler: Callback function for received data | 接收數據的回調函數
        """
        self.source = source
        self.config = config
        self.data_handler = data_handler
        
        # Connection state | 連接狀態
        self.ws: Optional[websocket.WebSocketApp] = None
        self.metrics = ConnectionMetrics(source=source, status=ConnectionStatus.DISCONNECTED)
        
        # Reconnection settings | 重連設置
        self.max_reconnect_attempts = config.get('websocket', {}).get('max_reconnect_attempts', 10)
        self.reconnect_interval = config.get('websocket', {}).get('reconnect_interval', 5)
        self.heartbeat_interval = config.get('websocket', {}).get('heartbeat_interval', 30)
        
        # Threading | 線程控制
        self.reconnect_thread: Optional[threading.Thread] = None
        self.heartbeat_thread: Optional[threading.Thread] = None
        self.should_reconnect = True
        self.is_running = False
        
        logger.info(f"WebSocket manager initialized for {source.value}")
    
    def connect(self) -> bool:
        """
        Establish WebSocket connection | 建立WebSocket連接
        """
        try:
            self.metrics.status = ConnectionStatus.CONNECTING
            url = self._get_websocket_url()
            
            self.ws = websocket.WebSocketApp(
                url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start connection in background thread | 在後台線程中啟動連接
            self.is_running = True
            self.ws.run_forever()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket for {self.source.value}: {e}")
            self.metrics.status = ConnectionStatus.ERROR
            self.metrics.error_count += 1
            return False
    
    def disconnect(self):
        """
        Disconnect WebSocket | 斷開WebSocket連接
        """
        self.should_reconnect = False
        self.is_running = False
        
        if self.ws:
            self.ws.close()
        
        if self.reconnect_thread:
            self.reconnect_thread.join(timeout=5)
        
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)
        
        self.metrics.status = ConnectionStatus.DISCONNECTED
        self.metrics.disconnected_at = datetime.now()
        
        logger.info(f"WebSocket disconnected for {self.source.value}")
    
    def _get_websocket_url(self) -> str:
        """
        Get WebSocket URL based on source | 根據數據源獲取WebSocket URL
        """
        if self.source == DataSource.YAHOO:
            return self.config['websocket']['url']
        elif self.source == DataSource.OANDA:
            # OANDA uses HTTP streaming, not WebSocket
            return f"{self.config['streaming']['base_url']}/pricing/stream"
        else:
            raise ValueError(f"Unsupported source for WebSocket: {self.source.value}")
    
    def _on_open(self, ws):
        """
        WebSocket connection opened callback | WebSocket連接開啟回調
        """
        self.metrics.status = ConnectionStatus.CONNECTED
        self.metrics.connected_at = datetime.now()
        self.metrics.reconnect_count = 0
        
        logger.info(f"WebSocket connected for {self.source.value}")
        
        # Subscribe to forex pairs | 訂閱外匯對
        self._subscribe_to_symbols()
        
        # Start heartbeat thread | 啟動心跳線程
        self._start_heartbeat()
    
    def _on_message(self, ws, message):
        """
        WebSocket message received callback | WebSocket消息接收回調
        """
        try:
            self.metrics.last_data_received = datetime.now()
            self.metrics.total_messages += 1
            
            # Parse message based on source | 根據數據源解析消息
            tick_data = self._parse_message(message)
            
            if tick_data:
                self.data_handler(tick_data)
            
        except Exception as e:
            logger.error(f"Error processing message from {self.source.value}: {e}")
            self.metrics.error_count += 1
    
    def _on_error(self, ws, error):
        """
        WebSocket error callback | WebSocket錯誤回調
        """
        logger.error(f"WebSocket error for {self.source.value}: {error}")
        self.metrics.status = ConnectionStatus.ERROR
        self.metrics.error_count += 1
    
    def _on_close(self, ws, close_status_code, close_msg):
        """
        WebSocket connection closed callback | WebSocket連接關閉回調
        """
        self.metrics.status = ConnectionStatus.DISCONNECTED
        self.metrics.disconnected_at = datetime.now()
        
        logger.warning(f"WebSocket closed for {self.source.value}: {close_status_code} - {close_msg}")
        
        # Attempt reconnection if enabled | 如果啟用，嘗試重連
        if self.should_reconnect and self.is_running:
            self._start_reconnection()
    
    def _subscribe_to_symbols(self):
        """
        Subscribe to forex symbols | 訂閱外匯品種
        """
        if self.source == DataSource.YAHOO:
            # Yahoo Finance subscription format | Yahoo Finance訂閱格式
            symbols = list(self.config['symbols'].values())
            subscription_msg = {
                "subscribe": symbols
            }
            
            if self.ws and self.ws.sock:
                self.ws.send(json.dumps(subscription_msg))
                logger.info(f"Subscribed to {len(symbols)} forex pairs on Yahoo Finance")
    
    def _parse_message(self, message: str) -> Optional[ForexTick]:
        """
        Parse WebSocket message to ForexTick | 解析WebSocket消息為ForexTick
        """
        try:
            if self.source == DataSource.YAHOO:
                return self._parse_yahoo_message(message)
            elif self.source == DataSource.OANDA:
                return self._parse_oanda_message(message)
            
        except Exception as e:
            logger.error(f"Failed to parse message from {self.source.value}: {e}")
            
        return None
    
    def _parse_yahoo_message(self, message: str) -> Optional[ForexTick]:
        """
        Parse Yahoo Finance WebSocket message | 解析Yahoo Finance WebSocket消息
        """
        try:
            data = json.loads(message)
            
            # Yahoo Finance message structure | Yahoo Finance消息結構
            if 'id' in data and 'price' in data:
                symbol = data.get('id', '').replace('=X', '')
                price = data.get('price')
                timestamp = datetime.fromtimestamp(data.get('time', time.time()))
                
                # For Yahoo Finance, we only get last price, estimate bid/ask | 
                # Yahoo Finance只提供最新價格，估算買賣價
                spread = 0.0001  # Assume 1 pip spread for major pairs
                bid = price - spread / 2
                ask = price + spread / 2
                
                return ForexTick(
                    symbol=symbol,
                    bid=bid,
                    ask=ask,
                    timestamp=timestamp,
                    source=DataSource.YAHOO,
                    volume=data.get('volume')
                )
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message from Yahoo Finance: {message[:100]}")
        except Exception as e:
            logger.error(f"Error parsing Yahoo Finance message: {e}")
            
        return None
    
    def _parse_oanda_message(self, message: str) -> Optional[ForexTick]:
        """
        Parse OANDA streaming message | 解析OANDA流式消息
        """
        try:
            data = json.loads(message)
            
            if 'type' in data and data['type'] == 'PRICE':
                price_data = data
                symbol = price_data['instrument'].replace('_', '')
                
                return ForexTick(
                    symbol=symbol,
                    bid=float(price_data['bids'][0]['price']),
                    ask=float(price_data['asks'][0]['price']),
                    timestamp=datetime.fromisoformat(price_data['time'].replace('Z', '+00:00')),
                    source=DataSource.OANDA
                )
                
        except Exception as e:
            logger.error(f"Error parsing OANDA message: {e}")
            
        return None
    
    def _start_heartbeat(self):
        """
        Start heartbeat thread to keep connection alive | 啟動心跳線程保持連接活躍
        """
        def heartbeat_worker():
            """Heartbeat worker thread | 心跳工作線程"""
            while self.is_running and self.metrics.status == ConnectionStatus.CONNECTED:
                try:
                    if self.ws and self.ws.sock:
                        self.ws.ping()
                    time.sleep(self.heartbeat_interval)
                except Exception as e:
                    logger.warning(f"Heartbeat failed for {self.source.value}: {e}")
                    break
        
        if self.heartbeat_thread is None or not self.heartbeat_thread.is_alive():
            self.heartbeat_thread = threading.Thread(target=heartbeat_worker, daemon=True)
            self.heartbeat_thread.start()
    
    def _start_reconnection(self):
        """
        Start automatic reconnection | 啟動自動重連
        """
        def reconnect_worker():
            """Reconnection worker thread | 重連工作線程"""
            attempt = 0
            
            while (self.should_reconnect and 
                   attempt < self.max_reconnect_attempts and 
                   self.metrics.status != ConnectionStatus.CONNECTED):
                
                attempt += 1
                self.metrics.status = ConnectionStatus.RECONNECTING
                self.metrics.reconnect_count += 1
                
                logger.info(f"Attempting to reconnect {self.source.value} (attempt {attempt}/{self.max_reconnect_attempts})")
                
                # Exponential backoff | 指數退避
                wait_time = min(self.reconnect_interval * (2 ** (attempt - 1)), 300)
                time.sleep(wait_time)
                
                try:
                    self.connect()
                    break
                except Exception as e:
                    logger.error(f"Reconnection attempt {attempt} failed for {self.source.value}: {e}")
        
        if self.reconnect_thread is None or not self.reconnect_thread.is_alive():
            self.reconnect_thread = threading.Thread(target=reconnect_worker, daemon=True)
            self.reconnect_thread.start()


class RealTimeForexFeed:
    """
    Main real-time forex data feed manager | 主要實時外匯數據源管理器
    """
    
    def __init__(self, config_path: str = "config/data-sources.yaml"):
        """
        Initialize real-time forex feed | 初始化實時外匯數據源
        
        Args:
            config_path: Path to configuration file | 配置文件路徑
        """
        # Load configuration | 載入配置
        self.config = self._load_config(config_path)
        
        # WebSocket managers | WebSocket管理器
        self.websocket_managers: Dict[DataSource, WebSocketManager] = {}
        
        # Data handlers | 數據處理器
        self.data_subscribers: List[Callable[[ForexTick], None]] = []
        
        # Redis connection for caching | Redis連接用於緩存
        self.redis_client: Optional[redis.Redis] = None
        self._init_redis_connection()
        
        # Metrics | 指標
        self.start_time = datetime.now()
        self.total_ticks_received = 0
        
        logger.info("Real-time forex feed initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file | 從YAML文件載入配置
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    
    def _init_redis_connection(self):
        """
        Initialize Redis connection for caching | 初始化Redis連接用於緩存
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
            
            # Test connection | 測試連接
            self.redis_client.ping()
            logger.info("Redis connection established for forex data caching")
            
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def add_data_subscriber(self, handler: Callable[[ForexTick], None]):
        """
        Add data subscriber callback | 添加數據訂閱者回調
        """
        self.data_subscribers.append(handler)
        logger.info(f"Added data subscriber: {handler.__name__}")
    
    def start(self, sources: Optional[List[DataSource]] = None):
        """
        Start real-time data feed | 啟動實時數據源
        
        Args:
            sources: List of data sources to start | 要啟動的數據源列表
        """
        if sources is None:
            # Start all enabled sources | 啟動所有啟用的數據源
            sources = []
            for source_name, source_config in self.config['forex_data_sources'].items():
                if source_config.get('enabled', False):
                    sources.append(DataSource(source_name))
        
        logger.info(f"Starting real-time forex feed with sources: {[s.value for s in sources]}")
        
        for source in sources:
            self._start_source(source)
        
        logger.info("Real-time forex feed started successfully")
    
    def stop(self):
        """
        Stop all data feeds | 停止所有數據源
        """
        logger.info("Stopping real-time forex feed...")
        
        for manager in self.websocket_managers.values():
            manager.disconnect()
        
        self.websocket_managers.clear()
        
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("Real-time forex feed stopped")
    
    def _start_source(self, source: DataSource):
        """
        Start specific data source | 啟動特定數據源
        """
        try:
            source_config = self.config['forex_data_sources'][source.value]
            
            if not source_config.get('enabled', False):
                logger.warning(f"Data source {source.value} is disabled")
                return
            
            # Create WebSocket manager | 創建WebSocket管理器
            manager = WebSocketManager(
                source=source,
                config=source_config,
                data_handler=self._handle_tick_data
            )
            
            self.websocket_managers[source] = manager
            
            # Start connection in background thread | 在後台線程中啟動連接
            connection_thread = threading.Thread(
                target=manager.connect,
                daemon=True
            )
            connection_thread.start()
            
            logger.info(f"Started data source: {source.value}")
            
        except Exception as e:
            logger.error(f"Failed to start data source {source.value}: {e}")
    
    def _handle_tick_data(self, tick: ForexTick):
        """
        Handle incoming tick data | 處理傳入的跳動數據
        """
        try:
            self.total_ticks_received += 1
            
            # Cache tick data in Redis | 在Redis中緩存跳動數據
            self._cache_tick_data(tick)
            
            # Notify all subscribers | 通知所有訂閱者
            for subscriber in self.data_subscribers:
                try:
                    subscriber(tick)
                except Exception as e:
                    logger.error(f"Error in data subscriber {subscriber.__name__}: {e}")
            
            # Log periodic statistics | 記錄定期統計
            if self.total_ticks_received % 1000 == 0:
                logger.info(f"Processed {self.total_ticks_received} ticks from all sources")
            
        except Exception as e:
            logger.error(f"Error handling tick data: {e}")
    
    def _cache_tick_data(self, tick: ForexTick):
        """
        Cache tick data in Redis | 在Redis中緩存跳動數據
        """
        if not self.redis_client:
            return
        
        try:
            # Create cache key | 創建緩存鍵
            cache_key = f"forex:realtime:{tick.symbol}:{tick.source.value}"
            
            # Store latest tick | 存儲最新跳動
            self.redis_client.hset(cache_key, mapping=tick.to_dict())
            
            # Set expiration | 設置過期時間
            ttl = self.config.get('caching', {}).get('default_ttl', 300)
            self.redis_client.expire(cache_key, ttl)
            
            # Add to time series stream if enabled | 如果啟用，添加到時間序列流
            if self.config.get('caching', {}).get('redis', {}).get('use_streams', False):
                stream_key = f"forex:stream:{tick.symbol}"
                self.redis_client.xadd(stream_key, tick.to_dict())
            
        except Exception as e:
            logger.error(f"Failed to cache tick data: {e}")
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get current connection status for all sources | 獲取所有數據源的當前連接狀態
        """
        status = {
            'total_sources': len(self.websocket_managers),
            'connected_sources': 0,
            'total_ticks_received': self.total_ticks_received,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'sources': {}
        }
        
        for source, manager in self.websocket_managers.items():
            metrics = manager.metrics
            status['sources'][source.value] = {
                'status': metrics.status.value,
                'connected_at': metrics.connected_at.isoformat() if metrics.connected_at else None,
                'uptime_seconds': metrics.uptime_seconds,
                'total_messages': metrics.total_messages,
                'error_count': metrics.error_count,
                'reconnect_count': metrics.reconnect_count,
                'is_healthy': metrics.is_healthy,
                'average_latency_ms': metrics.average_latency_ms
            }
            
            if metrics.status == ConnectionStatus.CONNECTED:
                status['connected_sources'] += 1
        
        return status
    
    def get_latest_tick(self, symbol: str, source: Optional[DataSource] = None) -> Optional[ForexTick]:
        """
        Get latest tick for a symbol | 獲取品種的最新跳動
        """
        if not self.redis_client:
            return None
        
        try:
            if source:
                cache_key = f"forex:realtime:{symbol}:{source.value}"
                data = self.redis_client.hgetall(cache_key)
                
                if data:
                    return ForexTick(
                        symbol=data['symbol'],
                        bid=float(data['bid']),
                        ask=float(data['ask']),
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        source=DataSource(data['source']),
                        volume=int(data['volume']) if data.get('volume') else None
                    )
            else:
                # Get from any available source | 從任何可用數據源獲取
                pattern = f"forex:realtime:{symbol}:*"
                keys = self.redis_client.keys(pattern)
                
                if keys:
                    # Return most recent data | 返回最新數據
                    latest_key = sorted(keys)[-1]  # Simple sorting, could be improved
                    data = self.redis_client.hgetall(latest_key)
                    
                    if data:
                        return ForexTick(
                            symbol=data['symbol'],
                            bid=float(data['bid']),
                            ask=float(data['ask']),
                            timestamp=datetime.fromisoformat(data['timestamp']),
                            source=DataSource(data['source']),
                            volume=int(data['volume']) if data.get('volume') else None
                        )
        
        except Exception as e:
            logger.error(f"Failed to get latest tick for {symbol}: {e}")
        
        return None


# Usage example and factory function | 使用示例和工廠函數
def create_realtime_forex_feed(config_path: str = "config/data-sources.yaml") -> RealTimeForexFeed:
    """
    Create and configure real-time forex feed | 創建並配置實時外匯數據源
    """
    return RealTimeForexFeed(config_path)


if __name__ == "__main__":
    # Example usage | 使用示例
    def tick_handler(tick: ForexTick):
        """Example tick handler | 示例跳動處理器"""
        print(f"Received tick: {tick.symbol} | Bid: {tick.bid} | Ask: {tick.ask} | Source: {tick.source.value}")
    
    # Create and start feed | 創建並啟動數據源
    feed = create_realtime_forex_feed()
    feed.add_data_subscriber(tick_handler)
    
    try:
        feed.start([DataSource.YAHOO])  # Start only Yahoo Finance for testing
        
        # Keep running | 保持運行
        while True:
            time.sleep(60)
            status = feed.get_connection_status()
            print(f"Status: {status['connected_sources']}/{status['total_sources']} sources connected")
            
    except KeyboardInterrupt:
        print("Stopping feed...")
        feed.stop()