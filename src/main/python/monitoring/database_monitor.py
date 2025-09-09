"""
AIFX Database Monitoring & Alerting System | AIFX 資料庫監控與警報系統
Real-time database performance monitoring and intelligent alerting
實時資料庫性能監控和智能警報

Phase 4.1.4 Database Optimization Component
第四階段 4.1.4 資料庫優化組件
"""

import time
import logging
import threading
import psutil
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
import redis
import pymongo
import requests
import json

# Logger setup | 日誌設置
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels | 警報嚴重級別"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class MetricType(Enum):
    """Metric type classification | 指標類型分類"""
    PERFORMANCE = "PERFORMANCE"
    AVAILABILITY = "AVAILABILITY"
    CAPACITY = "CAPACITY"
    SECURITY = "SECURITY"
    BUSINESS = "BUSINESS"


@dataclass
class DatabaseMetric:
    """
    Database metric data structure | 資料庫指標數據結構
    """
    name: str
    value: float
    unit: str
    timestamp: datetime
    metric_type: MetricType
    source: str  # postgres, redis, mongodb, etc.
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Thresholds for alerting | 警報閾值
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None


@dataclass
class Alert:
    """
    Database alert information | 資料庫警報信息
    """
    id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    current_value: float
    threshold: float
    timestamp: datetime
    source: str
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    acknowledgment_time: Optional[datetime] = None
    acknowledged_by: Optional[str] = None


class DatabaseMonitor:
    """
    Comprehensive database monitoring system | 綜合資料庫監控系統
    """
    
    def __init__(self, connection_manager, config: Dict[str, Any]):
        """
        Initialize database monitor | 初始化資料庫監控器
        
        Args:
            connection_manager: Database connection manager | 資料庫連接管理器
            config: Monitoring configuration | 監控配置
        """
        self.connection_manager = connection_manager
        self.config = config
        self.metrics_history: Dict[str, List[DatabaseMetric]] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable] = []
        
        # Monitoring thread control | 監控線程控制
        self._monitoring_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._monitoring_interval = config.get('monitoring_interval', 30)  # seconds
        
        # Alert thresholds | 警報閾值
        self._setup_default_thresholds()
        
        logger.info("Database monitor initialized")
    
    def _setup_default_thresholds(self):
        """Setup default monitoring thresholds | 設置默認監控閾值"""
        self.thresholds = {
            # PostgreSQL thresholds | PostgreSQL閾值
            'postgres_connections_ratio': {'warning': 0.8, 'critical': 0.95},
            'postgres_cpu_usage': {'warning': 80, 'critical': 95},
            'postgres_memory_usage': {'warning': 85, 'critical': 95},
            'postgres_disk_usage': {'warning': 80, 'critical': 90},
            'postgres_query_avg_time': {'warning': 1000, 'critical': 5000},  # milliseconds
            'postgres_cache_hit_ratio': {'warning': 0.85, 'critical': 0.75},  # inverted
            'postgres_locks_count': {'warning': 100, 'critical': 500},
            'postgres_deadlocks': {'warning': 5, 'critical': 20},
            
            # Redis thresholds | Redis閾值
            'redis_memory_usage': {'warning': 80, 'critical': 95},
            'redis_connections_ratio': {'warning': 0.8, 'critical': 0.95},
            'redis_keyspace_hit_ratio': {'warning': 0.85, 'critical': 0.75},  # inverted
            'redis_evicted_keys': {'warning': 100, 'critical': 1000},
            'redis_blocked_clients': {'warning': 10, 'critical': 50},
            
            # System thresholds | 系統閾值
            'system_cpu_usage': {'warning': 80, 'critical': 95},
            'system_memory_usage': {'warning': 85, 'critical': 95},
            'system_disk_usage': {'warning': 80, 'critical': 90},
            'system_network_errors': {'warning': 100, 'critical': 1000},
        }
    
    def collect_postgres_metrics(self) -> List[DatabaseMetric]:
        """
        Collect PostgreSQL performance metrics | 收集PostgreSQL性能指標
        """
        metrics = []
        current_time = datetime.now()
        
        try:
            with self.connection_manager.get_connection('postgres_primary') as conn:
                # Connection metrics | 連接指標
                result = conn.execute(text("""
                    SELECT 
                        count(*) as total_connections,
                        count(*) FILTER (WHERE state = 'active') as active_connections,
                        count(*) FILTER (WHERE state = 'idle') as idle_connections,
                        (SELECT setting::int FROM pg_settings WHERE name = 'max_connections') as max_connections
                    FROM pg_stat_activity
                """))
                
                row = result.fetchone()
                if row:
                    connection_ratio = row.total_connections / row.max_connections
                    
                    metrics.extend([
                        DatabaseMetric(
                            name="postgres_total_connections",
                            value=row.total_connections,
                            unit="count",
                            timestamp=current_time,
                            metric_type=MetricType.PERFORMANCE,
                            source="postgres",
                            warning_threshold=self.thresholds['postgres_connections_ratio']['warning'] * row.max_connections,
                            critical_threshold=self.thresholds['postgres_connections_ratio']['critical'] * row.max_connections
                        ),
                        DatabaseMetric(
                            name="postgres_active_connections",
                            value=row.active_connections,
                            unit="count",
                            timestamp=current_time,
                            metric_type=MetricType.PERFORMANCE,
                            source="postgres"
                        ),
                        DatabaseMetric(
                            name="postgres_connections_ratio",
                            value=connection_ratio,
                            unit="ratio",
                            timestamp=current_time,
                            metric_type=MetricType.CAPACITY,
                            source="postgres",
                            warning_threshold=self.thresholds['postgres_connections_ratio']['warning'],
                            critical_threshold=self.thresholds['postgres_connections_ratio']['critical']
                        )
                    ])
                
                # Query performance metrics | 查詢性能指標
                result = conn.execute(text("""
                    SELECT 
                        COALESCE(AVG(mean_exec_time), 0) as avg_query_time,
                        COALESCE(MAX(mean_exec_time), 0) as max_query_time,
                        COALESCE(SUM(calls), 0) as total_queries
                    FROM pg_stat_statements
                    WHERE last_exec > NOW() - INTERVAL '5 minutes'
                """))
                
                row = result.fetchone()
                if row:
                    metrics.extend([
                        DatabaseMetric(
                            name="postgres_query_avg_time",
                            value=row.avg_query_time,
                            unit="milliseconds",
                            timestamp=current_time,
                            metric_type=MetricType.PERFORMANCE,
                            source="postgres",
                            warning_threshold=self.thresholds['postgres_query_avg_time']['warning'],
                            critical_threshold=self.thresholds['postgres_query_avg_time']['critical']
                        ),
                        DatabaseMetric(
                            name="postgres_total_queries",
                            value=row.total_queries,
                            unit="count",
                            timestamp=current_time,
                            metric_type=MetricType.BUSINESS,
                            source="postgres"
                        )
                    ])
                
                # Cache hit ratio | 緩存命中率
                result = conn.execute(text("""
                    SELECT 
                        ROUND(
                            (sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read) + 1)) * 100, 2
                        ) as cache_hit_ratio
                    FROM pg_statio_user_tables
                """))
                
                row = result.fetchone()
                if row and row.cache_hit_ratio:
                    metrics.append(DatabaseMetric(
                        name="postgres_cache_hit_ratio",
                        value=row.cache_hit_ratio / 100,  # Convert to ratio
                        unit="ratio",
                        timestamp=current_time,
                        metric_type=MetricType.PERFORMANCE,
                        source="postgres",
                        warning_threshold=self.thresholds['postgres_cache_hit_ratio']['warning'],
                        critical_threshold=self.thresholds['postgres_cache_hit_ratio']['critical']
                    ))
                
                # Lock information | 鎖信息
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_locks,
                        COUNT(*) FILTER (WHERE mode LIKE '%ExclusiveLock%') as exclusive_locks,
                        COUNT(*) FILTER (WHERE NOT granted) as waiting_locks
                    FROM pg_locks
                """))
                
                row = result.fetchone()
                if row:
                    metrics.extend([
                        DatabaseMetric(
                            name="postgres_locks_count",
                            value=row.total_locks,
                            unit="count",
                            timestamp=current_time,
                            metric_type=MetricType.PERFORMANCE,
                            source="postgres",
                            warning_threshold=self.thresholds['postgres_locks_count']['warning'],
                            critical_threshold=self.thresholds['postgres_locks_count']['critical']
                        ),
                        DatabaseMetric(
                            name="postgres_waiting_locks",
                            value=row.waiting_locks,
                            unit="count",
                            timestamp=current_time,
                            metric_type=MetricType.PERFORMANCE,
                            source="postgres"
                        )
                    ])
                
                # Database size | 資料庫大小
                result = conn.execute(text("""
                    SELECT 
                        pg_database_size(current_database()) as db_size_bytes,
                        pg_size_pretty(pg_database_size(current_database())) as db_size_pretty
                """))
                
                row = result.fetchone()
                if row:
                    metrics.append(DatabaseMetric(
                        name="postgres_database_size",
                        value=row.db_size_bytes,
                        unit="bytes",
                        timestamp=current_time,
                        metric_type=MetricType.CAPACITY,
                        source="postgres",
                        tags={"size_pretty": row.db_size_pretty}
                    ))
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to collect PostgreSQL metrics: {e}")
        
        return metrics
    
    def collect_redis_metrics(self) -> List[DatabaseMetric]:
        """
        Collect Redis performance metrics | 收集Redis性能指標
        """
        metrics = []
        current_time = datetime.now()
        
        try:
            redis_manager = self.connection_manager  # Assume Redis manager is available
            if hasattr(redis_manager, 'get_client'):
                client = redis_manager.get_client('redis_primary')
                info = client.info()
                
                # Memory metrics | 內存指標
                used_memory = info.get('used_memory', 0)
                max_memory = info.get('maxmemory', psutil.virtual_memory().total)
                memory_usage_ratio = used_memory / max_memory if max_memory > 0 else 0
                
                metrics.extend([
                    DatabaseMetric(
                        name="redis_used_memory",
                        value=used_memory,
                        unit="bytes",
                        timestamp=current_time,
                        metric_type=MetricType.CAPACITY,
                        source="redis",
                        tags={"memory_human": info.get('used_memory_human', 'N/A')}
                    ),
                    DatabaseMetric(
                        name="redis_memory_usage",
                        value=memory_usage_ratio * 100,
                        unit="percent",
                        timestamp=current_time,
                        metric_type=MetricType.CAPACITY,
                        source="redis",
                        warning_threshold=self.thresholds['redis_memory_usage']['warning'],
                        critical_threshold=self.thresholds['redis_memory_usage']['critical']
                    )
                ])
                
                # Connection metrics | 連接指標
                connected_clients = info.get('connected_clients', 0)
                max_clients = info.get('maxclients', 10000)
                connection_ratio = connected_clients / max_clients
                
                metrics.extend([
                    DatabaseMetric(
                        name="redis_connected_clients",
                        value=connected_clients,
                        unit="count",
                        timestamp=current_time,
                        metric_type=MetricType.PERFORMANCE,
                        source="redis"
                    ),
                    DatabaseMetric(
                        name="redis_connections_ratio",
                        value=connection_ratio,
                        unit="ratio",
                        timestamp=current_time,
                        metric_type=MetricType.CAPACITY,
                        source="redis",
                        warning_threshold=self.thresholds['redis_connections_ratio']['warning'],
                        critical_threshold=self.thresholds['redis_connections_ratio']['critical']
                    )
                ])
                
                # Performance metrics | 性能指標
                keyspace_hits = info.get('keyspace_hits', 0)
                keyspace_misses = info.get('keyspace_misses', 0)
                total_keyspace = keyspace_hits + keyspace_misses
                hit_ratio = keyspace_hits / total_keyspace if total_keyspace > 0 else 1.0
                
                metrics.extend([
                    DatabaseMetric(
                        name="redis_keyspace_hit_ratio",
                        value=hit_ratio,
                        unit="ratio",
                        timestamp=current_time,
                        metric_type=MetricType.PERFORMANCE,
                        source="redis",
                        warning_threshold=self.thresholds['redis_keyspace_hit_ratio']['warning'],
                        critical_threshold=self.thresholds['redis_keyspace_hit_ratio']['critical']
                    ),
                    DatabaseMetric(
                        name="redis_evicted_keys",
                        value=info.get('evicted_keys', 0),
                        unit="count",
                        timestamp=current_time,
                        metric_type=MetricType.PERFORMANCE,
                        source="redis",
                        warning_threshold=self.thresholds['redis_evicted_keys']['warning'],
                        critical_threshold=self.thresholds['redis_evicted_keys']['critical']
                    ),
                    DatabaseMetric(
                        name="redis_blocked_clients",
                        value=info.get('blocked_clients', 0),
                        unit="count",
                        timestamp=current_time,
                        metric_type=MetricType.PERFORMANCE,
                        source="redis",
                        warning_threshold=self.thresholds['redis_blocked_clients']['warning'],
                        critical_threshold=self.thresholds['redis_blocked_clients']['critical']
                    )
                ])
                
        except Exception as e:
            logger.error(f"Failed to collect Redis metrics: {e}")
        
        return metrics
    
    def collect_system_metrics(self) -> List[DatabaseMetric]:
        """
        Collect system-level metrics | 收集系統級指標
        """
        metrics = []
        current_time = datetime.now()
        
        try:
            # CPU metrics | CPU指標
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(DatabaseMetric(
                name="system_cpu_usage",
                value=cpu_percent,
                unit="percent",
                timestamp=current_time,
                metric_type=MetricType.PERFORMANCE,
                source="system",
                warning_threshold=self.thresholds['system_cpu_usage']['warning'],
                critical_threshold=self.thresholds['system_cpu_usage']['critical']
            ))
            
            # Memory metrics | 內存指標
            memory = psutil.virtual_memory()
            metrics.extend([
                DatabaseMetric(
                    name="system_memory_usage",
                    value=memory.percent,
                    unit="percent",
                    timestamp=current_time,
                    metric_type=MetricType.CAPACITY,
                    source="system",
                    warning_threshold=self.thresholds['system_memory_usage']['warning'],
                    critical_threshold=self.thresholds['system_memory_usage']['critical']
                ),
                DatabaseMetric(
                    name="system_memory_available",
                    value=memory.available,
                    unit="bytes",
                    timestamp=current_time,
                    metric_type=MetricType.CAPACITY,
                    source="system"
                )
            ])
            
            # Disk metrics | 磁碟指標
            disk = psutil.disk_usage('/')
            metrics.extend([
                DatabaseMetric(
                    name="system_disk_usage",
                    value=disk.percent,
                    unit="percent",
                    timestamp=current_time,
                    metric_type=MetricType.CAPACITY,
                    source="system",
                    warning_threshold=self.thresholds['system_disk_usage']['warning'],
                    critical_threshold=self.thresholds['system_disk_usage']['critical']
                ),
                DatabaseMetric(
                    name="system_disk_free",
                    value=disk.free,
                    unit="bytes",
                    timestamp=current_time,
                    metric_type=MetricType.CAPACITY,
                    source="system"
                )
            ])
            
            # Network metrics | 網絡指標
            network = psutil.net_io_counters()
            if network:
                metrics.extend([
                    DatabaseMetric(
                        name="system_network_bytes_sent",
                        value=network.bytes_sent,
                        unit="bytes",
                        timestamp=current_time,
                        metric_type=MetricType.PERFORMANCE,
                        source="system"
                    ),
                    DatabaseMetric(
                        name="system_network_bytes_recv",
                        value=network.bytes_recv,
                        unit="bytes",
                        timestamp=current_time,
                        metric_type=MetricType.PERFORMANCE,
                        source="system"
                    ),
                    DatabaseMetric(
                        name="system_network_errors",
                        value=network.errin + network.errout,
                        unit="count",
                        timestamp=current_time,
                        metric_type=MetricType.PERFORMANCE,
                        source="system",
                        warning_threshold=self.thresholds['system_network_errors']['warning'],
                        critical_threshold=self.thresholds['system_network_errors']['critical']
                    )
                ])
                
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
        
        return metrics
    
    def check_thresholds(self, metrics: List[DatabaseMetric]) -> List[Alert]:
        """
        Check metrics against thresholds and generate alerts | 檢查指標閾值並生成警報
        """
        alerts = []
        current_time = datetime.now()
        
        for metric in metrics:
            if metric.warning_threshold is None and metric.critical_threshold is None:
                continue
            
            alert_id = f"{metric.source}_{metric.name}_{int(current_time.timestamp())}"
            severity = None
            threshold = None
            
            # Check critical threshold | 檢查關鍵閾值
            if metric.critical_threshold is not None:
                if (metric.name.endswith('_ratio') and 
                    metric.name in ['postgres_cache_hit_ratio', 'redis_keyspace_hit_ratio']):
                    # Inverted threshold (lower is worse) | 反向閾值（越低越差）
                    if metric.value < metric.critical_threshold:
                        severity = AlertSeverity.CRITICAL
                        threshold = metric.critical_threshold
                else:
                    # Normal threshold (higher is worse) | 正常閾值（越高越差）
                    if metric.value > metric.critical_threshold:
                        severity = AlertSeverity.CRITICAL
                        threshold = metric.critical_threshold
            
            # Check warning threshold | 檢查警告閾值
            if severity is None and metric.warning_threshold is not None:
                if (metric.name.endswith('_ratio') and 
                    metric.name in ['postgres_cache_hit_ratio', 'redis_keyspace_hit_ratio']):
                    # Inverted threshold | 反向閾值
                    if metric.value < metric.warning_threshold:
                        severity = AlertSeverity.HIGH
                        threshold = metric.warning_threshold
                else:
                    # Normal threshold | 正常閾值
                    if metric.value > metric.warning_threshold:
                        severity = AlertSeverity.HIGH
                        threshold = metric.warning_threshold
            
            # Generate alert if threshold exceeded | 如果超出閾值則生成警報
            if severity is not None:
                message = self._generate_alert_message(metric, threshold, severity)
                
                alert = Alert(
                    id=alert_id,
                    metric_name=metric.name,
                    severity=severity,
                    message=message,
                    current_value=metric.value,
                    threshold=threshold,
                    timestamp=current_time,
                    source=metric.source
                )
                
                alerts.append(alert)
                self.active_alerts[alert_id] = alert
        
        return alerts
    
    def _generate_alert_message(self, metric: DatabaseMetric, threshold: float, 
                              severity: AlertSeverity) -> str:
        """
        Generate alert message | 生成警報消息
        """
        unit_display = metric.unit if metric.unit != "ratio" else "%"
        value_display = f"{metric.value:.2f}" if metric.unit == "ratio" else str(metric.value)
        threshold_display = f"{threshold:.2f}" if metric.unit == "ratio" else str(threshold)
        
        if metric.unit == "ratio":
            value_display = f"{metric.value * 100:.1f}%"
            threshold_display = f"{threshold * 100:.1f}%"
        elif metric.unit == "bytes":
            value_display = self._format_bytes(metric.value)
            threshold_display = self._format_bytes(threshold)
        
        return (f"{severity.value}: {metric.source.upper()} {metric.name} is {value_display}, "
                f"exceeding {severity.value.lower()} threshold of {threshold_display}")
    
    def _format_bytes(self, bytes_value: float) -> str:
        """Format bytes in human readable format | 格式化字節為人類可讀格式"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024.0:
                return f"{bytes_value:.1f}{unit}"
            bytes_value /= 1024.0
        return f"{bytes_value:.1f}PB"
    
    def process_alerts(self, alerts: List[Alert]):
        """
        Process and send alerts | 處理並發送警報
        """
        for alert in alerts:
            # Log alert | 記錄警報
            logger.warning(f"Database Alert: {alert.message}")
            
            # Send to alert handlers | 發送到警報處理器
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """
        Add alert handler | 添加警報處理器
        """
        self.alert_handlers.append(handler)
    
    def send_email_alert(self, alert: Alert):
        """
        Send email alert | 發送電子郵件警報
        """
        # Implementation would use SMTP configuration from config
        # 實現將使用配置中的SMTP配置
        email_config = self.config.get('email_alerts', {})
        if not email_config.get('enabled', False):
            return
        
        logger.info(f"Email alert sent: {alert.message}")
    
    def send_slack_alert(self, alert: Alert):
        """
        Send Slack alert | 發送Slack警報
        """
        slack_config = self.config.get('slack_alerts', {})
        if not slack_config.get('enabled', False):
            return
        
        webhook_url = slack_config.get('webhook_url')
        if not webhook_url:
            return
        
        color_map = {
            AlertSeverity.CRITICAL: 'danger',
            AlertSeverity.HIGH: 'warning',
            AlertSeverity.MEDIUM: 'warning',
            AlertSeverity.LOW: 'good',
            AlertSeverity.INFO: 'good'
        }
        
        payload = {
            'attachments': [{
                'color': color_map.get(alert.severity, 'warning'),
                'title': f'AIFX Database Alert - {alert.severity.value}',
                'text': alert.message,
                'fields': [
                    {'title': 'Source', 'value': alert.source, 'short': True},
                    {'title': 'Metric', 'value': alert.metric_name, 'short': True},
                    {'title': 'Current Value', 'value': str(alert.current_value), 'short': True},
                    {'title': 'Threshold', 'value': str(alert.threshold), 'short': True}
                ],
                'timestamp': int(alert.timestamp.timestamp())
            }]
        }
        
        try:
            requests.post(webhook_url, json=payload, timeout=10)
            logger.info(f"Slack alert sent: {alert.id}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def start_monitoring(self):
        """
        Start the monitoring thread | 啟動監控線程
        """
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Monitoring already started")
            return
        
        def monitoring_loop():
            """Main monitoring loop | 主監控循環"""
            logger.info("Database monitoring started")
            
            while not self._shutdown_event.is_set():
                try:
                    # Collect all metrics | 收集所有指標
                    all_metrics = []
                    
                    # PostgreSQL metrics | PostgreSQL指標
                    postgres_metrics = self.collect_postgres_metrics()
                    all_metrics.extend(postgres_metrics)
                    
                    # Redis metrics | Redis指標
                    redis_metrics = self.collect_redis_metrics()
                    all_metrics.extend(redis_metrics)
                    
                    # System metrics | 系統指標
                    system_metrics = self.collect_system_metrics()
                    all_metrics.extend(system_metrics)
                    
                    # Store metrics history | 存儲指標歷史
                    for metric in all_metrics:
                        if metric.name not in self.metrics_history:
                            self.metrics_history[metric.name] = []
                        
                        self.metrics_history[metric.name].append(metric)
                        
                        # Keep only last 1000 points per metric | 每個指標只保留最後1000個點
                        if len(self.metrics_history[metric.name]) > 1000:
                            self.metrics_history[metric.name] = self.metrics_history[metric.name][-1000:]
                    
                    # Check thresholds and generate alerts | 檢查閾值並生成警報
                    alerts = self.check_thresholds(all_metrics)
                    if alerts:
                        self.process_alerts(alerts)
                    
                    # Clean up resolved alerts | 清理已解決的警報
                    self._cleanup_resolved_alerts()
                    
                    logger.debug(f"Collected {len(all_metrics)} metrics, generated {len(alerts)} alerts")
                    
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                
                # Wait for next iteration | 等待下次迭代
                self._shutdown_event.wait(self._monitoring_interval)
        
        # Setup default alert handlers | 設置默認警報處理器
        self.add_alert_handler(self.send_slack_alert)
        
        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()
    
    def stop_monitoring(self):
        """
        Stop the monitoring thread | 停止監控線程
        """
        logger.info("Stopping database monitoring...")
        self._shutdown_event.set()
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
        
        logger.info("Database monitoring stopped")
    
    def _cleanup_resolved_alerts(self):
        """
        Clean up resolved alerts | 清理已解決的警報
        """
        resolved_cutoff = datetime.now() - timedelta(hours=24)
        resolved_alerts = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.resolved and alert.resolution_time and alert.resolution_time < resolved_cutoff
        ]
        
        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        Get current monitoring status | 獲取當前監控狀態
        """
        return {
            "monitoring_active": self._monitoring_thread and self._monitoring_thread.is_alive(),
            "monitoring_interval": self._monitoring_interval,
            "metrics_tracked": len(self.metrics_history),
            "active_alerts": len([a for a in self.active_alerts.values() if not a.resolved]),
            "alert_handlers": len(self.alert_handlers),
            "last_collection": max(
                [max(metrics, key=lambda m: m.timestamp).timestamp
                 for metrics in self.metrics_history.values()]
            ) if self.metrics_history else None
        }
    
    def get_recent_metrics(self, metric_name: str, hours: int = 1) -> List[DatabaseMetric]:
        """
        Get recent metrics for a specific metric | 獲取特定指標的最近指標
        """
        if metric_name not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            metric for metric in self.metrics_history[metric_name]
            if metric.timestamp >= cutoff_time
        ]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """
        Acknowledge an alert | 確認警報
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledgment_time = datetime.now()
            alert.acknowledged_by = acknowledged_by
            logger.info(f"Alert acknowledged by {acknowledged_by}: {alert_id}")
    
    def resolve_alert(self, alert_id: str):
        """
        Mark alert as resolved | 將警報標記為已解決
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = datetime.now()
            logger.info(f"Alert resolved: {alert_id}")


# Factory function for easy initialization | 便於初始化的工廠函數
def create_database_monitor(connection_manager, config: Optional[Dict] = None) -> DatabaseMonitor:
    """
    Create and configure database monitor | 創建並配置資料庫監控器
    """
    if config is None:
        config = {
            'monitoring_interval': 30,
            'slack_alerts': {
                'enabled': False,
                'webhook_url': None
            },
            'email_alerts': {
                'enabled': False
            }
        }
    
    monitor = DatabaseMonitor(connection_manager, config)
    return monitor