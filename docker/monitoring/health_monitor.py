#!/usr/bin/env python3
"""
AIFX Health Monitoring System | AIFX 健康監控系統
Comprehensive container and application health monitoring for production deployment
生產部署的綜合容器和應用程式健康監控

Features | 功能:
- Container health checks | 容器健康檢查
- Application component monitoring | 應用程式組件監控
- Resource usage tracking | 資源使用追踪
- Alert generation | 警報生成
- Performance metrics collection | 性能指標收集
"""

import os
import sys
import time
import json
import psutil
import docker
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import redis
import psycopg2
from pathlib import Path

# Add project root to path | 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "main" / "python"))

# Configure logging | 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/app/logs/health_monitor.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# HEALTH CHECK DATA CLASSES | 健康檢查資料類別
# ============================================================================

@dataclass
class HealthStatus:
    """Health status for a component | 組件健康狀態"""
    component: str
    status: str  # "healthy", "warning", "critical", "unknown"
    message: str
    timestamp: datetime
    response_time_ms: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class ResourceMetrics:
    """System resource metrics | 系統資源指標"""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    timestamp: datetime

@dataclass
class ContainerStatus:
    """Docker container status | Docker容器狀態"""
    container_id: str
    name: str
    status: str
    health: str
    cpu_usage: float
    memory_usage: float
    network_io: Dict[str, int]
    timestamp: datetime

# ============================================================================
# HEALTH MONITOR CLASS | 健康監控類別
# ============================================================================

class AIFXHealthMonitor:
    """
    AIFX Health Monitoring System | AIFX 健康監控系統
    
    Monitors all aspects of the AIFX trading system including:
    - Application components health | 應用程式組件健康
    - Container status and resources | 容器狀態和資源
    - Database connectivity | 資料庫連接性
    - Cache performance | 緩存性能
    - AI model availability | AI模型可用性
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize health monitor | 初始化健康監控器"""
        self.config_path = config_path or "/app/config/production.yaml"
        self.docker_client = None
        self.redis_client = None
        self.postgres_conn = None
        self.alerts_enabled = True
        
        # Health check configurations | 健康檢查配置
        self.health_check_interval = 30  # seconds
        self.warning_thresholds = {
            'cpu_percent': 80,
            'memory_percent': 80,
            'disk_percent': 85,
            'response_time_ms': 1000
        }
        self.critical_thresholds = {
            'cpu_percent': 95,
            'memory_percent': 95,
            'disk_percent': 95,
            'response_time_ms': 5000
        }
        
        # Initialize monitoring components | 初始化監控組件
        self._initialize_monitoring()
        
        logger.info("🏥 AIFX Health Monitor initialized successfully")
    
    def _initialize_monitoring(self):
        """Initialize monitoring connections | 初始化監控連接"""
        try:
            # Initialize Docker client | 初始化Docker客戶端
            self.docker_client = docker.from_env()
            logger.info("✅ Docker client connected")
            
            # Initialize Redis client | 初始化Redis客戶端
            try:
                self.redis_client = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'redis'),
                    port=int(os.getenv('REDIS_PORT', 6379)),
                    db=0,
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5
                )
                self.redis_client.ping()
                logger.info("✅ Redis client connected")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}")
                self.redis_client = None
            
            # Initialize PostgreSQL connection | 初始化PostgreSQL連接
            try:
                self.postgres_conn = psycopg2.connect(
                    host=os.getenv('POSTGRES_HOST', 'postgres'),
                    port=int(os.getenv('POSTGRES_PORT', 5432)),
                    database=os.getenv('POSTGRES_DB', 'aifx_production'),
                    user=os.getenv('POSTGRES_USER', 'aifx_user'),
                    password=os.getenv('POSTGRES_PASSWORD', ''),
                    connect_timeout=10
                )
                logger.info("✅ PostgreSQL client connected")
            except Exception as e:
                logger.warning(f"⚠️ PostgreSQL connection failed: {e}")
                self.postgres_conn = None
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize monitoring: {e}")
    
    def check_system_resources(self) -> ResourceMetrics:
        """Check system resource usage | 檢查系統資源使用情況"""
        try:
            # CPU usage | CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage | 內存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage | 磁碟使用率
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network I/O | 網絡I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io=network_io,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to check system resources: {e}")
            return ResourceMetrics(0, 0, 0, {}, datetime.utcnow())
    
    def check_container_health(self) -> List[ContainerStatus]:
        """Check Docker container health | 檢查Docker容器健康狀況"""
        container_statuses = []
        
        if not self.docker_client:
            logger.warning("⚠️ Docker client not available")
            return container_statuses
        
        try:
            containers = self.docker_client.containers.list(all=True)
            
            for container in containers:
                try:
                    # Get container stats | 獲取容器統計信息
                    stats = container.stats(stream=False)
                    
                    # Calculate CPU usage | 計算CPU使用率
                    cpu_usage = self._calculate_cpu_usage(stats)
                    
                    # Calculate memory usage | 計算內存使用率
                    memory_usage = self._calculate_memory_usage(stats)
                    
                    # Get network I/O | 獲取網絡I/O
                    network_io = self._get_network_io(stats)
                    
                    # Get health status | 獲取健康狀態
                    health_status = container.attrs.get('State', {}).get('Health', {}).get('Status', 'unknown')
                    
                    container_status = ContainerStatus(
                        container_id=container.id[:12],
                        name=container.name,
                        status=container.status,
                        health=health_status,
                        cpu_usage=cpu_usage,
                        memory_usage=memory_usage,
                        network_io=network_io,
                        timestamp=datetime.utcnow()
                    )
                    
                    container_statuses.append(container_status)
                    
                except Exception as e:
                    logger.error(f"❌ Failed to get stats for container {container.name}: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to check container health: {e}")
        
        return container_statuses
    
    def check_database_health(self) -> HealthStatus:
        """Check PostgreSQL database health | 檢查PostgreSQL資料庫健康狀況"""
        start_time = time.time()
        
        try:
            if not self.postgres_conn:
                return HealthStatus(
                    component="postgresql",
                    status="critical",
                    message="Database connection not available",
                    timestamp=datetime.utcnow()
                )
            
            # Test database connection | 測試資料庫連接
            cursor = self.postgres_conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Check response time thresholds | 檢查響應時間閾值
            if response_time_ms > self.critical_thresholds['response_time_ms']:
                status = "critical"
                message = f"Database response time too high: {response_time_ms:.2f}ms"
            elif response_time_ms > self.warning_thresholds['response_time_ms']:
                status = "warning"
                message = f"Database response time elevated: {response_time_ms:.2f}ms"
            else:
                status = "healthy"
                message = f"Database connection OK: {response_time_ms:.2f}ms"
            
            return HealthStatus(
                component="postgresql",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time_ms
            )
            
        except Exception as e:
            return HealthStatus(
                component="postgresql",
                status="critical",
                message=f"Database health check failed: {str(e)}",
                timestamp=datetime.utcnow()
            )
    
    def check_redis_health(self) -> HealthStatus:
        """Check Redis cache health | 檢查Redis緩存健康狀況"""
        start_time = time.time()
        
        try:
            if not self.redis_client:
                return HealthStatus(
                    component="redis",
                    status="critical",
                    message="Redis connection not available",
                    timestamp=datetime.utcnow()
                )
            
            # Test Redis connection | 測試Redis連接
            self.redis_client.ping()
            
            # Get Redis info | 獲取Redis信息
            info = self.redis_client.info()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Check memory usage | 檢查內存使用
            memory_usage = info.get('used_memory', 0)
            max_memory = info.get('maxmemory', 0)
            memory_percent = (memory_usage / max_memory * 100) if max_memory > 0 else 0
            
            details = {
                'connected_clients': info.get('connected_clients', 0),
                'used_memory_human': info.get('used_memory_human', 'unknown'),
                'memory_usage_percent': memory_percent,
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
            
            # Determine status | 確定狀態
            if memory_percent > 90:
                status = "critical"
                message = f"Redis memory usage critical: {memory_percent:.1f}%"
            elif memory_percent > 80:
                status = "warning"
                message = f"Redis memory usage high: {memory_percent:.1f}%"
            else:
                status = "healthy"
                message = f"Redis connection OK: {response_time_ms:.2f}ms"
            
            return HealthStatus(
                component="redis",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time_ms,
                details=details
            )
            
        except Exception as e:
            return HealthStatus(
                component="redis",
                status="critical",
                message=f"Redis health check failed: {str(e)}",
                timestamp=datetime.utcnow()
            )
    
    def check_aifx_application_health(self) -> HealthStatus:
        """Check AIFX application health | 檢查AIFX應用程式健康狀況"""
        start_time = time.time()
        
        try:
            # Test AIFX core components | 測試AIFX核心組件
            import sys
            sys.path.append('/app/src/main/python')
            
            from core.risk_manager import AdvancedRiskManager
            from core.trading_strategy import AIFXTradingStrategy
            from core.signal_combiner import SignalAggregator
            
            # Test component initialization | 測試組件初始化
            risk_manager = AdvancedRiskManager()
            signal_aggregator = SignalAggregator()
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Check if models are available | 檢查模型是否可用
            model_path = Path("/app/models/trained")
            models_available = []
            
            for model_file in ["xgboost_model.joblib", "rf_model.joblib", "lstm_model.h5"]:
                if (model_path / model_file).exists():
                    models_available.append(model_file)
            
            details = {
                'models_available': models_available,
                'models_count': len(models_available),
                'risk_manager_status': 'initialized',
                'signal_aggregator_status': 'initialized'
            }
            
            # Determine status | 確定狀態
            if len(models_available) < 2:
                status = "warning"
                message = f"Only {len(models_available)} AI models available"
            elif response_time_ms > self.warning_thresholds['response_time_ms']:
                status = "warning" 
                message = f"Application initialization slow: {response_time_ms:.2f}ms"
            else:
                status = "healthy"
                message = f"AIFX application healthy: {response_time_ms:.2f}ms"
            
            return HealthStatus(
                component="aifx_application",
                status=status,
                message=message,
                timestamp=datetime.utcnow(),
                response_time_ms=response_time_ms,
                details=details
            )
            
        except Exception as e:
            return HealthStatus(
                component="aifx_application",
                status="critical",
                message=f"AIFX application health check failed: {str(e)}",
                timestamp=datetime.utcnow()
            )
    
    def run_comprehensive_health_check(self) -> Dict[str, Any]:
        """Run comprehensive health check | 運行綜合健康檢查"""
        logger.info("🔍 Running comprehensive health check...")
        
        health_report = {
            'timestamp': datetime.utcnow().isoformat(),
            'system_resources': None,
            'container_health': [],
            'component_health': {},
            'overall_status': 'unknown',
            'alerts': []
        }
        
        try:
            # Check system resources | 檢查系統資源
            health_report['system_resources'] = asdict(self.check_system_resources())
            
            # Check container health | 檢查容器健康
            containers = self.check_container_health()
            health_report['container_health'] = [asdict(c) for c in containers]
            
            # Check component health | 檢查組件健康
            components = ['postgresql', 'redis', 'aifx_application']
            component_statuses = []
            
            for component in components:
                if component == 'postgresql':
                    status = self.check_database_health()
                elif component == 'redis':
                    status = self.check_redis_health()
                elif component == 'aifx_application':
                    status = self.check_aifx_application_health()
                
                health_report['component_health'][component] = asdict(status)
                component_statuses.append(status.status)
            
            # Determine overall status | 確定整體狀態
            if 'critical' in component_statuses:
                health_report['overall_status'] = 'critical'
            elif 'warning' in component_statuses:
                health_report['overall_status'] = 'warning'
            else:
                health_report['overall_status'] = 'healthy'
            
            # Generate alerts if needed | 如需要則生成警報
            health_report['alerts'] = self._generate_alerts(health_report)
            
            logger.info(f"✅ Health check completed - Overall status: {health_report['overall_status']}")
            
        except Exception as e:
            logger.error(f"❌ Comprehensive health check failed: {e}")
            health_report['overall_status'] = 'critical'
            health_report['alerts'] = [f"Health check system error: {str(e)}"]
        
        return health_report
    
    def _calculate_cpu_usage(self, stats: Dict) -> float:
        """Calculate CPU usage percentage from Docker stats | 從Docker統計計算CPU使用百分比"""
        try:
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            
            if system_delta > 0:
                return (cpu_delta / system_delta) * len(stats['cpu_stats']['cpu_usage']['percpu_usage']) * 100
        except (KeyError, ZeroDivisionError):
            pass
        return 0.0
    
    def _calculate_memory_usage(self, stats: Dict) -> float:
        """Calculate memory usage percentage from Docker stats | 從Docker統計計算內存使用百分比"""
        try:
            usage = stats['memory_stats']['usage']
            limit = stats['memory_stats']['limit']
            return (usage / limit) * 100 if limit > 0 else 0.0
        except KeyError:
            return 0.0
    
    def _get_network_io(self, stats: Dict) -> Dict[str, int]:
        """Get network I/O from Docker stats | 從Docker統計獲取網絡I/O"""
        try:
            networks = stats['networks']
            total_rx = sum(net['rx_bytes'] for net in networks.values())
            total_tx = sum(net['tx_bytes'] for net in networks.values())
            return {'rx_bytes': total_rx, 'tx_bytes': total_tx}
        except KeyError:
            return {'rx_bytes': 0, 'tx_bytes': 0}
    
    def _generate_alerts(self, health_report: Dict[str, Any]) -> List[str]:
        """Generate alerts based on health report | 根據健康報告生成警報"""
        alerts = []
        
        if not self.alerts_enabled:
            return alerts
        
        try:
            # Check system resource alerts | 檢查系統資源警報
            resources = health_report.get('system_resources', {})
            if resources:
                cpu = resources.get('cpu_percent', 0)
                memory = resources.get('memory_percent', 0)
                disk = resources.get('disk_percent', 0)
                
                if cpu > self.critical_thresholds['cpu_percent']:
                    alerts.append(f"CRITICAL: CPU usage {cpu:.1f}% exceeds critical threshold")
                elif cpu > self.warning_thresholds['cpu_percent']:
                    alerts.append(f"WARNING: CPU usage {cpu:.1f}% exceeds warning threshold")
                
                if memory > self.critical_thresholds['memory_percent']:
                    alerts.append(f"CRITICAL: Memory usage {memory:.1f}% exceeds critical threshold")
                elif memory > self.warning_thresholds['memory_percent']:
                    alerts.append(f"WARNING: Memory usage {memory:.1f}% exceeds warning threshold")
                
                if disk > self.critical_thresholds['disk_percent']:
                    alerts.append(f"CRITICAL: Disk usage {disk:.1f}% exceeds critical threshold")
                elif disk > self.warning_thresholds['disk_percent']:
                    alerts.append(f"WARNING: Disk usage {disk:.1f}% exceeds warning threshold")
            
            # Check component health alerts | 檢查組件健康警報
            components = health_report.get('component_health', {})
            for component_name, component_data in components.items():
                status = component_data.get('status', 'unknown')
                message = component_data.get('message', '')
                
                if status == 'critical':
                    alerts.append(f"CRITICAL: {component_name.upper()} - {message}")
                elif status == 'warning':
                    alerts.append(f"WARNING: {component_name.upper()} - {message}")
            
        except Exception as e:
            alerts.append(f"ERROR: Failed to generate alerts - {str(e)}")
        
        return alerts
    
    def save_health_report(self, health_report: Dict[str, Any]):
        """Save health report to storage | 保存健康報告到存儲"""
        try:
            # Save to Redis if available | 如可用則保存到Redis
            if self.redis_client:
                key = f"health_report:{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                self.redis_client.setex(key, 3600, json.dumps(health_report))  # TTL: 1 hour
                
                # Keep only last 24 reports | 僅保留最後24份報告
                self.redis_client.ltrim("health_reports_list", 0, 23)
                self.redis_client.lpush("health_reports_list", key)
            
            # Save to file | 保存到文件
            reports_dir = Path("/app/logs/health_reports")
            reports_dir.mkdir(exist_ok=True)
            
            report_file = reports_dir / f"health_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(health_report, f, indent=2, default=str)
            
            logger.info(f"💾 Health report saved to {report_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save health report: {e}")

# ============================================================================
# MAIN EXECUTION | 主執行
# ============================================================================

def main():
    """Main health monitoring loop | 主健康監控循環"""
    monitor = AIFXHealthMonitor()
    
    logger.info("🚀 AIFX Health Monitor started")
    
    try:
        while True:
            # Run comprehensive health check | 運行綜合健康檢查
            health_report = monitor.run_comprehensive_health_check()
            
            # Save health report | 保存健康報告
            monitor.save_health_report(health_report)
            
            # Print summary | 打印摘要
            status = health_report['overall_status']
            alerts_count = len(health_report['alerts'])
            
            if status == 'healthy':
                logger.info(f"💚 System Status: HEALTHY - No issues detected")
            elif status == 'warning':
                logger.warning(f"⚠️ System Status: WARNING - {alerts_count} warnings")
            else:
                logger.error(f"🚨 System Status: CRITICAL - {alerts_count} critical issues")
            
            # Print alerts | 打印警報
            for alert in health_report['alerts']:
                logger.warning(f"🔔 ALERT: {alert}")
            
            # Wait for next check | 等待下次檢查
            time.sleep(monitor.health_check_interval)
            
    except KeyboardInterrupt:
        logger.info("🛑 Health monitor stopped by user")
    except Exception as e:
        logger.error(f"❌ Health monitor error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()