# -*- coding: utf-8 -*-
"""
AIFX Real-time Data Pipeline - Failover Manager
AIFX 實時數據管道 - 故障轉移管理器

This module implements automated failover and recovery mechanisms for forex data sources.
該模組實現外匯數據源的自動故障轉移和恢復機制。

Author: AIFX Development Team
Created: 2025-01-14
Version: 1.0.0
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
import yaml
import statistics
from threading import Lock, Thread
import json


class SourceStatus(Enum):
    """Data source connection status"""
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    DISABLED = "disabled"


class FailoverTrigger(Enum):
    """Triggers that can cause failover"""
    CONNECTION_TIMEOUT = "connection_timeout"
    DATA_QUALITY_POOR = "data_quality_poor"
    CONSECUTIVE_FAILURES = "consecutive_failures"
    LATENCY_THRESHOLD = "latency_threshold"
    MANUAL_OVERRIDE = "manual_override"
    HEALTH_CHECK_FAILED = "health_check_failed"


@dataclass
class SourceHealth:
    """Health metrics for a data source"""
    source_id: str
    status: SourceStatus = SourceStatus.ACTIVE
    last_update: datetime = field(default_factory=datetime.now)
    consecutive_failures: int = 0
    total_failures: int = 0
    success_count: int = 0
    avg_latency_ms: float = 0.0
    data_quality_score: float = 1.0
    uptime_percentage: float = 100.0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    recovery_attempts: int = 0
    
    def success_rate(self) -> float:
        """Calculate success rate"""
        total = self.success_count + self.total_failures
        return (self.success_count / total * 100) if total > 0 else 100.0


@dataclass
class FailoverEvent:
    """Represents a failover event"""
    timestamp: datetime
    from_source: str
    to_source: str
    trigger: FailoverTrigger
    reason: str
    recovery_time_ms: Optional[float] = None
    affected_symbols: List[str] = field(default_factory=list)


class HealthMonitor:
    """Monitors health of data sources"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.health_metrics: Dict[str, SourceHealth] = {}
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.quality_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self.lock = Lock()
        self.logger = logging.getLogger(__name__)
        
        # Configuration thresholds
        self.max_latency_ms = config.get('max_latency_ms', 5000)
        self.min_quality_score = config.get('min_quality_score', 0.8)
        self.max_consecutive_failures = config.get('consecutive_failures_limit', 3)
        self.health_check_interval = config.get('health_check_interval', 60)
        
    def update_health(self, source_id: str, latency_ms: float, 
                     quality_score: float, success: bool) -> None:
        """Update health metrics for a source"""
        with self.lock:
            if source_id not in self.health_metrics:
                self.health_metrics[source_id] = SourceHealth(source_id=source_id)
            
            health = self.health_metrics[source_id]
            health.last_update = datetime.now()
            
            # Update latency
            self.latency_history[source_id].append(latency_ms)
            if self.latency_history[source_id]:
                health.avg_latency_ms = statistics.mean(self.latency_history[source_id])
            
            # Update quality
            self.quality_history[source_id].append(quality_score)
            if self.quality_history[source_id]:
                health.data_quality_score = statistics.mean(self.quality_history[source_id])
            
            # Update success/failure counts
            if success:
                health.success_count += 1
                health.consecutive_failures = 0
                health.last_success_time = datetime.now()
                
                # Potentially recover from degraded state
                if health.status == SourceStatus.DEGRADED:
                    if self._should_recover(health):
                        health.status = SourceStatus.ACTIVE
                        self.logger.info(f"Source {source_id} recovered to ACTIVE status")
                        
            else:
                health.total_failures += 1
                health.consecutive_failures += 1
                health.last_failure_time = datetime.now()
                
                # Update status based on failures
                self._update_status_on_failure(health)
            
            # Calculate uptime
            self._update_uptime(health)
    
    def _should_recover(self, health: SourceHealth) -> bool:
        """Determine if a source should recover from degraded state"""
        # Recent success rate should be high
        recent_success_rate = health.success_rate()
        if recent_success_rate < 95:
            return False
            
        # Latency should be acceptable
        if health.avg_latency_ms > self.max_latency_ms:
            return False
            
        # Quality should be good
        if health.data_quality_score < self.min_quality_score:
            return False
            
        # No recent failures
        if health.consecutive_failures > 0:
            return False
            
        return True
    
    def _update_status_on_failure(self, health: SourceHealth) -> None:
        """Update source status based on failure patterns"""
        if health.consecutive_failures >= self.max_consecutive_failures:
            health.status = SourceStatus.FAILED
            self.logger.warning(f"Source {health.source_id} marked as FAILED")
        elif (health.avg_latency_ms > self.max_latency_ms or 
              health.data_quality_score < self.min_quality_score):
            if health.status == SourceStatus.ACTIVE:
                health.status = SourceStatus.DEGRADED
                self.logger.warning(f"Source {health.source_id} marked as DEGRADED")
    
    def _update_uptime(self, health: SourceHealth) -> None:
        """Calculate and update uptime percentage"""
        total_requests = health.success_count + health.total_failures
        if total_requests > 0:
            health.uptime_percentage = (health.success_count / total_requests) * 100
    
    def get_healthy_sources(self) -> List[str]:
        """Get list of healthy (active or degraded) sources"""
        with self.lock:
            return [
                source_id for source_id, health in self.health_metrics.items()
                if health.status in [SourceStatus.ACTIVE, SourceStatus.DEGRADED]
            ]
    
    def get_source_health(self, source_id: str) -> Optional[SourceHealth]:
        """Get health metrics for a specific source"""
        with self.lock:
            return self.health_metrics.get(source_id)


class FailoverManager:
    """Manages automated failover and recovery for forex data sources"""
    
    def __init__(self, config_path: str = "config/data-sources.yaml"):
        """Initialize failover manager"""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize health monitor
        self.health_monitor = HealthMonitor(self.config.get('data_quality', {}))
        
        # Source configuration
        self.sources = self.config.get('forex_data_sources', {})
        self.priority_order = self.config.get('failover', {}).get('priority_order', [])
        self.current_primary = None
        self.active_sources: Set[str] = set()
        
        # Failover configuration
        failover_config = self.config.get('failover', {})
        self.auto_failover_enabled = failover_config.get('enable_auto_failover', True)
        self.failover_timeout = failover_config.get('failover_timeout', 30)
        self.cross_validation = failover_config.get('cross_validation', True)
        self.max_price_deviation = failover_config.get('max_price_deviation', 0.001)
        
        # Event tracking
        self.failover_events: List[FailoverEvent] = []
        self.recovery_callbacks: List[Callable] = []
        self.failover_callbacks: List[Callable] = []
        
        # Recovery mechanism
        self.recovery_thread = None
        self.recovery_active = False
        
        # Initialize primary source
        self._initialize_primary_source()
        
        self.logger.info("FailoverManager initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config from {self.config_path}: {e}")
            return {}
    
    def _initialize_primary_source(self) -> None:
        """Initialize the primary data source based on priority"""
        enabled_sources = [
            source_id for source_id, config in self.sources.items()
            if config.get('enabled', False)
        ]
        
        if not enabled_sources:
            raise ValueError("No enabled data sources found")
        
        # Find highest priority enabled source
        for source_id in self.priority_order:
            if source_id in enabled_sources:
                self.current_primary = source_id
                self.active_sources.add(source_id)
                break
        
        if not self.current_primary:
            # Fallback to first enabled source
            self.current_primary = enabled_sources[0]
            self.active_sources.add(self.current_primary)
        
        self.logger.info(f"Primary source initialized: {self.current_primary}")
    
    def register_recovery_callback(self, callback: Callable) -> None:
        """Register callback to be called on source recovery"""
        self.recovery_callbacks.append(callback)
    
    def register_failover_callback(self, callback: Callable) -> None:
        """Register callback to be called on failover"""
        self.failover_callbacks.append(callback)
    
    def report_source_event(self, source_id: str, success: bool, 
                           latency_ms: float, quality_score: float = 1.0) -> None:
        """Report a data event from a source for health monitoring"""
        self.health_monitor.update_health(source_id, latency_ms, quality_score, success)
        
        # Check if failover is needed
        if not success and source_id == self.current_primary:
            self._check_failover_needed(source_id, FailoverTrigger.CONSECUTIVE_FAILURES)
    
    def _check_failover_needed(self, source_id: str, trigger: FailoverTrigger) -> bool:
        """Check if failover is needed and execute if necessary"""
        if not self.auto_failover_enabled:
            return False
        
        health = self.health_monitor.get_source_health(source_id)
        if not health:
            return False
        
        # Determine if failover is needed based on trigger
        needs_failover = False
        reason = ""
        
        if trigger == FailoverTrigger.CONSECUTIVE_FAILURES:
            if health.consecutive_failures >= self.health_monitor.max_consecutive_failures:
                needs_failover = True
                reason = f"Consecutive failures: {health.consecutive_failures}"
        
        elif trigger == FailoverTrigger.LATENCY_THRESHOLD:
            if health.avg_latency_ms > self.health_monitor.max_latency_ms:
                needs_failover = True
                reason = f"High latency: {health.avg_latency_ms}ms"
        
        elif trigger == FailoverTrigger.DATA_QUALITY_POOR:
            if health.data_quality_score < self.health_monitor.min_quality_score:
                needs_failover = True
                reason = f"Poor quality: {health.data_quality_score}"
        
        if needs_failover and source_id == self.current_primary:
            backup_source = self._find_best_backup_source()
            if backup_source:
                self._execute_failover(source_id, backup_source, trigger, reason)
                return True
        
        return False
    
    def _find_best_backup_source(self) -> Optional[str]:
        """Find the best backup source for failover"""
        healthy_sources = self.health_monitor.get_healthy_sources()
        
        # Remove current primary from options
        backup_candidates = [s for s in healthy_sources if s != self.current_primary]
        
        if not backup_candidates:
            return None
        
        # Find best backup based on priority order
        for source_id in self.priority_order:
            if source_id in backup_candidates:
                return source_id
        
        # Fallback to first healthy source
        return backup_candidates[0] if backup_candidates else None
    
    def _execute_failover(self, from_source: str, to_source: str, 
                         trigger: FailoverTrigger, reason: str) -> None:
        """Execute failover to backup source"""
        start_time = time.time()
        
        try:
            # Update current primary
            old_primary = self.current_primary
            self.current_primary = to_source
            self.active_sources.add(to_source)
            
            # Create failover event
            event = FailoverEvent(
                timestamp=datetime.now(),
                from_source=from_source,
                to_source=to_source,
                trigger=trigger,
                reason=reason,
                recovery_time_ms=(time.time() - start_time) * 1000
            )
            self.failover_events.append(event)
            
            # Notify callbacks
            for callback in self.failover_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Failover callback error: {e}")
            
            self.logger.warning(
                f"FAILOVER EXECUTED: {from_source} → {to_source} "
                f"(Trigger: {trigger.value}, Reason: {reason})"
            )
            
            # Start recovery monitoring for failed source
            self._start_recovery_monitoring(from_source)
            
        except Exception as e:
            self.logger.error(f"Failover execution failed: {e}")
            # Rollback if possible
            self.current_primary = old_primary
    
    def _start_recovery_monitoring(self, failed_source: str) -> None:
        """Start monitoring failed source for recovery"""
        if self.recovery_thread and self.recovery_thread.is_alive():
            return
        
        self.recovery_active = True
        self.recovery_thread = Thread(
            target=self._recovery_monitor_loop,
            args=(failed_source,),
            daemon=True
        )
        self.recovery_thread.start()
    
    def _recovery_monitor_loop(self, failed_source: str) -> None:
        """Monitor failed source for recovery"""
        self.logger.info(f"Starting recovery monitoring for {failed_source}")
        
        while self.recovery_active:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                health = self.health_monitor.get_source_health(failed_source)
                if not health:
                    continue
                
                # Check if source has recovered
                if self._check_source_recovery(failed_source, health):
                    self.logger.info(f"Source {failed_source} has recovered")
                    
                    # Notify recovery callbacks
                    for callback in self.recovery_callbacks:
                        try:
                            callback(failed_source)
                        except Exception as e:
                            self.logger.error(f"Recovery callback error: {e}")
                    
                    # Optionally fail back to recovered source if higher priority
                    if self._should_failback_to_recovered(failed_source):
                        self._execute_failover(
                            self.current_primary, 
                            failed_source,
                            FailoverTrigger.MANUAL_OVERRIDE,
                            "Failback to higher priority recovered source"
                        )
                    
                    break
                    
            except Exception as e:
                self.logger.error(f"Recovery monitoring error: {e}")
        
        self.recovery_active = False
    
    def _check_source_recovery(self, source_id: str, health: SourceHealth) -> bool:
        """Check if a failed source has recovered"""
        # Must have recent successful connections
        if not health.last_success_time:
            return False
        
        # Recent success within last 5 minutes
        if (datetime.now() - health.last_success_time).total_seconds() > 300:
            return False
        
        # No consecutive failures
        if health.consecutive_failures > 0:
            return False
        
        # Acceptable latency
        if health.avg_latency_ms > self.health_monitor.max_latency_ms:
            return False
        
        # Good quality score
        if health.data_quality_score < self.health_monitor.min_quality_score:
            return False
        
        return True
    
    def _should_failback_to_recovered(self, recovered_source: str) -> bool:
        """Determine if should failback to recovered source"""
        if recovered_source not in self.priority_order:
            return False
        
        current_priority = self.priority_order.index(self.current_primary) if self.current_primary in self.priority_order else float('inf')
        recovered_priority = self.priority_order.index(recovered_source)
        
        # Only failback if recovered source has higher priority
        return recovered_priority < current_priority
    
    def force_failover(self, to_source: str, reason: str = "Manual override") -> bool:
        """Manually force failover to specific source"""
        if to_source not in self.sources:
            self.logger.error(f"Unknown source: {to_source}")
            return False
        
        if not self.sources[to_source].get('enabled', False):
            self.logger.error(f"Source not enabled: {to_source}")
            return False
        
        self._execute_failover(
            self.current_primary,
            to_source,
            FailoverTrigger.MANUAL_OVERRIDE,
            reason
        )
        return True
    
    def get_current_primary(self) -> str:
        """Get current primary data source"""
        return self.current_primary
    
    def get_active_sources(self) -> Set[str]:
        """Get all active data sources"""
        return self.active_sources.copy()
    
    def get_failover_history(self, limit: int = 50) -> List[FailoverEvent]:
        """Get recent failover events"""
        return self.failover_events[-limit:]
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        report = {
            'current_primary': self.current_primary,
            'active_sources': list(self.active_sources),
            'auto_failover_enabled': self.auto_failover_enabled,
            'total_failovers': len(self.failover_events),
            'sources': {}
        }
        
        for source_id in self.sources:
            health = self.health_monitor.get_source_health(source_id)
            if health:
                report['sources'][source_id] = {
                    'status': health.status.value,
                    'success_rate': round(health.success_rate(), 2),
                    'avg_latency_ms': round(health.avg_latency_ms, 2),
                    'quality_score': round(health.data_quality_score, 3),
                    'uptime_percentage': round(health.uptime_percentage, 2),
                    'consecutive_failures': health.consecutive_failures,
                    'last_update': health.last_update.isoformat() if health.last_update else None
                }
        
        return report
    
    def shutdown(self) -> None:
        """Shutdown failover manager"""
        self.recovery_active = False
        if self.recovery_thread and self.recovery_thread.is_alive():
            self.recovery_thread.join(timeout=5)
        
        self.logger.info("FailoverManager shutdown complete")


class CircuitBreaker:
    """Circuit breaker pattern implementation for data sources"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'closed'  # closed, open, half-open
        self.logger = logging.getLogger(__name__)
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == 'open':
            if self._should_attempt_reset():
                self.state = 'half-open'
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if should attempt to reset circuit breaker"""
        if not self.last_failure_time:
            return True
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self) -> None:
        """Handle successful call"""
        self.failure_count = 0
        if self.state == 'half-open':
            self.state = 'closed'
            self.logger.info("Circuit breaker reset to CLOSED")
    
    def _on_failure(self) -> None:
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold and self.state == 'closed':
            self.state = 'open'
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


# Factory function for creating failover manager
def create_failover_manager(config_path: str = "config/data-sources.yaml") -> FailoverManager:
    """Create and configure failover manager"""
    return FailoverManager(config_path)


if __name__ == "__main__":
    # Example usage and testing
    import sys
    import os
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create failover manager
    try:
        manager = create_failover_manager()
        
        # Simulate some events
        print("\n=== Failover Manager Health Report ===")
        print(json.dumps(manager.get_health_report(), indent=2))
        
        # Test manual failover
        print(f"\nCurrent primary: {manager.get_current_primary()}")
        
        # Simulate source events
        manager.report_source_event("yahoo", True, 150.0, 0.95)
        manager.report_source_event("oanda", True, 45.0, 0.98)
        
        print("\n=== Updated Health Report ===")
        print(json.dumps(manager.get_health_report(), indent=2))
        
        # Simulate failures
        for i in range(4):  # This should trigger failover
            manager.report_source_event("yahoo", False, 5000.0, 0.3)
        
        print(f"\nAfter failures - Current primary: {manager.get_current_primary()}")
        print(f"Failover events: {len(manager.get_failover_history())}")
        
        manager.shutdown()
        print("\nFailover manager test completed successfully")
        
    except Exception as e:
        print(f"Error testing failover manager: {e}")
        sys.exit(1)