# -*- coding: utf-8 -*-
"""
AIFX Real-time Data Pipeline - Orchestrator
AIFX 實時數據管道 - 編排器

This module orchestrates the complete real-time forex data pipeline integrating
all components: data ingestion, validation, processing, database, failover, and monitoring.
該模組編排完整的實時外匯數據管道，整合所有組件：
數據攝取、驗證、處理、數據庫、故障轉移和監控。

Author: AIFX Development Team
Created: 2025-01-14
Version: 1.0.0
"""

import asyncio
import logging
import time
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import json
import yaml
from enum import Enum

# Import pipeline components
from .realtime_feed import ForexTick, DataSource, RealTimeForexFeed, create_realtime_forex_feed
from .stream_processor import StreamProcessor, create_stream_processor, ValidationResult
from .database_integration import DatabaseIntegrationManager, create_database_integration_manager
from .failover_manager import FailoverManager, create_failover_manager, SourceStatus
from .performance_test import PerformanceTestSuite, create_performance_test_suite


class PipelineStatus(Enum):
    """Pipeline operational status"""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DEGRADED = "DEGRADED"
    ERROR = "ERROR"
    STOPPING = "STOPPING"


@dataclass
class PipelineMetrics:
    """Comprehensive pipeline metrics"""
    pipeline_status: PipelineStatus = PipelineStatus.STOPPED
    start_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    
    # Data flow metrics
    total_ticks_ingested: int = 0
    total_ticks_processed: int = 0
    total_ticks_stored: int = 0
    
    # Performance metrics
    ingestion_rate_per_sec: float = 0.0
    processing_rate_per_sec: float = 0.0
    storage_rate_per_sec: float = 0.0
    end_to_end_latency_ms: float = 0.0
    
    # Quality metrics
    data_validation_rate: float = 1.0
    data_quality_score: str = "EXCELLENT"
    
    # Source metrics
    active_sources: int = 0
    healthy_sources: int = 0
    failed_sources: int = 0
    
    # System health
    database_status: str = "UNKNOWN"
    cache_status: str = "UNKNOWN"
    failover_events: int = 0
    
    def calculate_throughput(self):
        """Calculate throughput rates"""
        if self.uptime_seconds > 0:
            self.ingestion_rate_per_sec = self.total_ticks_ingested / self.uptime_seconds
            self.processing_rate_per_sec = self.total_ticks_processed / self.uptime_seconds
            self.storage_rate_per_sec = self.total_ticks_stored / self.uptime_seconds


class PipelineOrchestrator:
    """
    Main orchestrator for AIFX real-time forex data pipeline
    AIFX實時外匯數據管道的主要編排器
    """
    
    def __init__(self, config_path: str = "config/data-sources.yaml"):
        """Initialize pipeline orchestrator"""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
        # Pipeline components
        self.realtime_feed: Optional[RealTimeForexFeed] = None
        self.stream_processor: Optional[StreamProcessor] = None
        self.database_manager: Optional[DatabaseIntegrationManager] = None
        self.failover_manager: Optional[FailoverManager] = None
        
        # Pipeline state
        self.metrics = PipelineMetrics()
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Performance testing
        self.performance_tester: Optional[PerformanceTestSuite] = None
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {
            'tick_received': [],
            'tick_processed': [],
            'tick_stored': [],
            'validation_failed': [],
            'quality_alert': [],
            'failover_event': [],
            'error_occurred': []
        }
        
        # Monitoring threads
        self.monitoring_threads: List[threading.Thread] = []
        
        self.logger.info("Pipeline orchestrator initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return {}
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register event handler for pipeline events"""
        if event_type in self.event_handlers:
            self.event_handlers[event_type].append(handler)
            self.logger.info(f"Registered handler for {event_type}: {handler.__name__}")
    
    async def initialize_pipeline(self) -> bool:
        """Initialize all pipeline components"""
        try:
            self.logger.info("Initializing AIFX real-time data pipeline components...")
            
            # Initialize database manager
            self.logger.info("Initializing database integration manager...")
            self.database_manager = create_database_integration_manager(self.config_path)
            
            # Initialize failover manager
            self.logger.info("Initializing failover manager...")
            self.failover_manager = create_failover_manager(self.config_path)
            
            # Register failover callbacks
            self.failover_manager.register_failover_callback(self._handle_failover_event)
            self.failover_manager.register_recovery_callback(self._handle_source_recovery)
            
            # Initialize stream processor
            self.logger.info("Initializing stream processor...")
            self.stream_processor = create_stream_processor(self.config_path)
            
            # Add pipeline processors
            self.stream_processor.add_processor(self._process_validated_tick)
            self.stream_processor.add_quality_alert_handler(self._handle_quality_alert)
            
            # Initialize real-time feed
            self.logger.info("Initializing real-time forex feed...")
            self.realtime_feed = create_realtime_forex_feed(self.config_path)
            
            # Connect components
            self.realtime_feed.add_data_subscriber(self._handle_incoming_tick)
            
            # Initialize performance testing
            self.performance_tester = create_performance_test_suite()
            
            self.logger.info("All pipeline components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {e}")
            return False
    
    async def start_pipeline(self, sources: Optional[List[DataSource]] = None) -> bool:
        """Start the complete data pipeline"""
        try:
            self.logger.info("Starting AIFX real-time data pipeline...")
            self.metrics.pipeline_status = PipelineStatus.STARTING
            
            # Initialize components if not already done
            if not await self.initialize_pipeline():
                return False
            
            # Start components in order
            self.logger.info("Starting stream processor...")
            self.stream_processor.start(sources)
            
            self.logger.info("Starting real-time feed...")
            self.realtime_feed.start(sources)
            
            # Start monitoring
            self._start_monitoring()
            
            # Update status
            self.metrics.pipeline_status = PipelineStatus.RUNNING
            self.metrics.start_time = datetime.now()
            self.is_running = True
            
            self.logger.info("AIFX real-time data pipeline started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            self.metrics.pipeline_status = PipelineStatus.ERROR
            return False
    
    async def stop_pipeline(self) -> bool:
        """Stop the complete data pipeline"""
        try:
            self.logger.info("Stopping AIFX real-time data pipeline...")
            self.metrics.pipeline_status = PipelineStatus.STOPPING
            self.is_running = False
            self.shutdown_event.set()
            
            # Stop components in reverse order
            if self.realtime_feed:
                self.realtime_feed.stop()
            
            if self.stream_processor:
                self.stream_processor.stop()
            
            if self.database_manager:
                self.database_manager.shutdown()
            
            if self.failover_manager:
                self.failover_manager.shutdown()
            
            # Wait for monitoring threads
            for thread in self.monitoring_threads:
                thread.join(timeout=10)
            
            self.metrics.pipeline_status = PipelineStatus.STOPPED
            self.logger.info("AIFX real-time data pipeline stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping pipeline: {e}")
            return False
    
    def _handle_incoming_tick(self, tick: ForexTick):
        """Handle incoming tick from real-time feed"""
        try:
            self.metrics.total_ticks_ingested += 1
            
            # Report to failover manager for health monitoring
            self.failover_manager.report_source_event(
                tick.source.value, 
                True, 
                0.0,  # Latency will be calculated elsewhere
                1.0   # Quality will be determined by validation
            )
            
            # Emit tick received event
            self._emit_event('tick_received', tick)
            
        except Exception as e:
            self.logger.error(f"Error handling incoming tick: {e}")
            self._emit_event('error_occurred', {'error': str(e), 'context': 'incoming_tick'})
    
    async def _process_validated_tick(self, tick: ForexTick):
        """Process validated tick through database storage"""
        try:
            process_start = time.time()
            
            # Store in database
            success = await self.database_manager.process_tick(tick)
            
            if success:
                self.metrics.total_ticks_processed += 1
                self.metrics.total_ticks_stored += 1
                
                # Calculate end-to-end latency
                if hasattr(tick, 'ingestion_time'):
                    e2e_latency = (time.time() - tick.ingestion_time) * 1000
                    self.metrics.end_to_end_latency_ms = e2e_latency
                
                # Emit events
                self._emit_event('tick_processed', tick)
                self._emit_event('tick_stored', tick)
            else:
                self.logger.warning(f"Failed to store tick: {tick.symbol}")
                self._emit_event('error_occurred', {
                    'error': 'Failed to store tick',
                    'context': 'database_storage',
                    'tick': tick.to_dict()
                })
                
        except Exception as e:
            self.logger.error(f"Error processing validated tick: {e}")
            self._emit_event('error_occurred', {'error': str(e), 'context': 'tick_processing'})
    
    def _handle_quality_alert(self, alert_type: str, alert_data: Any):
        """Handle data quality alerts"""
        self.logger.warning(f"Quality alert: {alert_type} - {alert_data}")
        self._emit_event('quality_alert', {'type': alert_type, 'data': alert_data})
        
        # Update pipeline status if quality is critical
        if 'critical' in alert_type.lower() or 'poor' in alert_type.lower():
            self.metrics.pipeline_status = PipelineStatus.DEGRADED
    
    def _handle_failover_event(self, failover_event):
        """Handle failover events"""
        self.logger.warning(f"Failover executed: {failover_event.from_source} -> {failover_event.to_source}")
        self.metrics.failover_events += 1
        self._emit_event('failover_event', failover_event)
        
        # Update pipeline status
        self.metrics.pipeline_status = PipelineStatus.DEGRADED
    
    def _handle_source_recovery(self, recovered_source: str):
        """Handle source recovery events"""
        self.logger.info(f"Source recovered: {recovered_source}")
        
        # Check if pipeline can return to normal status
        healthy_sources = len(self.failover_manager.health_monitor.get_healthy_sources())
        if healthy_sources > 1:  # Multiple healthy sources
            self.metrics.pipeline_status = PipelineStatus.RUNNING
    
    def _emit_event(self, event_type: str, event_data: Any):
        """Emit event to registered handlers"""
        for handler in self.event_handlers.get(event_type, []):
            try:
                handler(event_data)
            except Exception as e:
                self.logger.error(f"Error in event handler {handler.__name__}: {e}")
    
    def _start_monitoring(self):
        """Start monitoring threads"""
        # Metrics collection thread
        metrics_thread = threading.Thread(
            target=self._metrics_collection_loop,
            name="MetricsCollector",
            daemon=True
        )
        metrics_thread.start()
        self.monitoring_threads.append(metrics_thread)
        
        # Health check thread
        health_thread = threading.Thread(
            target=self._health_check_loop,
            name="HealthChecker",
            daemon=True
        )
        health_thread.start()
        self.monitoring_threads.append(health_thread)
        
        self.logger.info("Pipeline monitoring started")
    
    def _metrics_collection_loop(self):
        """Collect and update metrics periodically"""
        while not self.shutdown_event.is_set():
            try:
                if self.metrics.start_time:
                    self.metrics.uptime_seconds = (datetime.now() - self.metrics.start_time).total_seconds()
                    self.metrics.calculate_throughput()
                
                # Collect component metrics
                if self.stream_processor:
                    processing_status = self.stream_processor.get_processing_status()
                    validation = processing_status.get('data_validation', {})
                    self.metrics.data_validation_rate = validation.get('validation_rate', 1.0)
                    self.metrics.data_quality_score = validation.get('quality_score', 'UNKNOWN')
                
                if self.failover_manager:
                    health_report = self.failover_manager.get_health_report()
                    self.metrics.active_sources = len(health_report.get('active_sources', []))
                    
                    # Count healthy vs failed sources
                    sources = health_report.get('sources', {})
                    healthy_count = sum(1 for s in sources.values() if s.get('status') == 'active')
                    failed_count = sum(1 for s in sources.values() if s.get('status') == 'failed')
                    
                    self.metrics.healthy_sources = healthy_count
                    self.metrics.failed_sources = failed_count
                
                if self.database_manager:
                    system_metrics = self.database_manager.get_system_metrics()
                    self.metrics.database_status = "CONNECTED" if system_metrics else "DISCONNECTED"
                    redis_info = system_metrics.get('redis', {})
                    self.metrics.cache_status = "CONNECTED" if redis_info.get('connected') else "DISCONNECTED"
                
                # Sleep for next collection
                self.shutdown_event.wait(10)  # Collect every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                self.shutdown_event.wait(5)
    
    def _health_check_loop(self):
        """Perform periodic health checks"""
        while not self.shutdown_event.is_set():
            try:
                # Check pipeline health
                self._perform_health_check()
                
                # Sleep for next check
                self.shutdown_event.wait(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in health check: {e}")
                self.shutdown_event.wait(10)
    
    def _perform_health_check(self):
        """Perform comprehensive health check"""
        health_issues = []
        
        # Check if we're receiving data
        if self.metrics.total_ticks_ingested == 0 and self.metrics.uptime_seconds > 60:
            health_issues.append("No data ingested after 60 seconds")
        
        # Check processing lag
        processing_lag = self.metrics.total_ticks_ingested - self.metrics.total_ticks_processed
        if processing_lag > 1000:  # More than 1000 ticks behind
            health_issues.append(f"High processing lag: {processing_lag} ticks")
        
        # Check end-to-end latency
        if self.metrics.end_to_end_latency_ms > 100:  # More than 100ms
            health_issues.append(f"High end-to-end latency: {self.metrics.end_to_end_latency_ms:.2f}ms")
        
        # Check data quality
        if self.metrics.data_validation_rate < 0.95:  # Less than 95%
            health_issues.append(f"Low data quality: {self.metrics.data_validation_rate:.1%}")
        
        # Check source health
        if self.metrics.healthy_sources == 0:
            health_issues.append("No healthy data sources")
        elif self.metrics.healthy_sources < 2:
            health_issues.append("Only one healthy data source (no redundancy)")
        
        # Update pipeline status based on health
        if health_issues:
            if len(health_issues) >= 3 or "No healthy data sources" in health_issues:
                self.metrics.pipeline_status = PipelineStatus.ERROR
            else:
                self.metrics.pipeline_status = PipelineStatus.DEGRADED
            
            self.logger.warning(f"Health check issues: {', '.join(health_issues)}")
        elif self.metrics.pipeline_status == PipelineStatus.DEGRADED:
            # Recovery to normal status
            self.metrics.pipeline_status = PipelineStatus.RUNNING
            self.logger.info("Pipeline health restored to normal")
    
    async def run_performance_test(self) -> Dict[str, Any]:
        """Run comprehensive performance test"""
        if not self.performance_tester:
            self.performance_tester = create_performance_test_suite()
        
        self.logger.info("Running performance validation tests...")
        
        try:
            # Run performance tests
            results = await self.performance_tester.run_full_performance_test()
            
            # Generate reports
            report_file, chart_file = self.performance_tester.generate_reports()
            
            # Validate requirements
            validation = self.performance_tester.validate_requirements()
            
            performance_summary = {
                'test_results': {name: metrics.to_dict() for name, metrics in results.items()},
                'validation': validation,
                'reports': {
                    'text_report': report_file,
                    'chart_file': chart_file
                }
            }
            
            self.logger.info(f"Performance test completed. Overall: {'PASS' if validation['overall_pass'] else 'FAIL'}")
            
            return performance_summary
            
        except Exception as e:
            self.logger.error(f"Performance test failed: {e}")
            return {'error': str(e)}
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status"""
        status = {
            'pipeline': {
                'status': self.metrics.pipeline_status.value,
                'uptime_seconds': self.metrics.uptime_seconds,
                'start_time': self.metrics.start_time.isoformat() if self.metrics.start_time else None
            },
            'data_flow': {
                'ticks_ingested': self.metrics.total_ticks_ingested,
                'ticks_processed': self.metrics.total_ticks_processed,
                'ticks_stored': self.metrics.total_ticks_stored,
                'ingestion_rate_per_sec': round(self.metrics.ingestion_rate_per_sec, 2),
                'processing_rate_per_sec': round(self.metrics.processing_rate_per_sec, 2),
                'storage_rate_per_sec': round(self.metrics.storage_rate_per_sec, 2)
            },
            'performance': {
                'end_to_end_latency_ms': round(self.metrics.end_to_end_latency_ms, 2),
                'data_validation_rate': round(self.metrics.data_validation_rate, 3),
                'data_quality_score': self.metrics.data_quality_score
            },
            'sources': {
                'active_sources': self.metrics.active_sources,
                'healthy_sources': self.metrics.healthy_sources,
                'failed_sources': self.metrics.failed_sources,
                'failover_events': self.metrics.failover_events
            },
            'infrastructure': {
                'database_status': self.metrics.database_status,
                'cache_status': self.metrics.cache_status
            }
        }
        
        # Add component-specific status
        if self.realtime_feed:
            status['realtime_feed'] = self.realtime_feed.get_connection_status()
        
        if self.stream_processor:
            status['stream_processor'] = self.stream_processor.get_processing_status()
        
        if self.failover_manager:
            status['failover_manager'] = self.failover_manager.get_health_report()
        
        if self.database_manager:
            status['database_manager'] = self.database_manager.get_system_metrics()
        
        return status
    
    def get_symbol_statistics(self, symbol: str) -> Dict[str, Any]:
        """Get detailed statistics for a specific symbol"""
        stats = {
            'symbol': symbol,
            'pipeline_metrics': {
                'status': self.metrics.pipeline_status.value,
                'uptime_seconds': self.metrics.uptime_seconds
            }
        }
        
        # Get latest tick data
        if self.realtime_feed:
            latest_tick = self.realtime_feed.get_latest_tick(symbol)
            if latest_tick:
                stats['latest_tick'] = latest_tick.to_dict()
        
        # Get quality report
        if self.stream_processor:
            quality_report = self.stream_processor.get_symbol_quality_report(symbol)
            stats['quality_report'] = quality_report
        
        # Get database statistics
        if self.database_manager:
            try:
                recent_data = asyncio.run(self.database_manager.get_latest_data(symbol, 10))
                stats['recent_data_points'] = len(recent_data)
            except Exception as e:
                stats['database_error'] = str(e)
        
        return stats


# Factory function
def create_pipeline_orchestrator(config_path: str = "config/data-sources.yaml") -> PipelineOrchestrator:
    """Create and configure pipeline orchestrator"""
    return PipelineOrchestrator(config_path)


# Context manager for pipeline lifecycle
@asynccontextmanager
async def pipeline_context(config_path: str = "config/data-sources.yaml", 
                          sources: Optional[List[DataSource]] = None):
    """Context manager for pipeline lifecycle management"""
    orchestrator = create_pipeline_orchestrator(config_path)
    
    try:
        # Start pipeline
        success = await orchestrator.start_pipeline(sources)
        if not success:
            raise RuntimeError("Failed to start pipeline")
        
        yield orchestrator
        
    finally:
        # Ensure cleanup
        await orchestrator.stop_pipeline()


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    async def main():
        """Main test function"""
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger = logging.getLogger(__name__)
        
        try:
            # Test pipeline lifecycle
            logger.info("Testing AIFX Real-time Data Pipeline Orchestrator")
            
            # Create orchestrator
            orchestrator = create_pipeline_orchestrator()
            
            # Example event handlers
            def on_tick_received(tick):
                logger.debug(f"Tick received: {tick.symbol} - {tick.mid_price:.5f}")
            
            def on_quality_alert(alert):
                logger.warning(f"Quality alert: {alert}")
            
            def on_failover_event(event):
                logger.error(f"Failover: {event.from_source} -> {event.to_source}")
            
            # Register handlers
            orchestrator.register_event_handler('tick_received', on_tick_received)
            orchestrator.register_event_handler('quality_alert', on_quality_alert)
            orchestrator.register_event_handler('failover_event', on_failover_event)
            
            # Start pipeline
            logger.info("Starting pipeline...")
            success = await orchestrator.start_pipeline([DataSource.YAHOO])
            
            if not success:
                logger.error("Failed to start pipeline")
                return
            
            # Monitor for a few minutes
            for i in range(12):  # 2 minutes (12 * 10 seconds)
                await asyncio.sleep(10)
                
                # Get status
                status = orchestrator.get_pipeline_status()
                
                logger.info(f"Pipeline Status - Iteration {i+1}")
                logger.info(f"  Status: {status['pipeline']['status']}")
                logger.info(f"  Ticks Ingested: {status['data_flow']['ticks_ingested']}")
                logger.info(f"  Ticks Processed: {status['data_flow']['ticks_processed']}")
                logger.info(f"  Processing Rate: {status['data_flow']['processing_rate_per_sec']:.2f}/sec")
                logger.info(f"  Data Quality: {status['performance']['data_quality_score']}")
                logger.info(f"  Healthy Sources: {status['sources']['healthy_sources']}/{status['sources']['active_sources']}")
            
            # Run performance test
            logger.info("Running performance validation...")
            perf_results = await orchestrator.run_performance_test()
            
            if 'validation' in perf_results:
                validation = perf_results['validation']
                logger.info(f"Performance Test Results:")
                logger.info(f"  Latency <50ms: {'PASS' if validation['latency_50ms'] else 'FAIL'}")
                logger.info(f"  Throughput >100 ops/sec: {'PASS' if validation['throughput_100ops'] else 'FAIL'}")
                logger.info(f"  Success Rate >95%: {'PASS' if validation['success_rate_95pct'] else 'FAIL'}")
                logger.info(f"  Overall: {'PASS' if validation['overall_pass'] else 'FAIL'}")
            
            # Stop pipeline
            logger.info("Stopping pipeline...")
            await orchestrator.stop_pipeline()
            
            logger.info("Pipeline orchestrator test completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline test failed: {e}")
            sys.exit(1)
    
    # Run test
    asyncio.run(main())