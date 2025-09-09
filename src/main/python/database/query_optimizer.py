"""
AIFX Database Query Optimizer | AIFX 資料庫查詢優化器
Advanced query optimization and indexing strategy for high-frequency trading
針對高頻交易的高級查詢優化和索引策略

Phase 4.1.4 Database Optimization Component
第四階段 4.1.4 資料庫優化組件
"""

import time
import logging
import hashlib
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager
from enum import Enum

from sqlalchemy import text, Index, Column, create_engine
from sqlalchemy.orm import Session
from sqlalchemy.sql import select, func
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

# Logger setup | 日誌設置
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query type classification | 查詢類型分類"""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    ANALYTICAL = "ANALYTICAL"
    TRADING = "TRADING"


class IndexType(Enum):
    """Index type classification | 索引類型分類"""
    BTREE = "BTREE"
    HASH = "HASH"
    GIN = "GIN"
    GIST = "GIST"
    PARTIAL = "PARTIAL"
    COMPOSITE = "COMPOSITE"


@dataclass
class QueryMetrics:
    """
    Query performance metrics | 查詢性能指標
    """
    query_hash: str
    query_type: QueryType
    execution_time: float
    rows_examined: int
    rows_returned: int
    cpu_time: float
    io_operations: int
    cache_hits: int
    cache_misses: int
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Performance ratios | 性能比率
    @property
    def efficiency_ratio(self) -> float:
        """Query efficiency ratio | 查詢效率比率"""
        if self.rows_examined == 0:
            return 1.0
        return self.rows_returned / self.rows_examined
    
    @property
    def cache_hit_ratio(self) -> float:
        """Cache hit ratio | 緩存命中率"""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total


@dataclass
class IndexRecommendation:
    """
    Index recommendation | 索引建議
    """
    table_name: str
    columns: List[str]
    index_type: IndexType
    estimated_benefit: float
    priority: str  # HIGH, MEDIUM, LOW
    reason: str
    creation_cost: float
    maintenance_cost: float
    storage_overhead_mb: float


class DatabaseQueryOptimizer:
    """
    Advanced database query optimizer | 高級資料庫查詢優化器
    """
    
    def __init__(self, connection_manager):
        """
        Initialize query optimizer | 初始化查詢優化器
        
        Args:
            connection_manager: Database connection manager | 資料庫連接管理器
        """
        self.connection_manager = connection_manager
        self.query_metrics: Dict[str, List[QueryMetrics]] = {}
        self.slow_query_threshold = 1.0  # seconds
        self.index_recommendations: List[IndexRecommendation] = []
        
        logger.info("Database query optimizer initialized")
    
    def analyze_query_performance(self, query: str, params: Dict = None,
                                 enable_profiling: bool = True) -> QueryMetrics:
        """
        Analyze query performance with detailed metrics | 分析查詢性能並提供詳細指標
        """
        query_hash = self._hash_query(query)
        start_time = time.time()
        
        try:
            with self.connection_manager.get_session() as session:
                # Enable query profiling if supported | 如果支持，啟用查詢分析
                if enable_profiling:
                    self._enable_query_profiling(session)
                
                # Execute query | 執行查詢
                result = session.execute(text(query), params or {})
                rows = result.fetchall()
                
                # Get query execution statistics | 獲取查詢執行統計
                stats = self._get_query_statistics(session, query)
                
                execution_time = time.time() - start_time
                
                # Create metrics object | 創建指標對象
                metrics = QueryMetrics(
                    query_hash=query_hash,
                    query_type=self._classify_query(query),
                    execution_time=execution_time,
                    rows_examined=stats.get('rows_examined', 0),
                    rows_returned=len(rows),
                    cpu_time=stats.get('cpu_time', 0),
                    io_operations=stats.get('io_operations', 0),
                    cache_hits=stats.get('cache_hits', 0),
                    cache_misses=stats.get('cache_misses', 0)
                )
                
                # Store metrics for analysis | 存儲指標用於分析
                if query_hash not in self.query_metrics:
                    self.query_metrics[query_hash] = []
                self.query_metrics[query_hash].append(metrics)
                
                # Log slow queries | 記錄慢查詢
                if execution_time > self.slow_query_threshold:
                    logger.warning(
                        f"Slow query detected: {execution_time:.3f}s - "
                        f"Efficiency: {metrics.efficiency_ratio:.3f} - "
                        f"Query: {query[:100]}..."
                    )
                
                return metrics
                
        except SQLAlchemyError as e:
            logger.error(f"Query analysis failed: {e}")
            raise
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query caching | 生成查詢緩存的哈希值"""
        # Normalize query for consistent hashing | 標準化查詢以進行一致哈希
        normalized = ' '.join(query.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _classify_query(self, query: str) -> QueryType:
        """Classify query type | 分類查詢類型"""
        query_lower = query.lower().strip()
        
        if query_lower.startswith('select'):
            # Check for analytical patterns | 檢查分析模式
            if any(keyword in query_lower for keyword in ['group by', 'having', 'window']):
                return QueryType.ANALYTICAL
            # Check for trading patterns | 檢查交易模式
            elif any(keyword in query_lower for keyword in ['trading_data', 'positions', 'orders']):
                return QueryType.TRADING
            else:
                return QueryType.SELECT
        elif query_lower.startswith('insert'):
            return QueryType.INSERT
        elif query_lower.startswith('update'):
            return QueryType.UPDATE
        elif query_lower.startswith('delete'):
            return QueryType.DELETE
        else:
            return QueryType.SELECT
    
    def _enable_query_profiling(self, session: Session):
        """Enable database query profiling | 啟用資料庫查詢分析"""
        try:
            # PostgreSQL specific profiling | PostgreSQL特定分析
            session.execute(text("SET track_io_timing = ON"))
            session.execute(text("SET log_min_duration_statement = 0"))
            session.execute(text("SET log_statement_stats = ON"))
        except SQLAlchemyError:
            # Profiling may not be available | 分析可能不可用
            pass
    
    def _get_query_statistics(self, session: Session, query: str) -> Dict[str, Any]:
        """Get detailed query execution statistics | 獲取詳細查詢執行統計"""
        stats = {
            'rows_examined': 0,
            'cpu_time': 0,
            'io_operations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        try:
            # Get PostgreSQL statistics | 獲取PostgreSQL統計
            result = session.execute(text(
                "SELECT * FROM pg_stat_statements "
                "WHERE query LIKE :query_pattern "
                "ORDER BY last_exec DESC LIMIT 1"
            ), {'query_pattern': f"{query[:50]}%"})
            
            row = result.fetchone()
            if row:
                stats.update({
                    'rows_examined': row.rows if hasattr(row, 'rows') else 0,
                    'cpu_time': row.total_time if hasattr(row, 'total_time') else 0,
                    'io_operations': (
                        getattr(row, 'shared_blks_read', 0) + 
                        getattr(row, 'local_blks_read', 0)
                    ),
                    'cache_hits': getattr(row, 'shared_blks_hit', 0),
                    'cache_misses': getattr(row, 'shared_blks_read', 0)
                })
        except SQLAlchemyError:
            # Statistics may not be available | 統計可能不可用
            pass
        
        return stats
    
    def analyze_trading_query_patterns(self) -> Dict[str, Any]:
        """
        Analyze trading-specific query patterns | 分析交易特定查詢模式
        """
        trading_queries = [
            metrics for query_list in self.query_metrics.values()
            for metrics in query_list
            if metrics.query_type in [QueryType.TRADING, QueryType.SELECT]
        ]
        
        if not trading_queries:
            return {"message": "No trading queries to analyze"}
        
        # Calculate aggregated metrics | 計算聚合指標
        total_queries = len(trading_queries)
        avg_execution_time = sum(m.execution_time for m in trading_queries) / total_queries
        slow_queries = [m for m in trading_queries if m.execution_time > self.slow_query_threshold]
        
        # Efficiency analysis | 效率分析
        efficiency_scores = [m.efficiency_ratio for m in trading_queries if m.efficiency_ratio > 0]
        avg_efficiency = sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0
        
        # Cache performance | 緩存性能
        cache_ratios = [m.cache_hit_ratio for m in trading_queries]
        avg_cache_hit_ratio = sum(cache_ratios) / len(cache_ratios) if cache_ratios else 0
        
        return {
            "total_trading_queries": total_queries,
            "average_execution_time": avg_execution_time,
            "slow_query_count": len(slow_queries),
            "slow_query_percentage": (len(slow_queries) / total_queries) * 100,
            "average_efficiency_ratio": avg_efficiency,
            "average_cache_hit_ratio": avg_cache_hit_ratio,
            "recommendations": self._generate_trading_optimizations(trading_queries)
        }
    
    def _generate_trading_optimizations(self, metrics: List[QueryMetrics]) -> List[str]:
        """Generate trading-specific optimization recommendations | 生成交易特定優化建議"""
        recommendations = []
        
        # Analyze slow queries | 分析慢查詢
        slow_queries = [m for m in metrics if m.execution_time > self.slow_query_threshold]
        if slow_queries:
            recommendations.append(
                f"Optimize {len(slow_queries)} slow trading queries "
                f"(avg time: {sum(m.execution_time for m in slow_queries) / len(slow_queries):.3f}s)"
            )
        
        # Analyze efficiency | 分析效率
        inefficient_queries = [m for m in metrics if m.efficiency_ratio < 0.1]
        if inefficient_queries:
            recommendations.append(
                f"Improve {len(inefficient_queries)} inefficient queries "
                f"with low selectivity ratios"
            )
        
        # Cache performance | 緩存性能
        low_cache_queries = [m for m in metrics if m.cache_hit_ratio < 0.8]
        if low_cache_queries:
            recommendations.append(
                f"Optimize caching for {len(low_cache_queries)} queries "
                f"with low cache hit ratios"
            )
        
        return recommendations
    
    def recommend_indexes(self) -> List[IndexRecommendation]:
        """
        Generate index recommendations based on query patterns | 根據查詢模式生成索引建議
        """
        recommendations = []
        
        # Analyze frequent query patterns | 分析頻繁查詢模式
        common_patterns = self._identify_common_query_patterns()
        
        for pattern in common_patterns:
            # Trading data time-based queries | 交易數據基於時間的查詢
            if pattern['type'] == 'time_range_trading':
                recommendations.append(IndexRecommendation(
                    table_name="trading_data_eurusd",
                    columns=["datetime", "symbol"],
                    index_type=IndexType.BTREE,
                    estimated_benefit=0.85,
                    priority="HIGH",
                    reason="Frequent time-based trading data queries",
                    creation_cost=2.5,
                    maintenance_cost=0.1,
                    storage_overhead_mb=50.0
                ))
            
            # Signal lookup optimization | 信號查找優化
            elif pattern['type'] == 'signal_lookup':
                recommendations.append(IndexRecommendation(
                    table_name="trading_signals",
                    columns=["symbol", "signal_type", "confidence"],
                    index_type=IndexType.COMPOSITE,
                    estimated_benefit=0.75,
                    priority="HIGH",
                    reason="Frequent signal filtering and sorting",
                    creation_cost=1.8,
                    maintenance_cost=0.15,
                    storage_overhead_mb=30.0
                ))
            
            # Performance analytics | 性能分析
            elif pattern['type'] == 'performance_analytics':
                recommendations.append(IndexRecommendation(
                    table_name="model_performance",
                    columns=["model_name", "evaluation_date"],
                    index_type=IndexType.BTREE,
                    estimated_benefit=0.65,
                    priority="MEDIUM",
                    reason="Analytics queries for model performance",
                    creation_cost=1.2,
                    maintenance_cost=0.08,
                    storage_overhead_mb=20.0
                ))
        
        # Hash indexes for exact match queries | 精確匹配查詢的哈希索引
        recommendations.append(IndexRecommendation(
            table_name="system_config",
            columns=["config_key"],
            index_type=IndexType.HASH,
            estimated_benefit=0.90,
            priority="MEDIUM",
            reason="Fast config lookups",
            creation_cost=0.5,
            maintenance_cost=0.02,
            storage_overhead_mb=5.0
        ))
        
        # Partial indexes for active positions | 活躍倉位的部分索引
        recommendations.append(IndexRecommendation(
            table_name="positions",
            columns=["status", "updated_at"],
            index_type=IndexType.PARTIAL,
            estimated_benefit=0.70,
            priority="HIGH",
            reason="Active position tracking (WHERE status = 'ACTIVE')",
            creation_cost=1.0,
            maintenance_cost=0.05,
            storage_overhead_mb=15.0
        ))
        
        self.index_recommendations = recommendations
        return recommendations
    
    def _identify_common_query_patterns(self) -> List[Dict[str, Any]]:
        """Identify common query patterns from metrics | 從指標中識別常見查詢模式"""
        patterns = []
        
        # Simulate common trading query patterns | 模擬常見交易查詢模式
        # In production, this would analyze actual query logs | 在生產中，這將分析實際查詢日誌
        patterns.extend([
            {
                'type': 'time_range_trading',
                'frequency': 150,  # queries per minute
                'avg_execution_time': 0.25,
                'table': 'trading_data_eurusd'
            },
            {
                'type': 'signal_lookup',
                'frequency': 80,
                'avg_execution_time': 0.15,
                'table': 'trading_signals'
            },
            {
                'type': 'performance_analytics',
                'frequency': 20,
                'avg_execution_time': 0.45,
                'table': 'model_performance'
            }
        ])
        
        return patterns
    
    def create_recommended_indexes(self, execute: bool = False) -> List[str]:
        """
        Create recommended indexes | 創建推薦的索引
        
        Args:
            execute: Whether to actually create the indexes | 是否真正創建索引
        """
        if not self.index_recommendations:
            self.recommend_indexes()
        
        creation_statements = []
        
        for rec in self.index_recommendations:
            if rec.priority == "HIGH":
                # Generate CREATE INDEX statement | 生成CREATE INDEX語句
                index_name = f"idx_{rec.table_name}_{'_'.join(rec.columns)}"
                columns_str = ', '.join(rec.columns)
                
                if rec.index_type == IndexType.BTREE:
                    stmt = f"CREATE INDEX {index_name} ON {rec.table_name} USING BTREE ({columns_str})"
                elif rec.index_type == IndexType.HASH:
                    stmt = f"CREATE INDEX {index_name} ON {rec.table_name} USING HASH ({columns_str})"
                elif rec.index_type == IndexType.COMPOSITE:
                    stmt = f"CREATE INDEX {index_name} ON {rec.table_name} ({columns_str})"
                elif rec.index_type == IndexType.PARTIAL:
                    if rec.table_name == "positions":
                        stmt = f"CREATE INDEX {index_name} ON {rec.table_name} ({columns_str}) WHERE status = 'ACTIVE'"
                    else:
                        stmt = f"CREATE INDEX {index_name} ON {rec.table_name} ({columns_str})"
                else:
                    stmt = f"CREATE INDEX {index_name} ON {rec.table_name} ({columns_str})"
                
                creation_statements.append(stmt)
                
                if execute:
                    try:
                        with self.connection_manager.get_session() as session:
                            session.execute(text(stmt))
                            logger.info(f"Index created successfully: {index_name}")
                    except SQLAlchemyError as e:
                        logger.error(f"Failed to create index {index_name}: {e}")
        
        return creation_statements
    
    def optimize_trading_queries(self) -> Dict[str, Any]:
        """
        Optimize common trading queries | 優化常見交易查詢
        """
        optimizations = {
            "query_rewrites": [],
            "configuration_changes": [],
            "index_suggestions": []
        }
        
        # Common trading query optimizations | 常見交易查詢優化
        
        # 1. Time-series data queries | 時間序列數據查詢
        optimizations["query_rewrites"].append({
            "description": "Use partitioning for time-series data",
            "before": "SELECT * FROM trading_data_eurusd WHERE datetime BETWEEN ? AND ?",
            "after": "Use table partitioning by month/week for better performance",
            "benefit": "50-80% improvement for time-range queries"
        })
        
        # 2. Signal filtering | 信號過濾
        optimizations["query_rewrites"].append({
            "description": "Optimize signal confidence filtering",
            "before": "SELECT * FROM trading_signals WHERE confidence > 0.7 ORDER BY confidence DESC",
            "after": "CREATE INDEX ON trading_signals(confidence DESC) WHERE confidence > 0.7",
            "benefit": "60% improvement for high-confidence signal queries"
        })
        
        # 3. Configuration optimization | 配置優化
        optimizations["configuration_changes"].extend([
            {
                "setting": "shared_buffers",
                "recommended_value": "25% of available RAM",
                "reason": "Increase buffer cache for frequently accessed trading data"
            },
            {
                "setting": "work_mem",
                "recommended_value": "256MB",
                "reason": "Optimize sorting and hashing operations for analytics"
            },
            {
                "setting": "effective_cache_size",
                "recommended_value": "75% of available RAM",
                "reason": "Help query planner make better decisions"
            },
            {
                "setting": "checkpoint_timeout",
                "recommended_value": "15min",
                "reason": "Balance between crash recovery and I/O performance"
            }
        ])
        
        # 4. Index suggestions from analysis | 分析的索引建議
        optimizations["index_suggestions"] = [
            {
                "table": rec.table_name,
                "columns": rec.columns,
                "type": rec.index_type.value,
                "priority": rec.priority,
                "benefit": f"{rec.estimated_benefit:.1%} improvement"
            }
            for rec in self.index_recommendations
            if rec.estimated_benefit > 0.5
        ]
        
        return optimizations
    
    def generate_performance_report(self) -> str:
        """
        Generate comprehensive performance report | 生成綜合性能報告
        """
        if not self.query_metrics:
            return "No query metrics available for analysis"
        
        all_metrics = [metric for metrics_list in self.query_metrics.values() 
                      for metric in metrics_list]
        
        # Basic statistics | 基本統計
        total_queries = len(all_metrics)
        avg_execution_time = sum(m.execution_time for m in all_metrics) / total_queries
        slow_queries = [m for m in all_metrics if m.execution_time > self.slow_query_threshold]
        
        # Query type breakdown | 查詢類型分解
        type_counts = {}
        for metric in all_metrics:
            type_counts[metric.query_type.value] = type_counts.get(metric.query_type.value, 0) + 1
        
        # Generate report | 生成報告
        report = f"""
AIFX Database Performance Analysis Report | AIFX 資料庫性能分析報告
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== OVERALL STATISTICS | 總體統計 ===
Total Queries Analyzed: {total_queries}
Average Execution Time: {avg_execution_time:.3f} seconds
Slow Queries (>{self.slow_query_threshold}s): {len(slow_queries)} ({len(slow_queries)/total_queries*100:.1f}%)

=== QUERY TYPE BREAKDOWN | 查詢類型分解 ===
"""
        
        for query_type, count in sorted(type_counts.items()):
            percentage = (count / total_queries) * 100
            report += f"{query_type}: {count} queries ({percentage:.1f}%)\n"
        
        # Top slow queries | 最慢查詢
        if slow_queries:
            sorted_slow = sorted(slow_queries, key=lambda x: x.execution_time, reverse=True)
            report += f"\n=== SLOWEST QUERIES | 最慢查詢 ===\n"
            for i, metric in enumerate(sorted_slow[:5], 1):
                report += f"{i}. {metric.execution_time:.3f}s - Efficiency: {metric.efficiency_ratio:.3f}\n"
        
        # Index recommendations | 索引建議
        if self.index_recommendations:
            report += f"\n=== INDEX RECOMMENDATIONS | 索引建議 ===\n"
            for rec in sorted(self.index_recommendations, 
                            key=lambda x: x.estimated_benefit, reverse=True)[:5]:
                report += (f"• {rec.table_name}.{rec.columns} ({rec.index_type.value}) "
                          f"- {rec.estimated_benefit:.1%} benefit - {rec.priority} priority\n")
        
        return report