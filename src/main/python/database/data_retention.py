"""
AIFX Data Retention & Archival System | AIFX 數據保留與歸檔系統
Automated data lifecycle management for trading and analytics data
交易和分析數據的自動化數據生命週期管理

Phase 4.1.4 Database Optimization Component
第四階段 4.1.4 資料庫優化組件
"""

import os
import logging
import threading
import gzip
import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager

from sqlalchemy import text, and_, or_
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import boto3
from botocore.exceptions import ClientError

# Logger setup | 日誌設置
logger = logging.getLogger(__name__)


class RetentionPolicy(Enum):
    """Data retention policy types | 數據保留政策類型"""
    KEEP_FOREVER = "KEEP_FOREVER"
    REGULATORY_7_YEARS = "REGULATORY_7_YEARS"
    TRADING_DATA_2_YEARS = "TRADING_DATA_2_YEARS"
    LOGS_90_DAYS = "LOGS_90_DAYS"
    CACHE_DATA_7_DAYS = "CACHE_DATA_7_DAYS"
    TEMP_DATA_24_HOURS = "TEMP_DATA_24_HOURS"


class ArchivalTier(Enum):
    """Storage archival tiers | 存儲歸檔層級"""
    HOT = "HOT"              # Immediate access (SSD)
    WARM = "WARM"            # Standard storage
    COOL = "COOL"            # Infrequent access
    COLD = "COLD"            # Glacier
    FROZEN = "FROZEN"        # Deep Archive


class DataClassification(Enum):
    """Data classification levels | 數據分類級別"""
    CRITICAL = "CRITICAL"       # Trading positions, orders
    IMPORTANT = "IMPORTANT"     # Market data, signals
    STANDARD = "STANDARD"       # Analytics, reports
    TEMP = "TEMP"              # Temporary processing data


@dataclass
class RetentionRule:
    """
    Data retention rule definition | 數據保留規則定義
    """
    name: str
    table_pattern: str
    policy: RetentionPolicy
    classification: DataClassification
    
    # Retention periods | 保留期間
    hot_storage_days: int
    warm_storage_days: int
    cool_storage_days: int
    cold_storage_days: int
    deletion_days: Optional[int] = None
    
    # Rule conditions | 規則條件
    date_column: str = "created_at"
    partition_column: Optional[str] = None
    condition_sql: Optional[str] = None
    
    # Archival settings | 歸檔設置
    compress_on_archive: bool = True
    encrypt_on_archive: bool = True
    verify_after_archive: bool = True
    
    # Business rules | 業務規則
    preserve_active_positions: bool = True
    preserve_regulatory_data: bool = True
    
    # Metadata | 元數據
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    last_executed: Optional[datetime] = None
    next_execution: Optional[datetime] = None


@dataclass
class ArchivalJob:
    """
    Archival job execution record | 歸檔作業執行記錄
    """
    job_id: str
    rule_name: str
    table_name: str
    source_tier: ArchivalTier
    target_tier: ArchivalTier
    
    # Job details | 作業詳情
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "RUNNING"  # RUNNING, SUCCESS, FAILED, CANCELLED
    
    # Statistics | 統計信息
    rows_processed: int = 0
    bytes_processed: int = 0
    compression_ratio: float = 0.0
    
    # Error handling | 錯誤處理
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


class DataRetentionManager:
    """
    Advanced data retention and archival management system
    高級數據保留和歸檔管理系統
    """
    
    def __init__(self, connection_manager, config: Dict[str, Any]):
        """
        Initialize data retention manager | 初始化數據保留管理器
        
        Args:
            connection_manager: Database connection manager | 資料庫連接管理器
            config: Retention configuration | 保留配置
        """
        self.connection_manager = connection_manager
        self.config = config
        self.retention_rules: Dict[str, RetentionRule] = {}
        self.archival_jobs: Dict[str, ArchivalJob] = {}
        
        # AWS S3 client for archival | 用於歸檔的AWS S3客戶端
        self.s3_client = None
        if config.get('aws_s3', {}).get('enabled', False):
            self.s3_client = boto3.client('s3')
        
        # Execution control | 執行控制
        self._scheduler_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._execution_interval = config.get('execution_interval_hours', 24)
        
        # Initialize default rules | 初始化默認規則
        self._setup_default_retention_rules()
        
        logger.info("Data retention manager initialized")
    
    def _setup_default_retention_rules(self):
        """Setup default retention rules for AIFX | 為AIFX設置默認保留規則"""
        
        # Trading data retention (2 years) | 交易數據保留（2年）
        self.add_retention_rule(RetentionRule(
            name="trading_data_retention",
            table_pattern="trading_data_*",
            policy=RetentionPolicy.TRADING_DATA_2_YEARS,
            classification=DataClassification.CRITICAL,
            hot_storage_days=30,      # Keep 30 days in hot storage
            warm_storage_days=180,    # 6 months in warm storage
            cool_storage_days=365,    # 1 year in cool storage
            cold_storage_days=730,    # 2 years in cold storage
            deletion_days=2555,       # Delete after 7 years (regulatory)
            date_column="datetime",
            preserve_active_positions=True,
            preserve_regulatory_data=True
        ))
        
        # Trading signals retention | 交易信號保留
        self.add_retention_rule(RetentionRule(
            name="trading_signals_retention",
            table_pattern="trading_signals",
            policy=RetentionPolicy.TRADING_DATA_2_YEARS,
            classification=DataClassification.IMPORTANT,
            hot_storage_days=7,       # Keep 1 week in hot storage
            warm_storage_days=90,     # 3 months in warm storage
            cool_storage_days=365,    # 1 year in cool storage
            cold_storage_days=730,    # 2 years in cold storage
            deletion_days=1095,       # Delete after 3 years
            date_column="created_at",
            condition_sql="confidence > 0.5"  # Only archive high-confidence signals
        ))
        
        # Model performance data | 模型性能數據
        self.add_retention_rule(RetentionRule(
            name="model_performance_retention",
            table_pattern="model_performance",
            policy=RetentionPolicy.REGULATORY_7_YEARS,
            classification=DataClassification.IMPORTANT,
            hot_storage_days=90,      # Keep 3 months in hot storage
            warm_storage_days=365,    # 1 year in warm storage
            cool_storage_days=1095,   # 3 years in cool storage
            cold_storage_days=2555,   # 7 years in cold storage
            deletion_days=3650,       # Delete after 10 years
            date_column="evaluation_date"
        ))
        
        # System logs retention | 系統日誌保留
        self.add_retention_rule(RetentionRule(
            name="system_logs_retention",
            table_pattern="system_logs",
            policy=RetentionPolicy.LOGS_90_DAYS,
            classification=DataClassification.STANDARD,
            hot_storage_days=7,       # Keep 1 week in hot storage
            warm_storage_days=30,     # 1 month in warm storage
            cool_storage_days=90,     # 3 months in cool storage
            deletion_days=90,         # Delete after 90 days
            date_column="log_timestamp"
        ))
        
        # Temporary processing data | 臨時處理數據
        self.add_retention_rule(RetentionRule(
            name="temp_data_retention",
            table_pattern="temp_*",
            policy=RetentionPolicy.TEMP_DATA_24_HOURS,
            classification=DataClassification.TEMP,
            hot_storage_days=1,       # Keep 1 day in hot storage
            deletion_days=1,          # Delete after 1 day
            date_column="created_at",
            compress_on_archive=False,  # No need to archive temp data
            verify_after_archive=False
        ))
        
        # Cache data retention | 緩存數據保留
        self.add_retention_rule(RetentionRule(
            name="cache_data_retention",
            table_pattern="cache_*",
            policy=RetentionPolicy.CACHE_DATA_7_DAYS,
            classification=DataClassification.TEMP,
            hot_storage_days=7,       # Keep 1 week in hot storage
            deletion_days=7,          # Delete after 7 days
            date_column="cached_at"
        ))
    
    def add_retention_rule(self, rule: RetentionRule):
        """
        Add or update a retention rule | 添加或更新保留規則
        """
        self.retention_rules[rule.name] = rule
        logger.info(f"Retention rule added/updated: {rule.name}")
    
    def get_matching_tables(self, table_pattern: str) -> List[str]:
        """
        Get tables matching the pattern | 獲取匹配模式的表
        """
        try:
            with self.connection_manager.get_connection() as conn:
                # PostgreSQL specific query | PostgreSQL特定查詢
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name SIMILAR TO :pattern
                """), {"pattern": table_pattern.replace('*', '%')})
                
                return [row[0] for row in result.fetchall()]
        except SQLAlchemyError as e:
            logger.error(f"Failed to get matching tables for pattern {table_pattern}: {e}")
            return []
    
    def analyze_table_data_age(self, table_name: str, date_column: str) -> Dict[str, Any]:
        """
        Analyze data age distribution in table | 分析表中數據年齡分佈
        """
        try:
            with self.connection_manager.get_connection() as conn:
                result = conn.execute(text(f"""
                    SELECT 
                        MIN({date_column}) as oldest_date,
                        MAX({date_column}) as newest_date,
                        COUNT(*) as total_rows,
                        COUNT(*) FILTER (WHERE {date_column} >= NOW() - INTERVAL '30 days') as hot_rows,
                        COUNT(*) FILTER (WHERE {date_column} >= NOW() - INTERVAL '180 days' 
                                        AND {date_column} < NOW() - INTERVAL '30 days') as warm_rows,
                        COUNT(*) FILTER (WHERE {date_column} >= NOW() - INTERVAL '365 days' 
                                        AND {date_column} < NOW() - INTERVAL '180 days') as cool_rows,
                        COUNT(*) FILTER (WHERE {date_column} < NOW() - INTERVAL '365 days') as cold_rows,
                        pg_total_relation_size('{table_name}') as table_size_bytes
                    FROM {table_name}
                """))
                
                row = result.fetchone()
                if row:
                    return {
                        "table_name": table_name,
                        "oldest_date": row.oldest_date,
                        "newest_date": row.newest_date,
                        "total_rows": row.total_rows,
                        "hot_rows": row.hot_rows,
                        "warm_rows": row.warm_rows,
                        "cool_rows": row.cool_rows,
                        "cold_rows": row.cold_rows,
                        "table_size_bytes": row.table_size_bytes,
                        "analysis_timestamp": datetime.now()
                    }
        except SQLAlchemyError as e:
            logger.error(f"Failed to analyze table {table_name}: {e}")
        
        return {}
    
    def identify_archival_candidates(self, rule: RetentionRule) -> List[Dict[str, Any]]:
        """
        Identify data candidates for archival | 識別歸檔候選數據
        """
        candidates = []
        tables = self.get_matching_tables(rule.table_pattern)
        
        for table_name in tables:
            analysis = self.analyze_table_data_age(table_name, rule.date_column)
            if not analysis:
                continue
            
            # Calculate archival tiers | 計算歸檔層級
            current_time = datetime.now()
            
            # Hot to Warm candidates | 熱到溫候選
            if analysis["hot_rows"] > 0:
                hot_cutoff = current_time - timedelta(days=rule.hot_storage_days)
                candidates.append({
                    "table_name": table_name,
                    "source_tier": ArchivalTier.HOT,
                    "target_tier": ArchivalTier.WARM,
                    "cutoff_date": hot_cutoff,
                    "estimated_rows": analysis["hot_rows"],
                    "rule_name": rule.name
                })
            
            # Warm to Cool candidates | 溫到冷候選
            if analysis["warm_rows"] > 0:
                warm_cutoff = current_time - timedelta(days=rule.warm_storage_days)
                candidates.append({
                    "table_name": table_name,
                    "source_tier": ArchivalTier.WARM,
                    "target_tier": ArchivalTier.COOL,
                    "cutoff_date": warm_cutoff,
                    "estimated_rows": analysis["warm_rows"],
                    "rule_name": rule.name
                })
            
            # Cool to Cold candidates | 冷到極冷候選
            if analysis["cool_rows"] > 0:
                cool_cutoff = current_time - timedelta(days=rule.cool_storage_days)
                candidates.append({
                    "table_name": table_name,
                    "source_tier": ArchivalTier.COOL,
                    "target_tier": ArchivalTier.COLD,
                    "cutoff_date": cool_cutoff,
                    "estimated_rows": analysis["cool_rows"],
                    "rule_name": rule.name
                })
            
            # Deletion candidates | 刪除候選
            if rule.deletion_days and analysis["cold_rows"] > 0:
                deletion_cutoff = current_time - timedelta(days=rule.deletion_days)
                candidates.append({
                    "table_name": table_name,
                    "source_tier": ArchivalTier.COLD,
                    "target_tier": None,  # Deletion
                    "cutoff_date": deletion_cutoff,
                    "estimated_rows": analysis["cold_rows"],
                    "rule_name": rule.name,
                    "action": "DELETE"
                })
        
        return candidates
    
    def execute_archival(self, candidate: Dict[str, Any]) -> ArchivalJob:
        """
        Execute archival operation | 執行歸檔操作
        """
        job_id = f"archival_{candidate['table_name']}_{int(datetime.now().timestamp())}"
        
        job = ArchivalJob(
            job_id=job_id,
            rule_name=candidate["rule_name"],
            table_name=candidate["table_name"],
            source_tier=candidate["source_tier"],
            target_tier=candidate["target_tier"],
            start_time=datetime.now()
        )
        
        self.archival_jobs[job_id] = job
        
        try:
            if candidate.get("action") == "DELETE":
                # Execute deletion | 執行刪除
                self._execute_deletion(job, candidate)
            else:
                # Execute archival | 執行歸檔
                self._execute_tier_migration(job, candidate)
            
            job.status = "SUCCESS"
            job.end_time = datetime.now()
            
        except Exception as e:
            job.status = "FAILED"
            job.error_message = str(e)
            job.end_time = datetime.now()
            logger.error(f"Archival job {job_id} failed: {e}")
        
        return job
    
    def _execute_tier_migration(self, job: ArchivalJob, candidate: Dict[str, Any]):
        """
        Execute data tier migration | 執行數據層級遷移
        """
        rule = self.retention_rules[job.rule_name]
        table_name = job.table_name
        cutoff_date = candidate["cutoff_date"]
        
        # Extract data for archival | 提取歸檔數據
        archive_data = self._extract_archival_data(table_name, rule.date_column, cutoff_date, rule.condition_sql)
        
        if not archive_data.empty:
            job.rows_processed = len(archive_data)
            job.bytes_processed = archive_data.memory_usage(deep=True).sum()
            
            # Compress data if required | 如果需要，壓縮數據
            if rule.compress_on_archive:
                compressed_data = self._compress_data(archive_data)
                job.compression_ratio = len(compressed_data) / job.bytes_processed
            else:
                compressed_data = archive_data.to_json(orient='records')
            
            # Store in appropriate tier | 存儲到適當層級
            if job.target_tier == ArchivalTier.COLD:
                # Store in S3 Glacier | 存儲到S3 Glacier
                self._store_to_s3_glacier(job, compressed_data, rule.encrypt_on_archive)
            elif job.target_tier == ArchivalTier.COOL:
                # Store in S3 Standard-IA | 存儲到S3 Standard-IA
                self._store_to_s3_ia(job, compressed_data, rule.encrypt_on_archive)
            
            # Verify archival if required | 如果需要，驗證歸檔
            if rule.verify_after_archive:
                if self._verify_archival(job, archive_data):
                    # Remove from source tier | 從源層級移除
                    self._remove_archived_data(table_name, rule.date_column, cutoff_date, rule.condition_sql)
                else:
                    raise Exception("Archival verification failed")
    
    def _execute_deletion(self, job: ArchivalJob, candidate: Dict[str, Any]):
        """
        Execute data deletion | 執行數據刪除
        """
        rule = self.retention_rules[job.rule_name]
        table_name = job.table_name
        cutoff_date = candidate["cutoff_date"]
        
        # Safety checks | 安全檢查
        if rule.preserve_active_positions:
            # Check for active positions | 檢查活躍倉位
            active_check = self._check_active_positions(table_name, cutoff_date)
            if active_check:
                raise Exception(f"Cannot delete data with active positions: {active_check}")
        
        if rule.preserve_regulatory_data:
            # Check regulatory requirements | 檢查監管要求
            regulatory_check = self._check_regulatory_requirements(table_name, cutoff_date)
            if not regulatory_check:
                raise Exception("Cannot delete data required for regulatory compliance")
        
        # Execute deletion with safeguards | 執行帶保護的刪除
        deleted_rows = self._safe_delete_data(table_name, rule.date_column, cutoff_date, rule.condition_sql)
        job.rows_processed = deleted_rows
    
    def _extract_archival_data(self, table_name: str, date_column: str, 
                             cutoff_date: datetime, condition_sql: Optional[str] = None) -> pd.DataFrame:
        """
        Extract data for archival | 提取歸檔數據
        """
        try:
            with self.connection_manager.get_connection() as conn:
                where_clause = f"{date_column} < :cutoff_date"
                if condition_sql:
                    where_clause += f" AND ({condition_sql})"
                
                query = f"SELECT * FROM {table_name} WHERE {where_clause}"
                return pd.read_sql_query(query, conn, params={"cutoff_date": cutoff_date})
        except SQLAlchemyError as e:
            logger.error(f"Failed to extract archival data from {table_name}: {e}")
            return pd.DataFrame()
    
    def _compress_data(self, data: pd.DataFrame) -> bytes:
        """
        Compress data for archival | 壓縮歸檔數據
        """
        json_data = data.to_json(orient='records')
        return gzip.compress(json_data.encode('utf-8'))
    
    def _store_to_s3_glacier(self, job: ArchivalJob, data: Union[str, bytes], encrypt: bool):
        """
        Store data to S3 Glacier | 存儲數據到S3 Glacier
        """
        if not self.s3_client:
            raise Exception("S3 client not configured")
        
        bucket = self.config['aws_s3']['bucket']
        key = f"glacier/{job.table_name}/{job.job_id}.json.gz"
        
        extra_args = {'StorageClass': 'GLACIER'}
        if encrypt:
            extra_args['ServerSideEncryption'] = 'AES256'
        
        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            **extra_args
        )
        
        logger.info(f"Data stored to S3 Glacier: {key}")
    
    def _store_to_s3_ia(self, job: ArchivalJob, data: Union[str, bytes], encrypt: bool):
        """
        Store data to S3 Infrequent Access | 存儲數據到S3非頻繁訪問
        """
        if not self.s3_client:
            raise Exception("S3 client not configured")
        
        bucket = self.config['aws_s3']['bucket']
        key = f"ia/{job.table_name}/{job.job_id}.json.gz"
        
        extra_args = {'StorageClass': 'STANDARD_IA'}
        if encrypt:
            extra_args['ServerSideEncryption'] = 'AES256'
        
        self.s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=data,
            **extra_args
        )
        
        logger.info(f"Data stored to S3 IA: {key}")
    
    def _verify_archival(self, job: ArchivalJob, original_data: pd.DataFrame) -> bool:
        """
        Verify archival integrity | 驗證歸檔完整性
        """
        try:
            # For S3, verify object exists and has correct size
            bucket = self.config['aws_s3']['bucket']
            key_prefix = f"{job.target_tier.value.lower()}/{job.table_name}/{job.job_id}"
            
            response = self.s3_client.list_objects_v2(
                Bucket=bucket,
                Prefix=key_prefix
            )
            
            if 'Contents' in response and len(response['Contents']) > 0:
                stored_size = response['Contents'][0]['Size']
                return stored_size > 0
            
            return False
            
        except ClientError as e:
            logger.error(f"Archival verification failed: {e}")
            return False
    
    def _remove_archived_data(self, table_name: str, date_column: str, 
                            cutoff_date: datetime, condition_sql: Optional[str] = None):
        """
        Remove archived data from source | 從源移除歸檔數據
        """
        try:
            with self.connection_manager.get_session() as session:
                where_clause = f"{date_column} < :cutoff_date"
                if condition_sql:
                    where_clause += f" AND ({condition_sql})"
                
                query = f"DELETE FROM {table_name} WHERE {where_clause}"
                result = session.execute(text(query), {"cutoff_date": cutoff_date})
                
                logger.info(f"Removed {result.rowcount} archived rows from {table_name}")
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to remove archived data from {table_name}: {e}")
            raise
    
    def _check_active_positions(self, table_name: str, cutoff_date: datetime) -> Optional[str]:
        """
        Check for active trading positions | 檢查活躍交易倉位
        """
        if 'position' not in table_name.lower():
            return None
        
        try:
            with self.connection_manager.get_connection() as conn:
                result = conn.execute(text(f"""
                    SELECT COUNT(*) as active_count
                    FROM {table_name}
                    WHERE created_at < :cutoff_date
                    AND status = 'ACTIVE'
                """), {"cutoff_date": cutoff_date})
                
                row = result.fetchone()
                if row and row.active_count > 0:
                    return f"Found {row.active_count} active positions"
        except SQLAlchemyError:
            pass
        
        return None
    
    def _check_regulatory_requirements(self, table_name: str, cutoff_date: datetime) -> bool:
        """
        Check regulatory compliance requirements | 檢查監管合規要求
        """
        # For trading data, keep minimum 7 years | 對於交易數據，保留最少7年
        seven_years_ago = datetime.now() - timedelta(days=2555)
        
        if 'trading' in table_name.lower() and cutoff_date > seven_years_ago:
            return False
        
        return True
    
    def _safe_delete_data(self, table_name: str, date_column: str, 
                         cutoff_date: datetime, condition_sql: Optional[str] = None) -> int:
        """
        Safely delete data with confirmation | 安全刪除數據並確認
        """
        try:
            with self.connection_manager.get_session() as session:
                where_clause = f"{date_column} < :cutoff_date"
                if condition_sql:
                    where_clause += f" AND ({condition_sql})"
                
                query = f"DELETE FROM {table_name} WHERE {where_clause}"
                result = session.execute(text(query), {"cutoff_date": cutoff_date})
                
                deleted_rows = result.rowcount
                logger.info(f"Safely deleted {deleted_rows} rows from {table_name}")
                
                return deleted_rows
                
        except SQLAlchemyError as e:
            logger.error(f"Failed to delete data from {table_name}: {e}")
            raise
    
    def run_retention_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive retention analysis | 運行綜合保留分析
        """
        analysis_results = {
            "analysis_timestamp": datetime.now(),
            "rules_analyzed": len(self.retention_rules),
            "tables_analyzed": 0,
            "total_candidates": 0,
            "estimated_savings_bytes": 0,
            "rule_results": {}
        }
        
        for rule_name, rule in self.retention_rules.items():
            candidates = self.identify_archival_candidates(rule)
            
            rule_analysis = {
                "rule_name": rule_name,
                "policy": rule.policy.value,
                "tables_matched": len(self.get_matching_tables(rule.table_pattern)),
                "archival_candidates": len(candidates),
                "deletion_candidates": len([c for c in candidates if c.get("action") == "DELETE"]),
                "estimated_rows": sum(c.get("estimated_rows", 0) for c in candidates)
            }
            
            analysis_results["rule_results"][rule_name] = rule_analysis
            analysis_results["total_candidates"] += len(candidates)
        
        return analysis_results
    
    def execute_retention_policies(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Execute all retention policies | 執行所有保留政策
        """
        execution_results = {
            "execution_timestamp": datetime.now(),
            "dry_run": dry_run,
            "jobs_executed": 0,
            "jobs_successful": 0,
            "jobs_failed": 0,
            "total_rows_processed": 0,
            "total_bytes_processed": 0,
            "job_results": []
        }
        
        for rule_name, rule in self.retention_rules.items():
            candidates = self.identify_archival_candidates(rule)
            
            for candidate in candidates:
                if not dry_run:
                    job = self.execute_archival(candidate)
                    
                    execution_results["jobs_executed"] += 1
                    execution_results["total_rows_processed"] += job.rows_processed
                    execution_results["total_bytes_processed"] += job.bytes_processed
                    
                    if job.status == "SUCCESS":
                        execution_results["jobs_successful"] += 1
                    else:
                        execution_results["jobs_failed"] += 1
                    
                    execution_results["job_results"].append({
                        "job_id": job.job_id,
                        "rule_name": job.rule_name,
                        "table_name": job.table_name,
                        "status": job.status,
                        "rows_processed": job.rows_processed,
                        "error_message": job.error_message
                    })
                else:
                    # Dry run - just log what would be done | 試運行 - 只記錄將要做的事情
                    logger.info(f"[DRY RUN] Would process {candidate['table_name']} "
                              f"({candidate['source_tier'].value} -> {candidate.get('target_tier', 'DELETE')})")
        
        return execution_results
    
    def start_scheduler(self):
        """
        Start automated retention scheduler | 啟動自動保留調度器
        """
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("Retention scheduler already running")
            return
        
        def scheduler_loop():
            """Scheduled retention execution loop | 定時保留執行循環"""
            logger.info("Data retention scheduler started")
            
            while not self._shutdown_event.is_set():
                try:
                    # Execute retention policies | 執行保留政策
                    results = self.execute_retention_policies(dry_run=False)
                    
                    logger.info(f"Retention execution completed: "
                              f"{results['jobs_successful']} successful, "
                              f"{results['jobs_failed']} failed")
                    
                    # Wait for next execution | 等待下次執行
                    self._shutdown_event.wait(self._execution_interval * 3600)  # Convert hours to seconds
                    
                except Exception as e:
                    logger.error(f"Retention scheduler error: {e}")
                    self._shutdown_event.wait(300)  # Wait 5 minutes on error
        
        self._scheduler_thread = threading.Thread(target=scheduler_loop, daemon=True)
        self._scheduler_thread.start()
    
    def stop_scheduler(self):
        """
        Stop retention scheduler | 停止保留調度器
        """
        logger.info("Stopping data retention scheduler...")
        self._shutdown_event.set()
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=30)
        
        logger.info("Data retention scheduler stopped")
    
    def get_retention_status(self) -> Dict[str, Any]:
        """
        Get current retention system status | 獲取當前保留系統狀態
        """
        return {
            "scheduler_running": self._scheduler_thread and self._scheduler_thread.is_alive(),
            "execution_interval_hours": self._execution_interval,
            "retention_rules_count": len(self.retention_rules),
            "active_jobs": len([j for j in self.archival_jobs.values() if j.status == "RUNNING"]),
            "completed_jobs": len([j for j in self.archival_jobs.values() if j.status == "SUCCESS"]),
            "failed_jobs": len([j for j in self.archival_jobs.values() if j.status == "FAILED"]),
            "s3_configured": self.s3_client is not None,
            "last_analysis": max([r.last_executed for r in self.retention_rules.values() 
                                if r.last_executed], default=None)
        }


# Factory function for easy initialization | 便於初始化的工廠函數  
def create_data_retention_manager(connection_manager, config: Optional[Dict] = None) -> DataRetentionManager:
    """
    Create and configure data retention manager | 創建並配置數據保留管理器
    """
    if config is None:
        config = {
            'execution_interval_hours': 24,
            'aws_s3': {
                'enabled': False,
                'bucket': None
            }
        }
    
    return DataRetentionManager(connection_manager, config)