"""
Model Management System | 模型管理系統
Comprehensive system for managing AI models lifecycle in AIFX.
AIFX中管理AI模型生命週期的綜合系統。
"""

import os
import json
import pickle
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

# Import model components | 導入模型組件
from ..models.base_model import BaseModel, ModelRegistry
from ..models.xgboost_model import XGBoostModel
from ..models.random_forest_model import RandomForestModel
from ..models.lstm_model import LSTMModel
from ..evaluation.performance_metrics import TradingPerformanceMetrics

logger = logging.getLogger(__name__)


class ModelVersioning:
    """
    Model versioning system | 模型版本控制系統
    
    Handles model versioning, comparison, and rollback capabilities.
    處理模型版本控制、比較和回滾功能。
    """
    
    def __init__(self, base_path: str = "models/versions"):
        """
        Initialize model versioning system | 初始化模型版本控制系統
        
        Args:
            base_path: Base path for storing model versions | 存儲模型版本的基礎路徑
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        self.version_registry_path = self.base_path / "version_registry.json"
        self.version_registry = self._load_version_registry()
        
    def _load_version_registry(self) -> Dict[str, Any]:
        """Load version registry from file | 從文件加載版本註冊表"""
        if self.version_registry_path.exists():
            with open(self.version_registry_path, 'r') as f:
                return json.load(f)
        return {'models': {}, 'last_updated': datetime.now().isoformat()}
    
    def _save_version_registry(self):
        """Save version registry to file | 保存版本註冊表到文件"""
        self.version_registry['last_updated'] = datetime.now().isoformat()
        with open(self.version_registry_path, 'w') as f:
            json.dump(self.version_registry, f, indent=2, default=str)
    
    def create_version(self, model: BaseModel, version_name: str = None,
                      performance_metrics: Optional[Dict] = None,
                      notes: str = "") -> str:
        """
        Create a new model version | 創建新的模型版本
        
        Args:
            model: Model instance to version | 要版本化的模型實例
            version_name: Custom version name | 自定義版本名稱
            performance_metrics: Performance metrics for this version | 此版本的性能指標
            notes: Version notes | 版本註釋
            
        Returns:
            Version identifier | 版本標識符
        """
        if version_name is None:
            version_name = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        model_name = model.model_name
        model_path = self.base_path / model_name
        model_path.mkdir(exist_ok=True)
        
        version_path = model_path / version_name
        version_path.mkdir(exist_ok=True)
        
        # Save model | 保存模型
        model_file = version_path / "model.pkl"
        model.save_model(str(model_file))
        
        # Create model hash for integrity checking | 創建模型雜湊以進行完整性檢查
        model_hash = self._calculate_model_hash(str(model_file))
        
        # Save version metadata | 保存版本元數據
        version_metadata = {
            'version_name': version_name,
            'model_name': model_name,
            'model_type': type(model).__name__,
            'created_at': datetime.now().isoformat(),
            'model_hash': model_hash,
            'performance_metrics': performance_metrics or {},
            'notes': notes,
            'model_info': model.get_model_info(),
            'file_size': os.path.getsize(model_file)
        }
        
        metadata_file = version_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(version_metadata, f, indent=2, default=str)
        
        # Update version registry | 更新版本註冊表
        if model_name not in self.version_registry['models']:
            self.version_registry['models'][model_name] = {'versions': [], 'active_version': None}
        
        self.version_registry['models'][model_name]['versions'].append({
            'version_name': version_name,
            'created_at': version_metadata['created_at'],
            'performance_summary': self._extract_performance_summary(performance_metrics),
            'model_hash': model_hash
        })
        
        # Set as active version if it's the first or best performing | 如果是第一個或性能最佳則設為活動版本
        if (not self.version_registry['models'][model_name]['active_version'] or 
            self._is_better_version(performance_metrics, model_name)):
            self.version_registry['models'][model_name]['active_version'] = version_name
        
        self._save_version_registry()
        
        logger.info(f"Created version {version_name} for model {model_name}")
        return version_name
    
    def _calculate_model_hash(self, model_path: str) -> str:
        """Calculate SHA256 hash of model file | 計算模型文件的SHA256雜湊"""
        sha256_hash = hashlib.sha256()
        with open(model_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _extract_performance_summary(self, metrics: Optional[Dict]) -> Dict[str, float]:
        """Extract key performance metrics for comparison | 提取關鍵性能指標進行比較"""
        if not metrics:
            return {}
        
        summary = {}
        
        # Extract key metrics | 提取關鍵指標
        key_metrics = ['accuracy', 'f1_score', 'auc_roc', 'precision', 'recall']
        
        for metric in key_metrics:
            if metric in metrics:
                summary[metric] = metrics[metric]
            elif 'classification_metrics' in metrics and metric in metrics['classification_metrics']:
                summary[metric] = metrics['classification_metrics'][metric]
            elif 'detailed_metrics' in metrics and metric in metrics['detailed_metrics']:
                summary[metric] = metrics['detailed_metrics'][metric]
        
        return summary
    
    def _is_better_version(self, new_metrics: Optional[Dict], model_name: str) -> bool:
        """Determine if new version is better than current active | 確定新版本是否優於當前活動版本"""
        if not new_metrics:
            return False
        
        current_active = self.version_registry['models'][model_name]['active_version']
        if not current_active:
            return True
        
        # Compare F1 scores (primary metric) | 比較F1分數（主要指標）
        new_summary = self._extract_performance_summary(new_metrics)
        if 'f1_score' in new_summary:
            current_version_path = self.base_path / model_name / current_active / "metadata.json"
            if current_version_path.exists():
                with open(current_version_path, 'r') as f:
                    current_metadata = json.load(f)
                current_summary = self._extract_performance_summary(current_metadata.get('performance_metrics'))
                
                if 'f1_score' in current_summary:
                    return new_summary['f1_score'] > current_summary['f1_score']
        
        return False
    
    def load_version(self, model_name: str, version_name: str = None) -> BaseModel:
        """
        Load a specific model version | 加載特定模型版本
        
        Args:
            model_name: Name of the model | 模型名稱
            version_name: Version to load (None for active version) | 要加載的版本（None表示活動版本）
            
        Returns:
            Loaded model instance | 已加載的模型實例
        """
        if version_name is None:
            if model_name not in self.version_registry['models']:
                raise ValueError(f"Model {model_name} not found in registry")
            version_name = self.version_registry['models'][model_name]['active_version']
            if not version_name:
                raise ValueError(f"No active version set for model {model_name}")
        
        version_path = self.base_path / model_name / version_name
        if not version_path.exists():
            raise ValueError(f"Version {version_name} not found for model {model_name}")
        
        # Load metadata to determine model type | 加載元數據以確定模型類型
        metadata_file = version_path / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Map model types to classes | 將模型類型映射到類
        model_classes = {
            'XGBoostModel': XGBoostModel,
            'RandomForestModel': RandomForestModel,
            'LSTMModel': LSTMModel
        }
        
        model_type = metadata['model_type']
        if model_type not in model_classes:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create model instance and load | 創建模型實例並加載
        model_class = model_classes[model_type]
        model = model_class()
        model.load_model(str(version_path / "model.pkl"))
        
        # Verify integrity | 驗證完整性
        current_hash = self._calculate_model_hash(str(version_path / "model.pkl"))
        if current_hash != metadata['model_hash']:
            logger.warning(f"Model hash mismatch for {model_name} v{version_name}. File may be corrupted.")
        
        logger.info(f"Loaded model {model_name} version {version_name}")
        return model
    
    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a model | 列出模型的所有版本
        
        Args:
            model_name: Name of the model | 模型名稱
            
        Returns:
            List of version information | 版本信息列表
        """
        if model_name not in self.version_registry['models']:
            return []
        
        versions = []
        for version_info in self.version_registry['models'][model_name]['versions']:
            version_path = self.base_path / model_name / version_info['version_name']
            if version_path.exists():
                metadata_file = version_path / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    versions.append(metadata)
        
        return sorted(versions, key=lambda x: x['created_at'], reverse=True)
    
    def set_active_version(self, model_name: str, version_name: str):
        """
        Set active version for a model | 設置模型的活動版本
        
        Args:
            model_name: Name of the model | 模型名稱
            version_name: Version to set as active | 要設置為活動的版本
        """
        if model_name not in self.version_registry['models']:
            raise ValueError(f"Model {model_name} not found")
        
        version_path = self.base_path / model_name / version_name
        if not version_path.exists():
            raise ValueError(f"Version {version_name} not found")
        
        self.version_registry['models'][model_name]['active_version'] = version_name
        self._save_version_registry()
        
        logger.info(f"Set active version for {model_name} to {version_name}")
    
    def compare_versions(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two model versions | 比較兩個模型版本
        
        Args:
            model_name: Name of the model | 模型名稱
            version1: First version to compare | 要比較的第一個版本
            version2: Second version to compare | 要比較的第二個版本
            
        Returns:
            Comparison results | 比較結果
        """
        # Load metadata for both versions | 加載兩個版本的元數據
        version1_path = self.base_path / model_name / version1 / "metadata.json"
        version2_path = self.base_path / model_name / version2 / "metadata.json"
        
        if not version1_path.exists():
            raise ValueError(f"Version {version1} not found")
        if not version2_path.exists():
            raise ValueError(f"Version {version2} not found")
        
        with open(version1_path, 'r') as f:
            metadata1 = json.load(f)
        with open(version2_path, 'r') as f:
            metadata2 = json.load(f)
        
        # Compare performance metrics | 比較性能指標
        perf1 = self._extract_performance_summary(metadata1.get('performance_metrics'))
        perf2 = self._extract_performance_summary(metadata2.get('performance_metrics'))
        
        comparison = {
            'model_name': model_name,
            'version1': {
                'name': version1,
                'created_at': metadata1['created_at'],
                'performance': perf1,
                'file_size': metadata1.get('file_size', 0)
            },
            'version2': {
                'name': version2,
                'created_at': metadata2['created_at'],
                'performance': perf2,
                'file_size': metadata2.get('file_size', 0)
            },
            'performance_comparison': {},
            'recommendation': None
        }
        
        # Compare each metric | 比較每個指標
        all_metrics = set(perf1.keys()) | set(perf2.keys())
        better_count = 0
        total_comparisons = 0
        
        for metric in all_metrics:
            if metric in perf1 and metric in perf2:
                val1, val2 = perf1[metric], perf2[metric]
                comparison['performance_comparison'][metric] = {
                    'version1': val1,
                    'version2': val2,
                    'difference': val2 - val1,
                    'percentage_change': ((val2 - val1) / val1) * 100 if val1 != 0 else 0,
                    'better_version': version2 if val2 > val1 else (version1 if val1 > val2 else 'tie')
                }
                
                if val2 > val1:
                    better_count += 1
                total_comparisons += 1
        
        # Make recommendation | 給出建議
        if total_comparisons > 0:
            if better_count > total_comparisons / 2:
                comparison['recommendation'] = f"Version {version2} performs better overall"
            elif better_count < total_comparisons / 2:
                comparison['recommendation'] = f"Version {version1} performs better overall"
            else:
                comparison['recommendation'] = "Performance is comparable between versions"
        
        return comparison
    
    def cleanup_old_versions(self, model_name: str, keep_last_n: int = 5):
        """
        Clean up old model versions | 清理舊模型版本
        
        Args:
            model_name: Name of the model | 模型名稱
            keep_last_n: Number of recent versions to keep | 要保留的最新版本數量
        """
        if model_name not in self.version_registry['models']:
            logger.warning(f"Model {model_name} not found in registry")
            return
        
        versions = self.list_versions(model_name)
        if len(versions) <= keep_last_n:
            logger.info(f"Only {len(versions)} versions found for {model_name}, no cleanup needed")
            return
        
        active_version = self.version_registry['models'][model_name]['active_version']
        versions_to_delete = versions[keep_last_n:]
        
        deleted_count = 0
        for version in versions_to_delete:
            version_name = version['version_name']
            
            # Never delete active version | 永不刪除活動版本
            if version_name == active_version:
                continue
            
            version_path = self.base_path / model_name / version_name
            if version_path.exists():
                shutil.rmtree(version_path)
                deleted_count += 1
                
                # Remove from registry | 從註冊表中移除
                self.version_registry['models'][model_name]['versions'] = [
                    v for v in self.version_registry['models'][model_name]['versions']
                    if v['version_name'] != version_name
                ]
        
        self._save_version_registry()
        logger.info(f"Cleaned up {deleted_count} old versions for model {model_name}")


class ModelDeploymentManager:
    """
    Model deployment management | 模型部署管理
    
    Manages model deployment, A/B testing, and production rollouts.
    管理模型部署、A/B測試和生產發布。
    """
    
    def __init__(self, deployment_path: str = "models/deployment"):
        """
        Initialize deployment manager | 初始化部署管理器
        
        Args:
            deployment_path: Path for deployment configurations | 部署配置的路徑
        """
        self.deployment_path = Path(deployment_path)
        self.deployment_path.mkdir(parents=True, exist_ok=True)
        
        self.config_file = self.deployment_path / "deployment_config.json"
        self.deployment_config = self._load_deployment_config()
        
    def _load_deployment_config(self) -> Dict[str, Any]:
        """Load deployment configuration | 加載部署配置"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        
        return {
            'environments': {
                'development': {'active_models': {}},
                'staging': {'active_models': {}},
                'production': {'active_models': {}}
            },
            'ab_tests': {},
            'deployment_history': []
        }
    
    def _save_deployment_config(self):
        """Save deployment configuration | 保存部署配置"""
        with open(self.config_file, 'w') as f:
            json.dump(self.deployment_config, f, indent=2, default=str)
    
    def deploy_model(self, model_name: str, version_name: str, 
                    environment: str = "development",
                    deployment_notes: str = "") -> str:
        """
        Deploy model to specified environment | 將模型部署到指定環境
        
        Args:
            model_name: Name of the model | 模型名稱
            version_name: Version to deploy | 要部署的版本
            environment: Target environment | 目標環境
            deployment_notes: Deployment notes | 部署註釋
            
        Returns:
            Deployment ID | 部署ID
        """
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate environment | 驗證環境
        if environment not in self.deployment_config['environments']:
            raise ValueError(f"Environment {environment} not configured")
        
        # Record deployment | 記錄部署
        deployment_record = {
            'deployment_id': deployment_id,
            'model_name': model_name,
            'version_name': version_name,
            'environment': environment,
            'deployed_at': datetime.now().isoformat(),
            'deployed_by': 'system',  # Could be enhanced with user tracking
            'notes': deployment_notes,
            'status': 'active'
        }
        
        # Update environment | 更新環境
        self.deployment_config['environments'][environment]['active_models'][model_name] = {
            'version_name': version_name,
            'deployment_id': deployment_id,
            'deployed_at': deployment_record['deployed_at']
        }
        
        # Add to deployment history | 添加到部署歷史
        self.deployment_config['deployment_history'].append(deployment_record)
        
        self._save_deployment_config()
        
        logger.info(f"Deployed {model_name} v{version_name} to {environment} (ID: {deployment_id})")
        return deployment_id
    
    def setup_ab_test(self, test_name: str, model_a: Tuple[str, str], 
                     model_b: Tuple[str, str], traffic_split: float = 0.5,
                     environment: str = "staging") -> str:
        """
        Setup A/B test between two models | 設置兩個模型之間的A/B測試
        
        Args:
            test_name: Name of the A/B test | A/B測試名稱
            model_a: (model_name, version) for model A | 模型A的(模型名, 版本)
            model_b: (model_name, version) for model B | 模型B的(模型名, 版本)
            traffic_split: Percentage of traffic for model A | 模型A的流量百分比
            environment: Environment for testing | 測試環境
            
        Returns:
            A/B test ID | A/B測試ID
        """
        test_id = f"ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        ab_test_config = {
            'test_id': test_id,
            'test_name': test_name,
            'environment': environment,
            'model_a': {
                'name': model_a[0],
                'version': model_a[1],
                'traffic_percentage': traffic_split
            },
            'model_b': {
                'name': model_b[0],
                'version': model_b[1],
                'traffic_percentage': 1.0 - traffic_split
            },
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'metrics': {
                'model_a': {'predictions': 0, 'performance': {}},
                'model_b': {'predictions': 0, 'performance': {}}
            }
        }
        
        self.deployment_config['ab_tests'][test_id] = ab_test_config
        self._save_deployment_config()
        
        logger.info(f"Created A/B test {test_name} (ID: {test_id})")
        return test_id
    
    def get_deployment_status(self, environment: str = None) -> Dict[str, Any]:
        """
        Get current deployment status | 獲取當前部署狀態
        
        Args:
            environment: Specific environment (None for all) | 特定環境（None表示全部）
            
        Returns:
            Deployment status information | 部署狀態信息
        """
        if environment:
            if environment in self.deployment_config['environments']:
                return {
                    'environment': environment,
                    'active_models': self.deployment_config['environments'][environment]['active_models'],
                    'last_updated': datetime.now().isoformat()
                }
            else:
                return {'error': f'Environment {environment} not found'}
        else:
            return {
                'all_environments': self.deployment_config['environments'],
                'active_ab_tests': {k: v for k, v in self.deployment_config['ab_tests'].items() 
                                  if v['status'] == 'active'},
                'last_updated': datetime.now().isoformat()
            }
    
    def rollback_deployment(self, model_name: str, environment: str) -> str:
        """
        Rollback to previous model version | 回滾到上一個模型版本
        
        Args:
            model_name: Name of the model | 模型名稱
            environment: Target environment | 目標環境
            
        Returns:
            Rollback deployment ID | 回滾部署ID
        """
        # Find previous deployment | 查找上一個部署
        deployments = [d for d in self.deployment_config['deployment_history'] 
                      if d['model_name'] == model_name and d['environment'] == environment]
        
        if len(deployments) < 2:
            raise ValueError(f"No previous deployment found for {model_name} in {environment}")
        
        # Get second-to-last deployment | 獲取倒數第二個部署
        previous_deployment = sorted(deployments, key=lambda x: x['deployed_at'], reverse=True)[1]
        
        # Deploy previous version | 部署上一個版本
        rollback_id = self.deploy_model(
            model_name, 
            previous_deployment['version_name'],
            environment,
            f"Rollback from {deployments[-1]['version_name']} to {previous_deployment['version_name']}"
        )
        
        logger.info(f"Rolled back {model_name} in {environment} to version {previous_deployment['version_name']}")
        return rollback_id


class ModelLifecycleManager:
    """
    Complete model lifecycle management | 完整的模型生命週期管理
    
    Orchestrates versioning, deployment, monitoring, and maintenance.
    編排版本控制、部署、監控和維護。
    """
    
    def __init__(self, base_path: str = "models"):
        """
        Initialize lifecycle manager | 初始化生命週期管理器
        
        Args:
            base_path: Base path for model storage | 模型存儲的基礎路徑
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components | 初始化組件
        self.versioning = ModelVersioning(str(self.base_path / "versions"))
        self.deployment = ModelDeploymentManager(str(self.base_path / "deployment"))
        self.registry = ModelRegistry(str(self.base_path / "registry"))
        self.metrics = TradingPerformanceMetrics()
        
        # Lifecycle tracking | 生命週期跟踪
        self.lifecycle_log_path = self.base_path / "lifecycle.log"
        self._setup_lifecycle_logging()
        
    def _setup_lifecycle_logging(self):
        """Setup dedicated lifecycle logging | 設置專用生命週期日誌記錄"""
        lifecycle_logger = logging.getLogger('model_lifecycle')
        
        if not lifecycle_logger.handlers:
            handler = logging.FileHandler(self.lifecycle_log_path)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            lifecycle_logger.addHandler(handler)
            lifecycle_logger.setLevel(logging.INFO)
    
    def register_and_version_model(self, model: BaseModel, 
                                  performance_metrics: Optional[Dict] = None,
                                  version_notes: str = "",
                                  auto_deploy_dev: bool = True) -> Dict[str, str]:
        """
        Register model and create initial version | 註冊模型並創建初始版本
        
        Args:
            model: Model instance | 模型實例
            performance_metrics: Performance evaluation results | 性能評估結果
            version_notes: Notes for this version | 此版本的註釋
            auto_deploy_dev: Auto-deploy to development environment | 自動部署到開發環境
            
        Returns:
            Dictionary with registration and version info | 包含註冊和版本信息的字典
        """
        # Register model | 註冊模型
        model_path = str(self.base_path / "registry" / f"{model.model_name}.pkl")
        self.registry.register_model(model, model_path, 
                                   tags=['active', 'versioned'])
        
        # Create version | 創建版本
        version_name = self.versioning.create_version(
            model, performance_metrics=performance_metrics, notes=version_notes
        )
        
        result = {
            'model_name': model.model_name,
            'version_name': version_name,
            'registered_at': datetime.now().isoformat()
        }
        
        # Auto-deploy to development if requested | 如果請求則自動部署到開發環境
        if auto_deploy_dev:
            deployment_id = self.deployment.deploy_model(
                model.model_name, version_name, 'development',
                f"Auto-deployment of new model version {version_name}"
            )
            result['deployment_id'] = deployment_id
        
        # Log lifecycle event | 記錄生命週期事件
        lifecycle_logger = logging.getLogger('model_lifecycle')
        lifecycle_logger.info(f"Model {model.model_name} registered and versioned as {version_name}")
        
        return result
    
    def promote_model(self, model_name: str, version_name: str,
                     from_env: str = "development", to_env: str = "staging",
                     promotion_notes: str = "") -> str:
        """
        Promote model from one environment to another | 將模型從一個環境提升到另一個環境
        
        Args:
            model_name: Name of the model | 模型名稱
            version_name: Version to promote | 要提升的版本
            from_env: Source environment | 源環境
            to_env: Target environment | 目標環境
            promotion_notes: Notes for promotion | 提升註釋
            
        Returns:
            Deployment ID | 部署ID
        """
        # Validate source deployment | 驗證源部署
        source_status = self.deployment.get_deployment_status(from_env)
        if (model_name not in source_status.get('active_models', {}) or
            source_status['active_models'][model_name]['version_name'] != version_name):
            raise ValueError(f"Model {model_name} v{version_name} not active in {from_env}")
        
        # Deploy to target environment | 部署到目標環境
        deployment_id = self.deployment.deploy_model(
            model_name, version_name, to_env,
            f"Promoted from {from_env}: {promotion_notes}"
        )
        
        # Log lifecycle event | 記錄生命週期事件
        lifecycle_logger = logging.getLogger('model_lifecycle')
        lifecycle_logger.info(f"Promoted {model_name} v{version_name} from {from_env} to {to_env}")
        
        return deployment_id
    
    def get_model_health_status(self, model_name: str) -> Dict[str, Any]:
        """
        Get comprehensive model health status | 獲取全面的模型健康狀態
        
        Args:
            model_name: Name of the model | 模型名稱
            
        Returns:
            Health status information | 健康狀態信息
        """
        health_status = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'issues': []
        }
        
        try:
            # Check version status | 檢查版本狀態
            versions = self.versioning.list_versions(model_name)
            health_status['version_info'] = {
                'total_versions': len(versions),
                'latest_version': versions[0]['version_name'] if versions else None,
                'active_version': self.versioning.version_registry['models'].get(model_name, {}).get('active_version')
            }
            
            # Check deployment status | 檢查部署狀態
            deployment_status = self.deployment.get_deployment_status()
            health_status['deployment_info'] = {}
            
            for env, env_info in deployment_status['all_environments'].items():
                if model_name in env_info['active_models']:
                    health_status['deployment_info'][env] = env_info['active_models'][model_name]
            
            # Check for issues | 檢查問題
            if not versions:
                health_status['issues'].append("No versions found")
                health_status['overall_status'] = 'unhealthy'
            
            if not health_status['deployment_info']:
                health_status['issues'].append("Not deployed to any environment")
                health_status['overall_status'] = 'warning'
            
            # Check model registry | 檢查模型註冊表
            registry_models = self.registry.list_models()
            model_in_registry = any(m['model_id'].startswith(model_name) for m in registry_models)
            
            if not model_in_registry:
                health_status['issues'].append("Not found in model registry")
                health_status['overall_status'] = 'unhealthy'
            
        except Exception as e:
            health_status['overall_status'] = 'error'
            health_status['error'] = str(e)
            health_status['issues'].append(f"Health check failed: {str(e)}")
        
        return health_status
    
    def cleanup_lifecycle(self, model_name: str = None, 
                         keep_versions: int = 5,
                         cleanup_deployments_older_than_days: int = 30):
        """
        Cleanup model lifecycle artifacts | 清理模型生命週期構件
        
        Args:
            model_name: Specific model to clean (None for all) | 要清理的特定模型（None表示全部）
            keep_versions: Number of versions to keep per model | 每個模型保留的版本數
            cleanup_deployments_older_than_days: Remove deployment records older than N days | 刪除N天前的部署記錄
        """
        lifecycle_logger = logging.getLogger('model_lifecycle')
        
        if model_name:
            models_to_clean = [model_name]
        else:
            # Get all models from versioning system | 從版本控制系統獲取所有模型
            models_to_clean = list(self.versioning.version_registry['models'].keys())
        
        # Clean up versions | 清理版本
        for model in models_to_clean:
            try:
                self.versioning.cleanup_old_versions(model, keep_versions)
                lifecycle_logger.info(f"Cleaned up old versions for {model}")
            except Exception as e:
                lifecycle_logger.error(f"Failed to cleanup versions for {model}: {str(e)}")
        
        # Clean up old deployment records | 清理舊部署記錄
        cutoff_date = datetime.now() - timedelta(days=cleanup_deployments_older_than_days)
        
        original_count = len(self.deployment.deployment_config['deployment_history'])
        self.deployment.deployment_config['deployment_history'] = [
            d for d in self.deployment.deployment_config['deployment_history']
            if datetime.fromisoformat(d['deployed_at']) > cutoff_date
        ]
        
        removed_count = original_count - len(self.deployment.deployment_config['deployment_history'])
        if removed_count > 0:
            self.deployment._save_deployment_config()
            lifecycle_logger.info(f"Removed {removed_count} old deployment records")
        
        lifecycle_logger.info("Lifecycle cleanup completed")