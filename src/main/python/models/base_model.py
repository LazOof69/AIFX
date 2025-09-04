"""
Base Model Framework | 基礎模型框架
Abstract base classes and interfaces for all AI models in AIFX system.
AIFX系統中所有AI模型的抽象基類和接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import pickle
import joblib
import json
from datetime import datetime
import logging

# Configure logger | 配置日誌器
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all AI models | 所有AI模型的抽象基類
    
    Provides common interface for training, prediction, and model management.
    為訓練、預測和模型管理提供通用接口。
    """
    
    def __init__(self, model_name: str, version: str = "1.0"):
        """
        Initialize base model | 初始化基礎模型
        
        Args:
            model_name: Name of the model | 模型名稱
            version: Model version | 模型版本
        """
        self.model_name = model_name
        self.version = version
        self.model = None
        self.is_trained = False
        self.training_history = {}
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'model_name': model_name,
            'version': version,
            'training_samples': 0,
            'features_used': [],
            'performance_metrics': {}
        }
        
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train the model | 訓練模型
        
        Args:
            X_train: Training features | 訓練特徵
            y_train: Training targets | 訓練目標
            X_val: Validation features | 驗證特徵
            y_val: Validation targets | 驗證目標
            **kwargs: Additional training parameters | 額外訓練參數
            
        Returns:
            Training history and metrics | 訓練歷史和指標
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions | 進行預測
        
        Args:
            X: Input features | 輸入特徵
            
        Returns:
            Predictions | 預測結果
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities | 預測類別機率
        
        Args:
            X: Input features | 輸入特徵
            
        Returns:
            Class probabilities | 類別機率
        """
        pass
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance | 評估模型性能
        
        Args:
            X_test: Test features | 測試特徵
            y_test: Test targets | 測試目標
            
        Returns:
            Performance metrics | 性能指標
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation | 模型必須在評估前進行訓練")
            
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'prediction_confidence': np.mean(np.max(y_pred_proba, axis=1))
        }
        
        # Update metadata | 更新元數據
        self.metadata['performance_metrics'].update(metrics)
        
        logger.info(f"Model {self.model_name} evaluation completed: {metrics}")
        
        return metrics
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores | 獲取特徵重要性評分
        
        Returns:
            Feature importance dictionary | 特徵重要性字典
        """
        if hasattr(self.model, 'feature_importances_'):
            if hasattr(self, 'feature_names'):
                return dict(zip(self.feature_names, self.model.feature_importances_))
            else:
                return dict(enumerate(self.model.feature_importances_))
        return None
    
    def save_model(self, filepath: str, include_metadata: bool = True):
        """
        Save model to file | 保存模型到文件
        
        Args:
            filepath: Path to save model | 保存模型的路徑
            include_metadata: Whether to save metadata | 是否保存元數據
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model | 無法保存未訓練的模型")
            
        # Save model using joblib for sklearn models | 使用joblib為sklearn模型保存
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'version': self.version,
            'is_trained': self.is_trained,
            'training_history': self.training_history
        }
        
        if include_metadata:
            model_data['metadata'] = self.metadata
            
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file | 從文件加載模型
        
        Args:
            filepath: Path to load model from | 加載模型的路徑
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.version = model_data['version']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data.get('training_history', {})
        
        if 'metadata' in model_data:
            self.metadata = model_data['metadata']
            
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information | 獲取全面的模型信息
        
        Returns:
            Model information dictionary | 模型信息字典
        """
        return {
            'model_name': self.model_name,
            'version': self.version,
            'is_trained': self.is_trained,
            'metadata': self.metadata,
            'training_history_keys': list(self.training_history.keys()),
            'feature_importance_available': self.get_feature_importance() is not None
        }


class ModelRegistry:
    """
    Model Registry for managing multiple models | 模型註冊表用於管理多個模型
    
    Provides centralized model management and versioning.
    提供集中化的模型管理和版本控制。
    """
    
    def __init__(self, registry_path: str = "models/registry"):
        """
        Initialize model registry | 初始化模型註冊表
        
        Args:
            registry_path: Path to store registry data | 存儲註冊表數據的路徑
        """
        self.registry_path = registry_path
        self.models = {}
        self.registry_file = f"{registry_path}/registry.json"
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from file | 從文件加載註冊表"""
        try:
            with open(self.registry_file, 'r') as f:
                registry_data = json.load(f)
                self.models = registry_data.get('models', {})
        except FileNotFoundError:
            logger.info("Registry file not found, creating new registry")
            self.models = {}
    
    def _save_registry(self):
        """Save registry to file | 保存註冊表到文件"""
        import os
        os.makedirs(self.registry_path, exist_ok=True)
        
        registry_data = {
            'models': self.models,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def register_model(self, model: BaseModel, model_path: str, 
                      tags: Optional[List[str]] = None):
        """
        Register a model in the registry | 在註冊表中註冊模型
        
        Args:
            model: Model instance | 模型實例
            model_path: Path where model is saved | 模型保存路徑
            tags: Optional tags for categorization | 用於分類的可選標籤
        """
        model_id = f"{model.model_name}_{model.version}"
        
        self.models[model_id] = {
            'model_name': model.model_name,
            'version': model.version,
            'model_path': model_path,
            'registered_at': datetime.now().isoformat(),
            'metadata': model.metadata,
            'tags': tags or [],
            'is_active': True
        }
        
        self._save_registry()
        logger.info(f"Model {model_id} registered successfully")
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model information from registry | 從註冊表獲取模型信息
        
        Args:
            model_id: Model identifier | 模型標識符
            
        Returns:
            Model information or None | 模型信息或None
        """
        return self.models.get(model_id)
    
    def list_models(self, tag_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered models | 列出所有已註冊的模型
        
        Args:
            tag_filter: Optional tag to filter by | 用於過濾的可選標籤
            
        Returns:
            List of model information | 模型信息列表
        """
        models = []
        for model_id, model_info in self.models.items():
            if tag_filter is None or tag_filter in model_info.get('tags', []):
                models.append({
                    'model_id': model_id,
                    **model_info
                })
        return models
    
    def deactivate_model(self, model_id: str):
        """
        Deactivate a model | 停用模型
        
        Args:
            model_id: Model identifier | 模型標識符
        """
        if model_id in self.models:
            self.models[model_id]['is_active'] = False
            self._save_registry()
            logger.info(f"Model {model_id} deactivated")
        else:
            raise ValueError(f"Model {model_id} not found in registry")