"""
Model Training Pipeline | 模型訓練管道
Comprehensive pipeline for training and validating AI models in AIFX system.
AIFX系統中用於訓練和驗證AI模型的綜合管道。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from datetime import datetime
import os
import json
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import model classes | 導入模型類
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base_model import BaseModel, ModelRegistry
from models.xgboost_model import XGBoostModel
from models.random_forest_model import RandomForestModel
from models.lstm_model import LSTMModel

# Data utilities already imported via models

logger = logging.getLogger(__name__)


class ModelTrainingPipeline:
    """
    Comprehensive training pipeline for AI models | AI模型的綜合訓練管道
    
    Handles data preparation, model training, validation, and evaluation.
    處理數據準備、模型訓練、驗證和評估。
    """
    
    def __init__(self, output_dir: str = "output/training_results"):
        """
        Initialize training pipeline | 初始化訓練管道
        
        Args:
            output_dir: Directory for saving training results | 保存訓練結果的目錄
        """
        self.output_dir = output_dir
        self.results_history = []
        
        # Create output directory | 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model registry | 初始化模型註冊表
        registry_path = os.path.join(output_dir, "model_registry")
        self.model_registry = ModelRegistry(registry_path)
        
        # Available models | 可用模型
        self.available_models = {
            'xgboost': XGBoostModel,
            'random_forest': RandomForestModel,
            'lstm': LSTMModel
        }
        
        logger.info(f"Training pipeline initialized with output directory: {output_dir}")
    
    def prepare_data(self, data: pd.DataFrame, target_column: str,
                    test_size: float = 0.2, validation_size: float = 0.2,
                    time_series_split: bool = True, random_state: int = 42) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for model training | 為模型訓練準備數據
        
        Args:
            data: Input dataset | 輸入數據集
            target_column: Name of target column | 目標列名稱
            test_size: Proportion of data for testing | 測試數據比例
            validation_size: Proportion of training data for validation | 驗證數據比例
            time_series_split: Whether to use time series splitting | 是否使用時間序列分割
            random_state: Random state for reproducibility | 隨機狀態用於可重現性
            
        Returns:
            Dictionary containing train/validation/test splits | 包含訓練/驗證/測試分割的字典
        """
        logger.info(f"Preparing data with {len(data)} samples and target column '{target_column}'")
        
        # Validate input | 驗證輸入
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Separate features and target | 分離特徵和目標
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle missing values in features | 處理特徵中的缺失值
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Encode categorical features | 編碼分類特徵
        categorical_columns = X.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            logger.info(f"Encoding categorical columns: {list(categorical_columns)}")
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        if time_series_split:
            # Time series split (chronological order) | 時間序列分割（按時間順序）
            n_samples = len(X)
            test_idx = int(n_samples * (1 - test_size))
            
            # Split train/test | 分割訓練/測試
            X_train_val = X.iloc[:test_idx]
            X_test = X.iloc[test_idx:]
            y_train_val = y.iloc[:test_idx]
            y_test = y.iloc[test_idx:]
            
            # Further split train/validation | 進一步分割訓練/驗證
            val_idx = int(len(X_train_val) * (1 - validation_size))
            X_train = X_train_val.iloc[:val_idx]
            X_val = X_train_val.iloc[val_idx:]
            y_train = y_train_val.iloc[:val_idx]
            y_val = y_train_val.iloc[val_idx:]
            
        else:
            # Random split | 隨機分割
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=validation_size, 
                random_state=random_state, stratify=y_train_val
            )
        
        # Log data splits | 記錄數據分割
        logger.info(f"Data split - Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}")
        
        return {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'feature_names': list(X.columns),
            'target_name': target_column
        }
    
    def train_model(self, model_type: str, data_splits: Dict[str, pd.DataFrame],
                   model_params: Optional[Dict[str, Any]] = None,
                   optimize_hyperparameters: bool = True,
                   save_model: bool = True) -> Dict[str, Any]:
        """
        Train a single model | 訓練單個模型
        
        Args:
            model_type: Type of model to train | 要訓練的模型類型
            data_splits: Data splits from prepare_data | 來自prepare_data的數據分割
            model_params: Optional model parameters | 可選模型參數
            optimize_hyperparameters: Whether to optimize hyperparameters | 是否優化超參數
            save_model: Whether to save the trained model | 是否保存訓練好的模型
            
        Returns:
            Training results | 訓練結果
        """
        if model_type not in self.available_models:
            raise ValueError(f"Model type '{model_type}' not available. Choose from: {list(self.available_models.keys())}")
        
        logger.info(f"Training {model_type} model")
        
        # Initialize model | 初始化模型
        model_class = self.available_models[model_type]
        model_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model = model_class(model_name=model_name)
        
        # Extract data splits | 提取數據分割
        X_train = data_splits['X_train']
        X_val = data_splits['X_val']
        X_test = data_splits['X_test']
        y_train = data_splits['y_train']
        y_val = data_splits['y_val']
        y_test = data_splits['y_test']
        
        # Train model | 訓練模型
        start_time = datetime.now()
        
        try:
            training_history = model.train(
                X_train, y_train, X_val, y_val,
                optimize_hyperparameters=optimize_hyperparameters,
                **(model_params or {})
            )
            
            training_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Model training completed in {training_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'model_type': model_type,
                'model_name': model_name
            }
        
        # Evaluate model | 評估模型
        evaluation_results = self.evaluate_model(model, data_splits)
        
        # Combine results | 合併結果
        results = {
            'success': True,
            'model_type': model_type,
            'model_name': model_name,
            'model_instance': model,
            'training_history': training_history,
            'evaluation_results': evaluation_results,
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save model if requested | 如果請求則保存模型
        if save_model:
            model_path = os.path.join(self.output_dir, "models", f"{model_name}.pkl")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            try:
                model.save_model(model_path)
                results['model_path'] = model_path
                
                # Register model | 註冊模型
                self.model_registry.register_model(
                    model, model_path, 
                    tags=[model_type, 'trained', datetime.now().strftime('%Y%m%d')]
                )
                
                logger.info(f"Model saved to {model_path}")
                
            except Exception as e:
                logger.warning(f"Failed to save model: {str(e)}")
                results['save_error'] = str(e)
        
        # Store results | 存儲結果
        self.results_history.append(results)
        
        return results
    
    def evaluate_model(self, model: BaseModel, data_splits: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Comprehensive model evaluation | 全面的模型評估
        
        Args:
            model: Trained model instance | 訓練好的模型實例
            data_splits: Data splits for evaluation | 用於評估的數據分割
            
        Returns:
            Evaluation results | 評估結果
        """
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        # Basic evaluation | 基本評估
        try:
            basic_metrics = model.evaluate(X_test, y_test)
        except Exception as e:
            logger.warning(f"Basic evaluation failed: {str(e)}")
            basic_metrics = {}
        
        # Detailed predictions | 詳細預測
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Calculate additional metrics | 計算額外指標
            detailed_metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Add AUC if binary classification | 如果是二分類則添加AUC
            if len(np.unique(y_test)) == 2:
                detailed_metrics['auc'] = roc_auc_score(y_test, y_pred_proba[:, 1])
            
        except Exception as e:
            logger.warning(f"Detailed evaluation failed: {str(e)}")
            detailed_metrics = {}
        
        # Feature importance | 特徵重要性
        feature_importance = model.get_feature_importance()
        
        # Model info | 模型信息
        model_info = model.get_model_info()
        
        return {
            'basic_metrics': basic_metrics,
            'detailed_metrics': detailed_metrics,
            'feature_importance': feature_importance,
            'model_info': model_info,
            'test_samples': len(X_test),
            'prediction_samples': len(y_pred) if 'y_pred' in locals() else 0
        }
    
    def train_multiple_models(self, data_splits: Dict[str, pd.DataFrame],
                            model_types: List[str] = None,
                            model_configs: Optional[Dict[str, Dict[str, Any]]] = None,
                            optimize_hyperparameters: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Train multiple models and compare results | 訓練多個模型並比較結果
        
        Args:
            data_splits: Data splits from prepare_data | 來自prepare_data的數據分割
            model_types: List of model types to train | 要訓練的模型類型列表
            model_configs: Optional configurations for each model | 每個模型的可選配置
            optimize_hyperparameters: Whether to optimize hyperparameters | 是否優化超參數
            
        Returns:
            Results from all trained models | 所有訓練模型的結果
        """
        if model_types is None:
            model_types = list(self.available_models.keys())
        
        if model_configs is None:
            model_configs = {}
        
        logger.info(f"Training multiple models: {model_types}")
        
        all_results = {}
        
        for model_type in model_types:
            logger.info(f"Training {model_type} model...")
            
            model_params = model_configs.get(model_type, {})
            
            try:
                results = self.train_model(
                    model_type=model_type,
                    data_splits=data_splits,
                    model_params=model_params,
                    optimize_hyperparameters=optimize_hyperparameters,
                    save_model=True
                )
                
                all_results[model_type] = results
                
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {str(e)}")
                all_results[model_type] = {
                    'success': False,
                    'error': str(e),
                    'model_type': model_type
                }
        
        # Compare models | 比較模型
        comparison = self.compare_models(all_results)
        
        # Save comparison results | 保存比較結果
        comparison_path = os.path.join(self.output_dir, "model_comparison.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        logger.info(f"Model comparison saved to {comparison_path}")
        
        return {
            'individual_results': all_results,
            'comparison': comparison,
            'best_model': comparison.get('best_model', None)
        }
    
    def compare_models(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple trained models | 比較多個訓練好的模型
        
        Args:
            results: Results from multiple models | 多個模型的結果
            
        Returns:
            Model comparison results | 模型比較結果
        """
        successful_results = {k: v for k, v in results.items() if v.get('success', False)}
        
        if not successful_results:
            return {'error': 'No successful model training results to compare'}
        
        comparison_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        comparison = {
            'models_compared': list(successful_results.keys()),
            'metrics_comparison': {},
            'training_times': {},
            'best_model': None,
            'model_rankings': {}
        }
        
        # Extract metrics for comparison | 提取指標進行比較
        for metric in comparison_metrics:
            comparison['metrics_comparison'][metric] = {}
            
            for model_type, result in successful_results.items():
                detailed_metrics = result.get('evaluation_results', {}).get('detailed_metrics', {})
                if metric in detailed_metrics:
                    comparison['metrics_comparison'][metric][model_type] = detailed_metrics[metric]
                    
                # Also record training time | 同時記錄訓練時間
                comparison['training_times'][model_type] = result.get('training_time', 0)
        
        # Determine best model based on F1 score | 基於F1分數確定最佳模型
        if 'f1_score' in comparison['metrics_comparison'] and comparison['metrics_comparison']['f1_score']:
            best_model_type = max(
                comparison['metrics_comparison']['f1_score'],
                key=comparison['metrics_comparison']['f1_score'].get
            )
            comparison['best_model'] = {
                'model_type': best_model_type,
                'f1_score': comparison['metrics_comparison']['f1_score'][best_model_type],
                'model_details': successful_results[best_model_type]
            }
        
        # Create model rankings | 創建模型排名
        for metric in comparison_metrics:
            if metric in comparison['metrics_comparison'] and comparison['metrics_comparison'][metric]:
                sorted_models = sorted(
                    comparison['metrics_comparison'][metric].items(),
                    key=lambda x: x[1], reverse=True
                )
                comparison['model_rankings'][metric] = [model for model, score in sorted_models]
        
        return comparison
    
    def cross_validate_model(self, model_type: str, data: pd.DataFrame, 
                           target_column: str, cv_folds: int = 5,
                           model_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform cross-validation for a model | 對模型執行交叉驗證
        
        Args:
            model_type: Type of model to validate | 要驗證的模型類型
            data: Complete dataset | 完整數據集
            target_column: Name of target column | 目標列名稱
            cv_folds: Number of cross-validation folds | 交叉驗證折數
            model_params: Optional model parameters | 可選模型參數
            
        Returns:
            Cross-validation results | 交叉驗證結果
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation for {model_type}")
        
        # Prepare data | 準備數據
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle missing values | 處理缺失值
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Time series split for cross-validation | 交叉驗證的時間序列分割
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_results = {
            'model_type': model_type,
            'cv_folds': cv_folds,
            'fold_results': [],
            'mean_scores': {},
            'std_scores': {}
        }
        
        fold_scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': []}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")
            
            # Split data | 分割數據
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # Initialize and train model | 初始化並訓練模型
            model_class = self.available_models[model_type]
            fold_model = model_class(model_name=f"{model_type}_fold_{fold}")
            
            try:
                fold_model.train(
                    X_train_fold, y_train_fold,
                    optimize_hyperparameters=False,  # Don't optimize for each fold
                    **(model_params or {})
                )
                
                # Evaluate fold | 評估折
                fold_evaluation = fold_model.evaluate(X_val_fold, y_val_fold)
                
                # Extract scores | 提取分數
                for metric in fold_scores.keys():
                    if metric in fold_evaluation:
                        fold_scores[metric].append(fold_evaluation[metric])
                
                cv_results['fold_results'].append({
                    'fold': fold,
                    'evaluation': fold_evaluation,
                    'train_samples': len(X_train_fold),
                    'val_samples': len(X_val_fold)
                })
                
            except Exception as e:
                logger.error(f"Fold {fold} failed: {str(e)}")
                cv_results['fold_results'].append({
                    'fold': fold,
                    'error': str(e)
                })
        
        # Calculate mean and std scores | 計算平均和標準差分數
        for metric, scores in fold_scores.items():
            if scores:
                cv_results['mean_scores'][metric] = np.mean(scores)
                cv_results['std_scores'][metric] = np.std(scores)
        
        logger.info(f"Cross-validation completed. Mean accuracy: {cv_results['mean_scores'].get('accuracy', 'N/A'):.4f}")
        
        return cv_results
    
    def save_training_report(self, filename: str = None) -> str:
        """
        Save comprehensive training report | 保存全面的訓練報告
        
        Args:
            filename: Optional custom filename | 可選的自定義文件名
            
        Returns:
            Path to saved report | 保存報告的路徑
        """
        if filename is None:
            filename = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_path = os.path.join(self.output_dir, filename)
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'total_models_trained': len(self.results_history),
            'training_history': self.results_history,
            'available_models': list(self.available_models.keys()),
            'output_directory': self.output_dir,
            'model_registry_info': {
                'registry_path': self.model_registry.registry_path,
                'registered_models': len(self.model_registry.models)
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Training report saved to {report_path}")
        
        return report_path