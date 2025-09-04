"""
Random Forest Model Implementation | 隨機森林模型實現
Ensemble learning model for robust forex price direction prediction.
用於穩健外匯價格方向預測的集成學習模型。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, validation_curve
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest classifier for forex price direction prediction | 用於外匯價格方向預測的隨機森林分類器
    
    Implements ensemble learning with advanced bootstrapping and feature selection.
    實現具有高級自助抽樣和特徵選擇的集成學習。
    """
    
    def __init__(self, model_name: str = "RandomForest_Classifier", version: str = "1.0"):
        """
        Initialize Random Forest model | 初始化隨機森林模型
        
        Args:
            model_name: Name of the model | 模型名稱
            version: Model version | 模型版本
        """
        super().__init__(model_name, version)
        
        # Default hyperparameters | 默認超參數
        self.default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'bootstrap': True,
            'oob_score': True,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.best_params = None
        self.feature_names = None
        self.cv_scores = None
        self.oob_score = None
        self.trees_used = []
        
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for Random Forest training | 為隨機森林訓練準備特徵
        
        Args:
            X: Input features | 輸入特徵
            
        Returns:
            Processed features | 處理後的特徵
        """
        # Store feature names for later use | 存儲特徵名稱以供後續使用
        if self.feature_names is None:
            self.feature_names = list(X.columns)
        
        # Handle any missing values | 處理任何缺失值
        X_processed = X.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure all features are numeric | 確保所有特徵都是數值型
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
        
        # Fill any remaining NaN values with median | 用中位數填充任何剩餘的NaN值
        X_processed = X_processed.fillna(X_processed.median())
        
        return X_processed
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              optimize_hyperparameters: bool = True, cv_folds: int = 5,
              **kwargs) -> Dict[str, Any]:
        """
        Train Random Forest model with optional hyperparameter optimization | 訓練帶有可選超參數優化的隨機森林模型
        
        Args:
            X_train: Training features | 訓練特徵
            y_train: Training targets | 訓練目標
            X_val: Validation features | 驗證特徵
            y_val: Validation targets | 驗證目標
            optimize_hyperparameters: Whether to perform hyperparameter optimization | 是否進行超參數優化
            cv_folds: Number of cross-validation folds | 交叉驗證折數
            **kwargs: Additional training parameters | 額外訓練參數
            
        Returns:
            Training history and metrics | 訓練歷史和指標
        """
        logger.info(f"Training {self.model_name} with {len(X_train)} samples")
        
        # Prepare features | 準備特徵
        X_train_processed = self._prepare_features(X_train)
        
        # Update metadata | 更新元數據
        self.metadata['training_samples'] = len(X_train)
        self.metadata['features_used'] = list(X_train_processed.columns)
        
        training_history = {}
        
        if optimize_hyperparameters:
            logger.info("Performing hyperparameter optimization...")
            
            # Define parameter grid | 定義參數網格
            param_grid = {
                'n_estimators': [50, 100, 150, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
            
            # Initialize base model | 初始化基礎模型
            base_model = RandomForestClassifier(**self.default_params)
            
            # Perform grid search | 執行網格搜索
            grid_search = GridSearchCV(
                base_model, param_grid, cv=cv_folds, 
                scoring='accuracy', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train_processed, y_train)
            
            self.best_params = grid_search.best_params_
            self.model = grid_search.best_estimator_
            
            training_history['best_params'] = self.best_params
            training_history['best_cv_score'] = grid_search.best_score_
            training_history['cv_results'] = grid_search.cv_results_
            
            logger.info(f"Best parameters: {self.best_params}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            # Use default parameters | 使用默認參數
            params = {**self.default_params, **kwargs}
            self.model = RandomForestClassifier(**params)
            
            # Train the model | 訓練模型
            self.model.fit(X_train_processed, y_train)
        
        # Get OOB score if available | 如果可用，獲取OOB評分
        if self.model.bootstrap and hasattr(self.model, 'oob_score_'):
            self.oob_score = self.model.oob_score_
            training_history['oob_score'] = self.oob_score
            logger.info(f"OOB Score: {self.oob_score:.4f}")
        
        # Perform cross-validation | 執行交叉驗證
        self.cv_scores = cross_val_score(
            self.model, X_train_processed, y_train, 
            cv=cv_folds, scoring='accuracy'
        )
        
        training_history['cv_scores'] = self.cv_scores.tolist()
        training_history['cv_mean'] = self.cv_scores.mean()
        training_history['cv_std'] = self.cv_scores.std()
        
        # Analyze learning curves | 分析學習曲線
        n_estimators_range = [10, 25, 50, 75, 100, 125, 150]
        train_scores, val_scores = validation_curve(
            RandomForestClassifier(**self.model.get_params()),
            X_train_processed, y_train,
            param_name='n_estimators',
            param_range=n_estimators_range,
            cv=3, scoring='accuracy', n_jobs=-1
        )
        
        training_history['learning_curve'] = {
            'n_estimators_range': n_estimators_range,
            'train_scores_mean': train_scores.mean(axis=1).tolist(),
            'train_scores_std': train_scores.std(axis=1).tolist(),
            'val_scores_mean': val_scores.mean(axis=1).tolist(),
            'val_scores_std': val_scores.std(axis=1).tolist()
        }
        
        # Store training history | 存儲訓練歷史
        self.training_history = training_history
        self.is_trained = True
        
        # Log training completion | 記錄訓練完成
        logger.info(f"Training completed. CV Score: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std() * 2:.4f})")
        
        return training_history
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions | 進行預測
        
        Args:
            X: Input features | 輸入特徵
            
        Returns:
            Predictions | 預測結果
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction | 模型必須在預測前進行訓練")
        
        X_processed = self._prepare_features(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities | 預測類別機率
        
        Args:
            X: Input features | 輸入特徵
            
        Returns:
            Class probabilities | 類別機率
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction | 模型必須在預測前進行訓練")
        
        X_processed = self._prepare_features(X)
        return self.model.predict_proba(X_processed)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores | 獲取特徵重要性評分
        
        Returns:
            Feature importance dictionary | 特徵重要性字典
        """
        if not self.is_trained:
            return None
            
        importance_scores = self.model.feature_importances_
        if self.feature_names is not None:
            return dict(zip(self.feature_names, importance_scores))
        else:
            return dict(enumerate(importance_scores))
    
    def get_tree_diversity_metrics(self) -> Dict[str, float]:
        """
        Analyze diversity metrics of the ensemble | 分析集成的多樣性指標
        
        Returns:
            Diversity metrics | 多樣性指標
        """
        if not self.is_trained:
            return {}
        
        # Get predictions from individual trees | 獲取各個樹的預測
        n_samples = 100  # Sample size for diversity analysis | 多樣性分析的樣本大小
        if hasattr(self, 'X_sample') and len(self.X_sample) >= n_samples:
            X_sample = self.X_sample.iloc[:n_samples]
        else:
            # Create a small random sample for analysis | 創建一個小的隨機樣本進行分析
            logger.warning("No sample data available, skipping diversity metrics")
            return {}
        
        tree_predictions = []
        for estimator in self.model.estimators_:
            pred = estimator.predict(X_sample)
            tree_predictions.append(pred)
        
        tree_predictions = np.array(tree_predictions)
        
        # Calculate pairwise disagreement | 計算配對分歧
        n_trees = len(self.model.estimators_)
        disagreements = []
        
        for i in range(n_trees):
            for j in range(i + 1, n_trees):
                disagreement = np.mean(tree_predictions[i] != tree_predictions[j])
                disagreements.append(disagreement)
        
        return {
            'mean_pairwise_disagreement': np.mean(disagreements),
            'std_pairwise_disagreement': np.std(disagreements),
            'n_trees_analyzed': n_trees
        }
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        Plot feature importance | 繪製特徵重要性
        
        Args:
            top_n: Number of top features to display | 顯示的頂級特徵數量
            save_path: Optional path to save the plot | 保存圖表的可選路徑
        """
        if not self.is_trained:
            raise ValueError("Model must be trained to plot feature importance")
        
        try:
            import matplotlib.pyplot as plt
            
            importance_dict = self.get_feature_importance()
            if importance_dict is None:
                logger.warning("No feature importance available")
                return
                
            # Sort features by importance | 按重要性排序特徵
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:top_n]
            
            # Create plot | 創建圖表
            features, importances = zip(*top_features)
            
            plt.figure(figsize=(12, 8))
            bars = plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance | 特徵重要性')
            plt.title(f'{self.model_name} - Top {top_n} Feature Importances | 前{top_n}個特徵重要性')
            plt.gca().invert_yaxis()
            
            # Add value labels on bars | 在條形上添加數值標籤
            for bar, importance in zip(bars, importances):
                plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{importance:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def plot_learning_curve(self, save_path: Optional[str] = None):
        """
        Plot learning curve showing model performance vs number of trees | 繪製顯示模型性能與樹數量關係的學習曲線
        
        Args:
            save_path: Optional path to save the plot | 保存圖表的可選路徑
        """
        if not self.is_trained or 'learning_curve' not in self.training_history:
            raise ValueError("Model must be trained with learning curve data")
        
        try:
            import matplotlib.pyplot as plt
            
            lc_data = self.training_history['learning_curve']
            n_estimators = lc_data['n_estimators_range']
            train_mean = lc_data['train_scores_mean']
            train_std = lc_data['train_scores_std']
            val_mean = lc_data['val_scores_mean']
            val_std = lc_data['val_scores_std']
            
            plt.figure(figsize=(10, 6))
            
            # Plot training scores | 繪製訓練評分
            plt.plot(n_estimators, train_mean, 'o-', color='blue', label='Training Accuracy | 訓練準確度')
            plt.fill_between(n_estimators, 
                           np.array(train_mean) - np.array(train_std),
                           np.array(train_mean) + np.array(train_std),
                           alpha=0.1, color='blue')
            
            # Plot validation scores | 繪製驗證評分
            plt.plot(n_estimators, val_mean, 'o-', color='red', label='Validation Accuracy | 驗證準確度')
            plt.fill_between(n_estimators,
                           np.array(val_mean) - np.array(val_std),
                           np.array(val_mean) + np.array(val_std),
                           alpha=0.1, color='red')
            
            plt.xlabel('Number of Trees | 樹的數量')
            plt.ylabel('Accuracy Score | 準確度評分')
            plt.title(f'{self.model_name} - Learning Curve | 學習曲線')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Learning curve plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def get_ensemble_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive ensemble statistics | 獲取全面的集成統計信息
        
        Returns:
            Ensemble statistics | 集成統計信息
        """
        if not self.is_trained:
            return {}
        
        stats = {
            'n_estimators': self.model.n_estimators,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'max_depth': self.model.max_depth,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf,
            'bootstrap': self.model.bootstrap,
            'oob_score': self.oob_score
        }
        
        # Add tree-specific statistics | 添加樹特定統計信息
        if hasattr(self.model, 'estimators_'):
            tree_depths = [tree.tree_.max_depth for tree in self.model.estimators_]
            tree_leaves = [tree.tree_.n_leaves for tree in self.model.estimators_]
            
            stats.update({
                'tree_depths_mean': np.mean(tree_depths),
                'tree_depths_std': np.std(tree_depths),
                'tree_leaves_mean': np.mean(tree_leaves),
                'tree_leaves_std': np.std(tree_leaves),
                'total_nodes': sum([tree.tree_.node_count for tree in self.model.estimators_])
            })
        
        return stats
    
    def explain_prediction_with_trees(self, X_sample: pd.DataFrame, 
                                    n_trees_to_show: int = 5) -> Dict[str, Any]:
        """
        Explain prediction using individual trees | 使用各個樹解釋預測
        
        Args:
            X_sample: Single sample to explain | 要解釋的單個樣本
            n_trees_to_show: Number of trees to analyze | 要分析的樹數量
            
        Returns:
            Tree-based explanation | 基於樹的解釋
        """
        if not self.is_trained:
            raise ValueError("Model must be trained for explanation")
        
        if len(X_sample) != 1:
            raise ValueError("Only single sample explanation supported")
        
        X_processed = self._prepare_features(X_sample)
        
        # Get overall prediction | 獲取總體預測
        prediction = self.model.predict(X_processed)[0]
        probabilities = self.model.predict_proba(X_processed)[0]
        
        # Get predictions from individual trees | 獲取各個樹的預測
        tree_predictions = []
        tree_probabilities = []
        
        for i, tree in enumerate(self.model.estimators_[:n_trees_to_show]):
            tree_pred = tree.predict(X_processed)[0]
            tree_prob = tree.predict_proba(X_processed)[0]
            
            tree_predictions.append({
                'tree_index': i,
                'prediction': int(tree_pred),
                'probabilities': tree_prob.tolist(),
                'confidence': float(max(tree_prob))
            })
        
        # Calculate prediction consensus | 計算預測一致性
        all_tree_preds = [self.model.estimators_[i].predict(X_processed)[0] 
                         for i in range(len(self.model.estimators_))]
        consensus = np.mean([pred == prediction for pred in all_tree_preds])
        
        return {
            'ensemble_prediction': int(prediction),
            'ensemble_probabilities': probabilities.tolist(),
            'ensemble_confidence': float(max(probabilities)),
            'individual_tree_predictions': tree_predictions,
            'prediction_consensus': float(consensus),
            'total_trees': len(self.model.estimators_),
            'trees_analyzed': n_trees_to_show
        }