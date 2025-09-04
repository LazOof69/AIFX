"""
XGBoost Model Implementation | XGBoost模型實現
High-performance gradient boosting model for forex price direction prediction.
用於外匯價格方向預測的高性能梯度提升模型。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report
import xgboost as xgb
from .base_model import BaseModel

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost classifier for forex price direction prediction | 用於外匯價格方向預測的XGBoost分類器
    
    Implements gradient boosting with advanced hyperparameter optimization.
    實現帶有高級超參數優化的梯度提升。
    """
    
    def __init__(self, model_name: str = "XGBoost_Classifier", version: str = "1.0"):
        """
        Initialize XGBoost model | 初始化XGBoost模型
        
        Args:
            model_name: Name of the model | 模型名稱
            version: Model version | 模型版本
        """
        super().__init__(model_name, version)
        
        # Default hyperparameters | 默認超參數
        self.default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.best_params = None
        self.feature_names = None
        self.cv_scores = None
        
    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for XGBoost training | 為XGBoost訓練準備特徵
        
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
        
        # Fill any remaining NaN values with 0 | 用0填充任何剩餘的NaN值
        X_processed = X_processed.fillna(0)
        
        return X_processed
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              optimize_hyperparameters: bool = True, cv_folds: int = 5,
              **kwargs) -> Dict[str, Any]:
        """
        Train XGBoost model with optional hyperparameter optimization | 訓練帶有可選超參數優化的XGBoost模型
        
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
                'max_depth': [3, 4, 5, 6, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                'n_estimators': [50, 100, 150, 200],
                'subsample': [0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
            }
            
            # Initialize base model | 初始化基礎模型
            base_model = xgb.XGBClassifier(**self.default_params)
            
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
            self.model = xgb.XGBClassifier(**params)
            
            # Train the model | 訓練模型
            if X_val is not None and y_val is not None:
                X_val_processed = self._prepare_features(X_val)
                eval_set = [(X_train_processed, y_train), (X_val_processed, y_val)]
                
                self.model.fit(
                    X_train_processed, y_train,
                    eval_set=eval_set,
                    verbose=False
                )
                
                # Get evaluation results | 獲取評估結果
                evals_result = self.model.evals_result()
                training_history['evals_result'] = evals_result
            else:
                self.model.fit(X_train_processed, y_train)
        
        # Perform cross-validation | 執行交叉驗證
        self.cv_scores = cross_val_score(
            self.model, X_train_processed, y_train, 
            cv=cv_folds, scoring='accuracy'
        )
        
        training_history['cv_scores'] = self.cv_scores.tolist()
        training_history['cv_mean'] = self.cv_scores.mean()
        training_history['cv_std'] = self.cv_scores.std()
        
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
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance | 特徵重要性')
            plt.title(f'{self.model_name} - Top {top_n} Feature Importances | 前{top_n}個特徵重要性')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Feature importance plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """
        Get model complexity information | 獲取模型複雜度信息
        
        Returns:
            Model complexity metrics | 模型複雜度指標
        """
        if not self.is_trained:
            return {}
        
        return {
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'learning_rate': self.model.learning_rate,
            'total_parameters': self.model.n_estimators * (2 ** self.model.max_depth - 1)
        }
    
    def explain_prediction(self, X_sample: pd.DataFrame, top_features: int = 10) -> Dict[str, Any]:
        """
        Explain individual prediction | 解釋個別預測
        
        Args:
            X_sample: Single sample to explain | 要解釋的單個樣本
            top_features: Number of top contributing features | 頂級貢獻特徵數量
            
        Returns:
            Explanation dictionary | 解釋字典
        """
        if not self.is_trained:
            raise ValueError("Model must be trained for explanation")
        
        if len(X_sample) != 1:
            raise ValueError("Only single sample explanation supported")
        
        # Get prediction and probability | 獲取預測和機率
        X_processed = self._prepare_features(X_sample)
        prediction = self.model.predict(X_processed)[0]
        probabilities = self.model.predict_proba(X_processed)[0]
        
        # Get feature values for the sample | 獲取樣本的特徵值
        feature_values = X_processed.iloc[0].to_dict()
        
        # Get feature importance | 獲取特徵重要性
        feature_importance = self.get_feature_importance()
        
        # Calculate contribution (simplified) | 計算貢獻（簡化）
        contributions = {}
        if feature_importance:
            for feature, importance in feature_importance.items():
                if feature in feature_values:
                    contributions[feature] = importance * feature_values[feature]
        
        # Get top contributing features | 獲取頂級貢獻特徵
        top_contributions = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:top_features]
        
        return {
            'prediction': int(prediction),
            'probabilities': probabilities.tolist(),
            'confidence': float(max(probabilities)),
            'top_contributing_features': top_contributions,
            'feature_values': feature_values
        }