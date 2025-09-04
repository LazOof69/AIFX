"""
LSTM Neural Network Implementation | LSTM神經網路實現
Deep learning model for time series forex price direction prediction.
用於時間序列外匯價格方向預測的深度學習模型。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import warnings

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    warnings.warn("TensorFlow not available. LSTM model will not function without TensorFlow.")

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class LSTMModel(BaseModel):
    """
    LSTM classifier for forex price direction prediction | 用於外匯價格方向預測的LSTM分類器
    
    Implements deep learning with LSTM layers for time series analysis.
    實現用於時間序列分析的LSTM層深度學習。
    """
    
    def __init__(self, model_name: str = "LSTM_Classifier", version: str = "1.0",
                 sequence_length: int = 60, n_features: Optional[int] = None):
        """
        Initialize LSTM model | 初始化LSTM模型
        
        Args:
            model_name: Name of the model | 模型名稱
            version: Model version | 模型版本
            sequence_length: Length of input sequences | 輸入序列長度
            n_features: Number of features | 特徵數量
        """
        super().__init__(model_name, version)
        
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM model. Please install: pip install tensorflow")
        
        self.sequence_length = sequence_length
        self.n_features = n_features
        
        # Default architecture parameters | 默認架構參數
        self.default_params = {
            'lstm_units': [50, 50],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'validation_split': 0.2,
            'patience': 10,
            'min_delta': 0.001
        }
        
        # Preprocessing components | 預處理組件
        self.scaler = MinMaxScaler()
        self.is_scaler_fitted = False
        
        # Model components | 模型組件
        self.best_params = None
        self.feature_names = None
        self.training_history_detailed = None
        
    def _prepare_sequences(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare sequential data for LSTM training | 為LSTM訓練準備序列數據
        
        Args:
            X: Input features | 輸入特徵
            y: Target values (optional) | 目標值（可選）
            
        Returns:
            Tuple of (X_sequences, y_sequences) | (X_序列, y_序列)的元組
        """
        # Store feature names | 存儲特徵名稱
        if self.feature_names is None:
            self.feature_names = list(X.columns)
            
        # Handle missing values | 處理缺失值
        X_processed = X.fillna(method='ffill').fillna(method='bfill')
        
        # Scale features if not already fitted | 如果尚未擬合，則縮放特徵
        if not self.is_scaler_fitted:
            X_scaled = self.scaler.fit_transform(X_processed)
            self.is_scaler_fitted = True
        else:
            X_scaled = self.scaler.transform(X_processed)
        
        # Create sequences | 創建序列
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i-self.sequence_length:i])
            if y is not None:
                y_sequences.append(y.iloc[i])
        
        X_sequences = np.array(X_sequences)
        
        if y is not None:
            y_sequences = np.array(y_sequences)
            return X_sequences, y_sequences
        else:
            return X_sequences, None
    
    def _build_model(self, input_shape: Tuple[int, int], **kwargs):  # -> tf.keras.Model when available
        """
        Build LSTM neural network architecture | 構建LSTM神經網絡架構
        
        Args:
            input_shape: Shape of input sequences | 輸入序列形狀
            **kwargs: Additional architecture parameters | 額外架構參數
            
        Returns:
            Compiled Keras model | 編譯好的Keras模型
        """
        params = {**self.default_params, **kwargs}
        
        model = Sequential()
        
        # Add LSTM layers | 添加LSTM層
        lstm_units = params['lstm_units']
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1  # Return sequences for all but last layer
            
            if i == 0:
                # First LSTM layer with input shape | 第一個LSTM層帶輸入形狀
                model.add(LSTM(
                    units, 
                    return_sequences=return_sequences,
                    input_shape=input_shape,
                    kernel_regularizer=l2(0.001)
                ))
            else:
                # Subsequent LSTM layers | 後續LSTM層
                model.add(LSTM(
                    units, 
                    return_sequences=return_sequences,
                    kernel_regularizer=l2(0.001)
                ))
            
            # Add dropout and batch normalization | 添加dropout和批量標準化
            model.add(Dropout(params['dropout_rate']))
            if i < len(lstm_units) - 1:  # Don't add BatchNorm after the last LSTM layer
                model.add(BatchNormalization())
        
        # Add dense layers | 添加密集層
        model.add(Dense(50, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(params['dropout_rate']))
        model.add(Dense(25, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(params['dropout_rate']))
        
        # Output layer for binary classification | 二分類的輸出層
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model | 編譯模型
        optimizer = Adam(learning_rate=params['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
              optimize_hyperparameters: bool = False, **kwargs) -> Dict[str, Any]:
        """
        Train LSTM model | 訓練LSTM模型
        
        Args:
            X_train: Training features | 訓練特徵
            y_train: Training targets | 訓練目標
            X_val: Validation features | 驗證特徵
            y_val: Validation targets | 驗證目標
            optimize_hyperparameters: Whether to perform hyperparameter optimization | 是否進行超參數優化
            **kwargs: Additional training parameters | 額外訓練參數
            
        Returns:
            Training history and metrics | 訓練歷史和指標
        """
        logger.info(f"Training {self.model_name} with {len(X_train)} samples")
        
        # Prepare sequences | 準備序列
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train)
        
        if len(X_train_seq) == 0:
            raise ValueError(f"Not enough data to create sequences. Need at least {self.sequence_length + 1} samples.")
        
        # Update metadata | 更新元數據
        self.metadata['training_samples'] = len(X_train_seq)
        self.metadata['features_used'] = list(X_train.columns)
        self.metadata['sequence_length'] = self.sequence_length
        
        # Set number of features | 設置特徵數量
        if self.n_features is None:
            self.n_features = X_train_seq.shape[2]
        
        training_history = {}
        
        # Prepare validation data if provided | 如果提供，準備驗證數據
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
            if len(X_val_seq) > 0:
                validation_data = (X_val_seq, y_val_seq)
        
        # Merge training parameters | 合併訓練參數
        params = {**self.default_params, **kwargs}
        
        if optimize_hyperparameters:
            logger.info("Hyperparameter optimization not implemented for LSTM. Using default parameters.")
            # Note: Hyperparameter optimization for neural networks is complex
            # Could implement using Keras Tuner or similar in the future
        
        # Build model | 構建模型
        input_shape = (self.sequence_length, self.n_features)
        self.model = self._build_model(input_shape, **params)
        
        # Log model architecture | 記錄模型架構
        logger.info(f"LSTM Model Architecture:")
        self.model.summary(print_fn=logger.info)
        
        # Setup callbacks | 設置回調
        callbacks = []
        
        # Early stopping | 早期停止
        early_stopping = EarlyStopping(
            monitor='val_loss' if validation_data else 'loss',
            patience=params['patience'],
            min_delta=params['min_delta'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction | 學習率降低
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss' if validation_data else 'loss',
            factor=0.5,
            patience=params['patience'] // 2,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # Model checkpointing | 模型檢查點
        checkpoint_path = f"models/checkpoints/{self.model_name}_checkpoint.h5"
        import os
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss' if validation_data else 'loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Train the model | 訓練模型
        history = self.model.fit(
            X_train_seq, y_train_seq,
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            validation_data=validation_data,
            validation_split=params['validation_split'] if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store training history | 存儲訓練歷史
        self.training_history_detailed = history.history
        training_history['detailed_history'] = history.history
        training_history['final_train_loss'] = history.history['loss'][-1]
        training_history['final_train_accuracy'] = history.history['accuracy'][-1]
        
        if 'val_loss' in history.history:
            training_history['final_val_loss'] = history.history['val_loss'][-1]
            training_history['final_val_accuracy'] = history.history['val_accuracy'][-1]
        
        training_history['epochs_trained'] = len(history.history['loss'])
        training_history['model_params'] = params
        
        # Store training history | 存儲訓練歷史
        self.training_history = training_history
        self.is_trained = True
        
        # Log training completion | 記錄訓練完成
        logger.info(f"Training completed after {training_history['epochs_trained']} epochs")
        logger.info(f"Final training accuracy: {training_history['final_train_accuracy']:.4f}")
        if 'final_val_accuracy' in training_history:
            logger.info(f"Final validation accuracy: {training_history['final_val_accuracy']:.4f}")
        
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
        
        X_sequences, _ = self._prepare_sequences(X)
        
        if len(X_sequences) == 0:
            raise ValueError(f"Not enough data to create sequences for prediction. Need at least {self.sequence_length} samples.")
        
        predictions = self.model.predict(X_sequences)
        return (predictions > 0.5).astype(int).flatten()
    
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
        
        X_sequences, _ = self._prepare_sequences(X)
        
        if len(X_sequences) == 0:
            raise ValueError(f"Not enough data to create sequences for prediction. Need at least {self.sequence_length} samples.")
        
        predictions = self.model.predict(X_sequences).flatten()
        
        # Convert to binary classification probabilities | 轉換為二分類機率
        proba = np.column_stack([1 - predictions, predictions])
        return proba
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history | 繪製訓練歷史
        
        Args:
            save_path: Optional path to save the plot | 保存圖表的可選路徑
        """
        if not self.is_trained or self.training_history_detailed is None:
            raise ValueError("Model must be trained to plot training history")
        
        try:
            import matplotlib.pyplot as plt
            
            history = self.training_history_detailed
            epochs = range(1, len(history['loss']) + 1)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot training & validation loss | 繪製訓練和驗證損失
            ax1.plot(epochs, history['loss'], 'bo-', label='Training Loss | 訓練損失')
            if 'val_loss' in history:
                ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss | 驗證損失')
            ax1.set_title('Model Loss | 模型損失')
            ax1.set_xlabel('Epoch | 周期')
            ax1.set_ylabel('Loss | 損失')
            ax1.legend()
            ax1.grid(True)
            
            # Plot training & validation accuracy | 繪製訓練和驗證準確度
            ax2.plot(epochs, history['accuracy'], 'bo-', label='Training Accuracy | 訓練準確度')
            if 'val_accuracy' in history:
                ax2.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Accuracy | 驗證準確度')
            ax2.set_title('Model Accuracy | 模型準確度')
            ax2.set_xlabel('Epoch | 周期')
            ax2.set_ylabel('Accuracy | 準確度')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training history plot saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")
    
    def get_model_architecture(self) -> Dict[str, Any]:
        """
        Get detailed model architecture information | 獲取詳細的模型架構信息
        
        Returns:
            Architecture information | 架構信息
        """
        if not self.is_trained:
            return {}
        
        # Count parameters | 計算參數
        total_params = self.model.count_params()
        if TENSORFLOW_AVAILABLE:
            trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        else:
            trainable_params = 0
        non_trainable_params = total_params - trainable_params
        
        # Get layer information | 獲取層信息
        layers_info = []
        for layer in self.model.layers:
            layer_info = {
                'name': layer.name,
                'type': type(layer).__name__,
                'output_shape': str(layer.output_shape),
                'params': layer.count_params()
            }
            
            # Add layer-specific info | 添加層特定信息
            if TENSORFLOW_AVAILABLE:
                if isinstance(layer, tf.keras.layers.LSTM):
                    layer_info['units'] = layer.units
                    layer_info['return_sequences'] = layer.return_sequences
                elif isinstance(layer, tf.keras.layers.Dense):
                    layer_info['units'] = layer.units
                    layer_info['activation'] = layer.activation.__name__
                elif isinstance(layer, tf.keras.layers.Dropout):
                    layer_info['rate'] = layer.rate
                
            layers_info.append(layer_info)
        
        return {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'layers': layers_info,
            'optimizer': self.model.optimizer.__class__.__name__,
            'loss_function': self.model.loss,
            'metrics': self.model.metrics_names
        }
    
    def save_model(self, filepath: str, include_metadata: bool = True):
        """
        Save LSTM model to file | 保存LSTM模型到文件
        
        Args:
            filepath: Path to save model | 保存模型的路徑
            include_metadata: Whether to save metadata | 是否保存元數據
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model | 無法保存未訓練的模型")
        
        # Create directory if it doesn't exist | 如果目錄不存在則創建
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save Keras model | 保存Keras模型
        model_path = filepath.replace('.pkl', '.h5')
        self.model.save(model_path)
        
        # Save additional data using joblib | 使用joblib保存額外數據
        import joblib
        model_data = {
            'model_name': self.model_name,
            'version': self.version,
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'scaler': self.scaler,
            'is_scaler_fitted': self.is_scaler_fitted,
            'feature_names': self.feature_names
        }
        
        if include_metadata:
            model_data['metadata'] = self.metadata
        
        data_path = filepath.replace('.h5', '_data.pkl')
        joblib.dump(model_data, data_path)
        
        logger.info(f"LSTM model saved to {model_path} and {data_path}")
    
    def load_model(self, filepath: str):
        """
        Load LSTM model from file | 從文件加載LSTM模型
        
        Args:
            filepath: Path to load model from | 加載模型的路徑
        """
        import joblib
        
        # Load Keras model | 加載Keras模型
        model_path = filepath.replace('.pkl', '.h5')
        if TENSORFLOW_AVAILABLE:
            self.model = tf.keras.models.load_model(model_path)
        else:
            raise ImportError("TensorFlow not available. Cannot load LSTM model.")
        
        # Load additional data | 加載額外數據
        data_path = filepath.replace('.h5', '_data.pkl')
        model_data = joblib.load(data_path)
        
        self.model_name = model_data['model_name']
        self.version = model_data['version']
        self.sequence_length = model_data['sequence_length']
        self.n_features = model_data['n_features']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data.get('training_history', {})
        self.scaler = model_data['scaler']
        self.is_scaler_fitted = model_data['is_scaler_fitted']
        self.feature_names = model_data['feature_names']
        
        if 'metadata' in model_data:
            self.metadata = model_data['metadata']
        
        logger.info(f"LSTM model loaded from {model_path} and {data_path}")
    
    def get_attention_weights(self, X_sample: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get attention weights for input sequence (if attention is implemented) | 獲取輸入序列的注意力權重（如果實現了注意力機制）
        
        Args:
            X_sample: Input sample | 輸入樣本
            
        Returns:
            Attention weights or None | 注意力權重或None
        """
        # Note: This would require implementing attention mechanism in the model
        # For now, return None as basic LSTM doesn't have attention
        logger.info("Attention mechanism not implemented in basic LSTM model")
        return None