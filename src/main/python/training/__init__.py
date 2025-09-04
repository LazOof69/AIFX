# Model Training Scripts | 模型訓練腳本
# Scripts and utilities for training AI models.
# 訓練AI模型的腳本和工具。

"""
Model training components for AIFX system:
AIFX系統的模型訓練組件：

- Training pipelines | 訓練管道
- Hyperparameter optimization | 超參數優化
- Cross-validation procedures | 交叉驗證程序
- Model persistence and versioning | 模型持久化和版本控制
- Multi-model comparison | 多模型比較
- Automated training workflows | 自動化訓練工作流程
"""

# Import training pipeline | 導入訓練管道
from .model_pipeline import ModelTrainingPipeline

__all__ = [
    'ModelTrainingPipeline'
]