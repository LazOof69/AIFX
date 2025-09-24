"""
Performance Metrics and Evaluation | 性能指標和評估
Comprehensive evaluation metrics for AI models in forex trading context.
外匯交易環境中AI模型的綜合評估指標。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings

# Sklearn metrics | Sklearn指標
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    log_loss, matthews_corrcoef
)

# Visualization | 可視化
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib/Seaborn not available. Plotting functions will not work.")

logger = logging.getLogger(__name__)


class TradingPerformanceMetrics:
    """
    Specialized performance metrics for trading models | 交易模型的專用性能指標
    
    Combines standard ML metrics with trading-specific evaluations.
    結合標準ML指標與交易特定評估。
    """
    
    def __init__(self):
        """Initialize performance metrics calculator | 初始化性能指標計算器"""
        self.metrics_cache = {}
        
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics | 計算全面的分類指標
        
        Args:
            y_true: True labels | 真實標籤
            y_pred: Predicted labels | 預測標籤
            y_pred_proba: Predicted probabilities | 預測概率
            
        Returns:
            Dictionary of classification metrics | 分類指標字典
        """
        metrics = {}
        
        try:
            # Basic metrics | 基本指標
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Binary classification specific | 二分類特定
            if len(np.unique(y_true)) == 2:
                metrics['precision_binary'] = precision_score(y_true, y_pred, zero_division=0)
                metrics['recall_binary'] = recall_score(y_true, y_pred, zero_division=0)
                metrics['f1_binary'] = f1_score(y_true, y_pred, zero_division=0)
                metrics['specificity'] = self._calculate_specificity(y_true, y_pred)
                
                # ROC AUC if probabilities provided | 如果提供概率則計算ROC AUC
                if y_pred_proba is not None:
                    if y_pred_proba.ndim == 2:
                        y_pred_proba_binary = y_pred_proba[:, 1]
                    else:
                        y_pred_proba_binary = y_pred_proba
                        
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba_binary)
                    metrics['log_loss'] = log_loss(y_true, y_pred_proba_binary)
            
            # Matthews Correlation Coefficient | 馬修斯相關係數
            metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
            
            # Balanced accuracy | 平衡準確度
            metrics['balanced_accuracy'] = self._calculate_balanced_accuracy(y_true, y_pred)
            
        except Exception as e:
            logger.warning(f"Error calculating classification metrics: {str(e)}")
            metrics['error'] = str(e)
            
        return metrics
    
    def calculate_trading_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_pred_proba: Optional[np.ndarray] = None,
                                returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate trading-specific performance metrics | 計算交易特定的性能指標
        
        Args:
            y_true: True direction (1 for up, 0 for down) | 真實方向（1上漲，0下跌）
            y_pred: Predicted direction | 預測方向
            y_pred_proba: Prediction confidence | 預測信心
            returns: Actual returns if available | 實際回報（如果可用）
            
        Returns:
            Dictionary of trading metrics | 交易指標字典
        """
        metrics = {}
        
        try:
            # Directional accuracy | 方向準確度
            metrics['directional_accuracy'] = accuracy_score(y_true, y_pred)
            
            # Calculate win rate and loss rate | 計算勝率和損失率
            correct_predictions = (y_true == y_pred)
            metrics['win_rate'] = np.mean(correct_predictions)
            metrics['loss_rate'] = 1 - metrics['win_rate']
            
            # Up/Down prediction accuracy | 上漲/下跌預測準確度
            up_mask = (y_true == 1)
            down_mask = (y_true == 0)
            
            if np.sum(up_mask) > 0:
                metrics['up_prediction_accuracy'] = np.mean(y_pred[up_mask] == 1)
            else:
                metrics['up_prediction_accuracy'] = 0.0
                
            if np.sum(down_mask) > 0:
                metrics['down_prediction_accuracy'] = np.mean(y_pred[down_mask] == 0)
            else:
                metrics['down_prediction_accuracy'] = 0.0
            
            # Prediction confidence analysis | 預測信心分析
            if y_pred_proba is not None:
                if y_pred_proba.ndim == 2:
                    confidence = np.max(y_pred_proba, axis=1)
                else:
                    confidence = np.abs(y_pred_proba - 0.5) + 0.5
                    
                metrics['mean_confidence'] = np.mean(confidence)
                metrics['confidence_std'] = np.std(confidence)
                
                # High confidence accuracy | 高信心準確度
                high_conf_threshold = np.percentile(confidence, 75)
                high_conf_mask = confidence >= high_conf_threshold
                if np.sum(high_conf_mask) > 0:
                    metrics['high_confidence_accuracy'] = np.mean(correct_predictions[high_conf_mask])
                    metrics['high_confidence_samples'] = np.sum(high_conf_mask)
                else:
                    metrics['high_confidence_accuracy'] = 0.0
                    metrics['high_confidence_samples'] = 0
            
            # Calculate hypothetical trading performance | 計算假設交易表現
            if returns is not None:
                trading_returns = self._calculate_trading_returns(y_true, y_pred, returns)
                metrics.update(trading_returns)
            
        except Exception as e:
            logger.warning(f"Error calculating trading metrics: {str(e)}")
            metrics['error'] = str(e)
            
        return metrics
    
    def _calculate_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate) | 計算特異性（真陰性率）"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        if (tn + fp) == 0:
            return 0.0
        return tn / (tn + fp)
    
    def _calculate_balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy | 計算平衡準確度"""
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = self._calculate_specificity(y_true, y_pred)
        return (sensitivity + specificity) / 2
    
    def _calculate_trading_returns(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 returns: np.ndarray) -> Dict[str, float]:
        """
        Calculate hypothetical trading returns | 計算假設交易回報
        
        Args:
            y_true: True direction | 真實方向
            y_pred: Predicted direction | 預測方向
            returns: Actual price returns | 實際價格回報
            
        Returns:
            Trading performance metrics | 交易表現指標
        """
        # Strategy returns: go long if pred=1, short if pred=0 | 策略回報：預測=1做多，預測=0做空
        strategy_returns = np.where(y_pred == 1, returns, -returns)
        
        # Buy and hold returns | 買入並持有回報
        buy_hold_returns = returns
        
        # Calculate cumulative returns | 計算累積回報
        cumulative_strategy = (1 + strategy_returns).cumprod()
        cumulative_buy_hold = (1 + buy_hold_returns).cumprod()
        
        # Performance metrics | 表現指標
        trading_metrics = {
            'total_return_strategy': cumulative_strategy[-1] - 1,
            'total_return_buy_hold': cumulative_buy_hold[-1] - 1,
            'annualized_return_strategy': np.mean(strategy_returns) * 252,  # Assuming daily returns
            'annualized_return_buy_hold': np.mean(buy_hold_returns) * 252,
            'volatility_strategy': np.std(strategy_returns) * np.sqrt(252),
            'volatility_buy_hold': np.std(buy_hold_returns) * np.sqrt(252),
            'win_ratio': np.mean(strategy_returns > 0),
            'average_win': np.mean(strategy_returns[strategy_returns > 0]) if np.any(strategy_returns > 0) else 0,
            'average_loss': np.mean(strategy_returns[strategy_returns < 0]) if np.any(strategy_returns < 0) else 0,
        }
        
        # Sharpe ratio | 夏普比率
        if trading_metrics['volatility_strategy'] > 0:
            trading_metrics['sharpe_ratio_strategy'] = trading_metrics['annualized_return_strategy'] / trading_metrics['volatility_strategy']
        else:
            trading_metrics['sharpe_ratio_strategy'] = 0
            
        if trading_metrics['volatility_buy_hold'] > 0:
            trading_metrics['sharpe_ratio_buy_hold'] = trading_metrics['annualized_return_buy_hold'] / trading_metrics['volatility_buy_hold']
        else:
            trading_metrics['sharpe_ratio_buy_hold'] = 0
        
        # Maximum drawdown | 最大回撤
        trading_metrics['max_drawdown_strategy'] = self._calculate_max_drawdown(cumulative_strategy)
        trading_metrics['max_drawdown_buy_hold'] = self._calculate_max_drawdown(cumulative_buy_hold)
        
        # Information ratio (excess return / tracking error) | 信息比率（超額回報/跟踪誤差）
        excess_returns = strategy_returns - buy_hold_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        if tracking_error > 0:
            trading_metrics['information_ratio'] = np.mean(excess_returns) * 252 / tracking_error
        else:
            trading_metrics['information_ratio'] = 0
        
        return trading_metrics
    
    def _calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown | 計算最大回撤"""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)
    
    def generate_performance_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_pred_proba: Optional[np.ndarray] = None,
                                  returns: Optional[np.ndarray] = None,
                                  model_name: str = "Model") -> Dict[str, Any]:
        """
        Generate comprehensive performance report | 生成全面的性能報告
        
        Args:
            y_true: True labels | 真實標籤
            y_pred: Predicted labels | 預測標籤
            y_pred_proba: Predicted probabilities | 預測概率
            returns: Actual returns | 實際回報
            model_name: Name of the model | 模型名稱
            
        Returns:
            Comprehensive performance report | 全面性能報告
        """
        report = {
            'model_name': model_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'sample_size': len(y_true),
            'class_distribution': {
                str(k): int(v) for k, v in zip(*np.unique(y_true, return_counts=True))
            }
        }
        
        # Classification metrics | 分類指標
        report['classification_metrics'] = self.calculate_classification_metrics(
            y_true, y_pred, y_pred_proba
        )
        
        # Trading metrics | 交易指標
        report['trading_metrics'] = self.calculate_trading_metrics(
            y_true, y_pred, y_pred_proba, returns
        )
        
        # Confusion matrix | 混淆矩陣
        report['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Classification report | 分類報告
        report['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        
        return report
    
    def plot_performance_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                y_pred_proba: Optional[np.ndarray] = None,
                                returns: Optional[np.ndarray] = None,
                                model_name: str = "Model",
                                save_path: Optional[str] = None):
        """
        Create comprehensive performance analysis plots | 創建全面的性能分析圖表
        
        Args:
            y_true: True labels | 真實標籤
            y_pred: Predicted labels | 預測標籤
            y_pred_proba: Predicted probabilities | 預測概率
            returns: Actual returns | 實際回報
            model_name: Name of the model | 模型名稱
            save_path: Optional path to save plots | 保存圖表的可選路徑
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Confusion Matrix | 混淆矩陣
        plt.subplot(3, 4, 1)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix | 混淆矩陣')
        plt.ylabel('True Label | 真實標籤')
        plt.xlabel('Predicted Label | 預測標籤')
        
        # 2. ROC Curve | ROC曲線
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            plt.subplot(3, 4, 2)
            if y_pred_proba.ndim == 2:
                y_prob = y_pred_proba[:, 1]
            else:
                y_prob = y_pred_proba
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_score = roc_auc_score(y_true, y_prob)
            
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate | 假陽性率')
            plt.ylabel('True Positive Rate | 真陽性率')
            plt.title('ROC Curve | ROC曲線')
            plt.legend()
            plt.grid(True)
        
        # 3. Precision-Recall Curve | 精確率-召回率曲線
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            plt.subplot(3, 4, 3)
            if y_pred_proba.ndim == 2:
                y_prob = y_pred_proba[:, 1]
            else:
                y_prob = y_pred_proba
                
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            plt.plot(recall, precision)
            plt.xlabel('Recall | 召回率')
            plt.ylabel('Precision | 精確率')
            plt.title('Precision-Recall Curve | 精確率-召回率曲線')
            plt.grid(True)
        
        # 4. Prediction Confidence Distribution | 預測信心分佈
        if y_pred_proba is not None:
            plt.subplot(3, 4, 4)
            if y_pred_proba.ndim == 2:
                confidence = np.max(y_pred_proba, axis=1)
            else:
                confidence = np.abs(y_pred_proba - 0.5) + 0.5
                
            plt.hist(confidence, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Prediction Confidence | 預測信心')
            plt.ylabel('Frequency | 頻率')
            plt.title('Confidence Distribution | 信心分佈')
            plt.grid(True)
        
        # 5. Accuracy by Confidence Level | 按信心水平的準確度
        if y_pred_proba is not None:
            plt.subplot(3, 4, 5)
            if y_pred_proba.ndim == 2:
                confidence = np.max(y_pred_proba, axis=1)
            else:
                confidence = np.abs(y_pred_proba - 0.5) + 0.5
            
            # Bin by confidence | 按信心分箱
            n_bins = 10
            conf_bins = np.linspace(0.5, 1.0, n_bins + 1)
            bin_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
            bin_accuracies = []
            
            for i in range(n_bins):
                mask = (confidence >= conf_bins[i]) & (confidence < conf_bins[i + 1])
                if np.sum(mask) > 0:
                    accuracy = np.mean(y_true[mask] == y_pred[mask])
                    bin_accuracies.append(accuracy)
                else:
                    bin_accuracies.append(0)
            
            plt.plot(bin_centers, bin_accuracies, 'bo-')
            plt.xlabel('Confidence Level | 信心水平')
            plt.ylabel('Accuracy | 準確度')
            plt.title('Accuracy vs Confidence | 準確度vs信心')
            plt.grid(True)
        
        # 6-8. Trading Performance if returns available | 如果有回報數據則顯示交易表現
        if returns is not None:
            # 6. Cumulative Returns | 累積回報
            plt.subplot(3, 4, 6)
            strategy_returns = np.where(y_pred == 1, returns, -returns)
            cum_strategy = (1 + strategy_returns).cumprod()
            cum_buy_hold = (1 + returns).cumprod()
            
            plt.plot(cum_strategy, label='Strategy | 策略', linewidth=2)
            plt.plot(cum_buy_hold, label='Buy & Hold | 買入持有', linewidth=2)
            plt.xlabel('Time | 時間')
            plt.ylabel('Cumulative Return | 累積回報')
            plt.title('Strategy Performance | 策略表現')
            plt.legend()
            plt.grid(True)
            
            # 7. Returns Distribution | 回報分佈
            plt.subplot(3, 4, 7)
            plt.hist(strategy_returns, bins=50, alpha=0.7, label='Strategy', density=True)
            plt.hist(returns, bins=50, alpha=0.7, label='Buy & Hold', density=True)
            plt.xlabel('Returns | 回報')
            plt.ylabel('Density | 密度')
            plt.title('Returns Distribution | 回報分佈')
            plt.legend()
            plt.grid(True)
            
            # 8. Rolling Sharpe Ratio | 滾動夏普比率
            plt.subplot(3, 4, 8)
            window = min(252, len(strategy_returns) // 4)  # Quarterly window or smaller
            if window > 30:
                rolling_sharpe_strategy = pd.Series(strategy_returns).rolling(window).mean() / pd.Series(strategy_returns).rolling(window).std() * np.sqrt(252)
                rolling_sharpe_buy_hold = pd.Series(returns).rolling(window).mean() / pd.Series(returns).rolling(window).std() * np.sqrt(252)
                
                plt.plot(rolling_sharpe_strategy, label='Strategy | 策略')
                plt.plot(rolling_sharpe_buy_hold, label='Buy & Hold | 買入持有')
                plt.xlabel('Time | 時間')
                plt.ylabel('Rolling Sharpe Ratio | 滾動夏普比率')
                plt.title(f'{window}-Period Rolling Sharpe | {window}期滾動夏普')
                plt.legend()
                plt.grid(True)
        
        # 9. Feature Importance (placeholder) | 特徵重要性（佔位符）
        plt.subplot(3, 4, 9)
        plt.text(0.5, 0.5, 'Feature Importance\n(Available in model-specific\nevaluation)', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Importance | 特徵重要性')
        
        # 10. Metrics Summary | 指標摘要
        plt.subplot(3, 4, 10)
        metrics = self.calculate_classification_metrics(y_true, y_pred, y_pred_proba)
        metrics_text = f"""Accuracy: {metrics.get('accuracy', 0):.3f}
Precision: {metrics.get('precision', 0):.3f}
Recall: {metrics.get('recall', 0):.3f}
F1-Score: {metrics.get('f1_score', 0):.3f}"""
        if 'auc_roc' in metrics:
            metrics_text += f"\nAUC: {metrics['auc_roc']:.3f}"
        
        plt.text(0.1, 0.7, metrics_text, transform=plt.gca().transAxes, 
                fontsize=12, verticalalignment='top')
        plt.axis('off')
        plt.title('Performance Summary | 性能摘要')
        
        # 11. Error Analysis | 錯誤分析
        plt.subplot(3, 4, 11)
        correct = (y_true == y_pred)
        error_by_class = pd.DataFrame({
            'True_Class': y_true,
            'Pred_Class': y_pred,
            'Correct': correct
        })
        
        error_summary = error_by_class.groupby(['True_Class', 'Correct']).size().unstack(fill_value=0)
        if error_summary.shape[1] == 2:
            error_rates = error_summary[False] / (error_summary[False] + error_summary[True])
            error_rates.plot(kind='bar')
            plt.xlabel('True Class | 真實類別')
            plt.ylabel('Error Rate | 錯誤率')
            plt.title('Error Rate by Class | 各類別錯誤率')
            plt.xticks(rotation=45)
        
        # 12. Model Comparison (placeholder) | 模型比較（佔位符）
        plt.subplot(3, 4, 12)
        plt.text(0.5, 0.5, f'Model: {model_name}\nEvaluation Date:\n{datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Model Info | 模型信息')
        plt.axis('off')
        
        plt.tight_layout()
        plt.suptitle(f'{model_name} - Performance Analysis | 性能分析', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance analysis plot saved to {save_path}")
        
        plt.show()
    
    def benchmark_model_speed(self, model, X_test: pd.DataFrame, 
                            n_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark model inference speed | 基準測試模型推理速度
        
        Args:
            model: Trained model instance | 訓練好的模型實例
            X_test: Test data | 測試數據
            n_iterations: Number of iterations for benchmarking | 基準測試的迭代次數
            
        Returns:
            Speed benchmark results | 速度基準測試結果
        """
        import time
        
        prediction_times = []
        proba_times = []
        
        # Warm up | 預熱
        for _ in range(5):
            model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                model.predict_proba(X_test)
        
        # Benchmark prediction | 基準測試預測
        for _ in range(n_iterations):
            start_time = time.time()
            model.predict(X_test)
            prediction_times.append(time.time() - start_time)
        
        # Benchmark probability prediction | 基準測試概率預測
        if hasattr(model, 'predict_proba'):
            for _ in range(n_iterations):
                start_time = time.time()
                model.predict_proba(X_test)
                proba_times.append(time.time() - start_time)
        
        results = {
            'samples_tested': len(X_test),
            'iterations': n_iterations,
            'avg_prediction_time': np.mean(prediction_times),
            'std_prediction_time': np.std(prediction_times),
            'predictions_per_second': len(X_test) / np.mean(prediction_times),
            'min_prediction_time': np.min(prediction_times),
            'max_prediction_time': np.max(prediction_times)
        }
        
        if proba_times:
            results.update({
                'avg_proba_time': np.mean(proba_times),
                'std_proba_time': np.std(proba_times),
                'proba_per_second': len(X_test) / np.mean(proba_times),
                'min_proba_time': np.min(proba_times),
                'max_proba_time': np.max(proba_times)
            })
        
        return results


# Backward compatibility aliases | 向後兼容別名
PerformanceMetrics = TradingPerformanceMetrics  # Main performance metrics class alias