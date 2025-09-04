"""
AIFX Phase 2 Comprehensive Test Suite | AIFX第二階段綜合測試套件

This script performs comprehensive testing of all Phase 2 AI model components.
此腳本對所有第二階段AI模型組件進行綜合測試。

Usage | 使用方法:
    python test_phase2_complete.py

Features | 功能:
- Base model framework validation | 基礎模型框架驗證
- XGBoost model implementation test | XGBoost模型實現測試
- Random Forest model implementation test | 隨機森林模型實現測試
- LSTM model implementation test | LSTM模型實現測試
- Training pipeline validation | 訓練管道驗證
- Performance metrics testing | 性能指標測試
- Model management system test | 模型管理系統測試
"""

import sys
import os
import traceback
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add src path for imports | 為導入添加src路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'main', 'python'))

class Colors:
    """Console colors for output formatting | 輸出格式的控制台顏色"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class Phase2Tester:
    """
    Comprehensive Phase 2 AI model testing class | 綜合第二階段AI模型測試類
    """
    
    def __init__(self):
        """Initialize the tester | 初始化測試器"""
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = []
        
        # Test start time | 測試開始時間
        self.start_time = datetime.now()
        
        print(f"{Colors.BOLD}{Colors.BLUE}🤖 AIFX Phase 2 AI Model Testing Suite{Colors.END}")
        print(f"{Colors.CYAN}Testing AI model implementations... | 正在測試AI模型實現...{Colors.END}\n")
    
    def log_test(self, test_name: str, passed: bool, message: str = "", warning: str = ""):
        """Log test result | 記錄測試結果"""
        self.total_tests += 1
        
        if passed:
            self.passed_tests += 1
            status = f"{Colors.GREEN}✅ PASS{Colors.END}"
        else:
            self.failed_tests += 1
            status = f"{Colors.RED}❌ FAIL{Colors.END}"
        
        self.test_results.append({
            'test': test_name,
            'passed': passed,
            'message': message,
            'warning': warning
        })
        
        print(f"{status} {test_name}")
        if message:
            print(f"    {Colors.WHITE}{message}{Colors.END}")
        if warning:
            print(f"    {Colors.YELLOW}⚠️ {warning}{Colors.END}")
            self.warnings.append(warning)
    
    def create_sample_data(self, n_samples=1000):
        """Create sample forex data for testing | 創建用於測試的樣本外匯數據"""
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='H')
        
        # Generate realistic forex price movements | 生成真實的外匯價格變動
        base_price = 1.1000
        returns = np.random.normal(0, 0.001, n_samples)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Create OHLCV data | 創建OHLCV數據
        data = pd.DataFrame({
            'Open': [p + np.random.normal(0, 0.0001) for p in prices],
            'High': [p + abs(np.random.normal(0, 0.0003)) for p in prices],
            'Low': [p - abs(np.random.normal(0, 0.0003)) for p in prices],
            'Close': prices,
            'Volume': np.random.uniform(1000, 5000, n_samples)
        }, index=dates)
        
        # Fix OHLC relationships | 修復OHLC關係
        for i in range(len(data)):
            max_oc = max(data.iloc[i]['Open'], data.iloc[i]['Close'])
            min_oc = min(data.iloc[i]['Open'], data.iloc[i]['Close'])
            data.iloc[i]['High'] = max(data.iloc[i]['High'], max_oc)
            data.iloc[i]['Low'] = min(data.iloc[i]['Low'], min_oc)
        
        return data
    
    def test_ai_dependencies(self):
        """Test AI-specific dependencies | 測試AI特定依賴項"""
        print(f"{Colors.BOLD}1. AI Dependencies Tests | AI依賴項測試{Colors.END}")
        
        # Test deep learning dependencies | 測試深度學習依賴項
        ai_deps = [
            ('tensorflow', 'Deep learning framework'),
            ('keras', 'High-level neural networks API'), 
            ('xgboost', 'Gradient boosting framework'),
            ('joblib', 'Model serialization'),
            ('sklearn', 'Machine learning library')
        ]
        
        for dep, description in ai_deps:
            try:
                if dep == 'sklearn':
                    import sklearn
                else:
                    __import__(dep)
                self.log_test(f"AI Dependency: {dep}", True, description)
            except ImportError:
                self.log_test(f"AI Dependency: {dep}", False, f"Missing: {description}")
    
    def test_base_model_framework(self):
        """Test base model framework | 測試基礎模型框架"""
        print(f"\n{Colors.BOLD}2. Base Model Framework Tests | 基礎模型框架測試{Colors.END}")
        
        try:
            # Test imports | 測試導入
            from models.base_model import BaseModel, ModelRegistry
            self.log_test("Base Model Import", True)
            
            # Test ModelRegistry | 測試模型註冊表
            registry = ModelRegistry()
            self.log_test("Model Registry Creation", True)
            
            # Test registry operations | 測試註冊表操作
            registry.register_model("test_model", "TestModel", "1.0")
            models = registry.list_models()
            if "test_model" in models:
                self.log_test("Model Registration", True, "Model registered successfully")
            else:
                self.log_test("Model Registration", False, "Failed to register model")
            
            # Test abstract methods exist | 測試抽象方法存在
            abstract_methods = ['train', 'predict', 'evaluate', 'save_model', 'load_model']
            for method in abstract_methods:
                if hasattr(BaseModel, method):
                    self.log_test(f"Abstract Method: {method}", True)
                else:
                    self.log_test(f"Abstract Method: {method}", False, "Method not found")
            
        except Exception as e:
            self.log_test("Base Model Framework", False, f"Error: {str(e)}")
    
    def test_xgboost_model(self):
        """Test XGBoost model implementation | 測試XGBoost模型實現"""
        print(f"\n{Colors.BOLD}3. XGBoost Model Tests | XGBoost模型測試{Colors.END}")
        
        try:
            # Test import | 測試導入
            from models.xgboost_model import XGBoostModel
            self.log_test("XGBoost Model Import", True)
            
            # Test model creation | 測試模型創建
            model = XGBoostModel()
            self.log_test("XGBoost Model Creation", True)
            
            # Create sample data | 創建樣本數據
            sample_data = self.create_sample_data(500)
            
            # Add basic features | 添加基本特徵
            sample_data['returns'] = sample_data['Close'].pct_change()
            sample_data['sma_10'] = sample_data['Close'].rolling(10).mean()
            sample_data['rsi'] = self._calculate_rsi(sample_data['Close'])
            sample_data = sample_data.dropna()
            
            if len(sample_data) < 100:
                self.log_test("XGBoost Data Preparation", False, "Insufficient data after preprocessing")
                return
                
            # Prepare features and target | 準備特徵和目標
            feature_cols = ['returns', 'sma_10', 'rsi']
            X = sample_data[feature_cols].fillna(method='ffill').fillna(method='bfill')
            y = (sample_data['Close'].shift(-1) > sample_data['Close']).astype(int)
            y = y[:-1]  # Remove last NaN
            X = X[:-1]  # Match lengths
            
            if len(X) != len(y):
                self.log_test("XGBoost Data Alignment", False, f"X shape: {X.shape}, y length: {len(y)}")
                return
            
            # Split data | 分割數據
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            self.log_test("XGBoost Data Preparation", True, 
                         f"Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Test training | 測試訓練
            training_history = model.train(X_train, y_train)
            if training_history and model.is_trained:
                self.log_test("XGBoost Training", True, "Model trained successfully")
            else:
                self.log_test("XGBoost Training", False, "Training failed")
                return
            
            # Test prediction | 測試預測
            predictions = model.predict(X_test)
            if len(predictions) == len(X_test):
                self.log_test("XGBoost Prediction", True, f"Predicted {len(predictions)} samples")
            else:
                self.log_test("XGBoost Prediction", False, "Prediction size mismatch")
            
            # Test probability prediction | 測試概率預測
            probabilities = model.predict_proba(X_test)
            if probabilities is not None and len(probabilities) == len(X_test):
                self.log_test("XGBoost Probabilities", True, "Probability prediction working")
            else:
                self.log_test("XGBoost Probabilities", False, "Probability prediction failed")
            
            # Test feature importance | 測試特徵重要性
            importance = model.get_feature_importance()
            if importance is not None and len(importance) == len(feature_cols):
                self.log_test("XGBoost Feature Importance", True, "Feature importance extracted")
            else:
                self.log_test("XGBoost Feature Importance", False, "Feature importance failed")
            
        except Exception as e:
            self.log_test("XGBoost Model", False, f"Error: {str(e)}")
            traceback.print_exc()
    
    def test_random_forest_model(self):
        """Test Random Forest model implementation | 測試隨機森林模型實現"""
        print(f"\n{Colors.BOLD}4. Random Forest Model Tests | 隨機森林模型測試{Colors.END}")
        
        try:
            # Test import | 測試導入
            from models.random_forest_model import RandomForestModel
            self.log_test("Random Forest Model Import", True)
            
            # Test model creation | 測試模型創建
            model = RandomForestModel()
            self.log_test("Random Forest Model Creation", True)
            
            # Use same sample data as XGBoost | 使用與XGBoost相同的樣本數據
            sample_data = self.create_sample_data(400)
            
            # Add features | 添加特徵
            sample_data['returns'] = sample_data['Close'].pct_change()
            sample_data['sma_5'] = sample_data['Close'].rolling(5).mean()
            sample_data['volume_ma'] = sample_data['Volume'].rolling(10).mean()
            sample_data = sample_data.dropna()
            
            if len(sample_data) < 100:
                self.log_test("Random Forest Data Preparation", False, "Insufficient data")
                return
            
            # Prepare features and target | 準備特徵和目標
            feature_cols = ['returns', 'sma_5', 'volume_ma']
            X = sample_data[feature_cols].fillna(method='ffill').fillna(method='bfill')
            y = (sample_data['Close'].shift(-1) > sample_data['Close']).astype(int)
            y = y[:-1]
            X = X[:-1]
            
            # Split data | 分割數據
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            self.log_test("Random Forest Data Preparation", True,
                         f"Train: {len(X_train)}, Test: {len(X_test)}")
            
            # Test training | 測試訓練
            training_history = model.train(X_train, y_train)
            if training_history and model.is_trained:
                self.log_test("Random Forest Training", True, "Model trained successfully")
            else:
                self.log_test("Random Forest Training", False, "Training failed")
                return
            
            # Test prediction | 測試預測
            predictions = model.predict(X_test)
            if len(predictions) == len(X_test):
                self.log_test("Random Forest Prediction", True, f"Predicted {len(predictions)} samples")
            else:
                self.log_test("Random Forest Prediction", False, "Prediction size mismatch")
            
            # Test ensemble statistics | 測試集成統計
            ensemble_stats = model.get_ensemble_statistics()
            if ensemble_stats and 'n_estimators' in ensemble_stats:
                self.log_test("Random Forest Ensemble Stats", True, 
                             f"Trees: {ensemble_stats['n_estimators']}")
            else:
                self.log_test("Random Forest Ensemble Stats", False, "Ensemble stats failed")
            
        except Exception as e:
            self.log_test("Random Forest Model", False, f"Error: {str(e)}")
            traceback.print_exc()
    
    def test_lstm_model(self):
        """Test LSTM model implementation | 測試LSTM模型實現"""
        print(f"\n{Colors.BOLD}5. LSTM Model Tests | LSTM模型測試{Colors.END}")
        
        try:
            # Test import | 測試導入
            from models.lstm_model import LSTMModel
            self.log_test("LSTM Model Import", True)
            
            # Test model creation | 測試模型創建  
            model = LSTMModel()
            self.log_test("LSTM Model Creation", True)
            
            # Create time series data | 創建時間序列數據
            sample_data = self.create_sample_data(200)
            
            # Prepare sequence data | 準備序列數據
            prices = sample_data['Close'].values
            sequence_length = 10
            
            # Create sequences | 創建序列
            X, y = [], []
            for i in range(sequence_length, len(prices)):
                X.append(prices[i-sequence_length:i])
                y.append(1 if prices[i] > prices[i-1] else 0)
            
            X = np.array(X).reshape((len(X), sequence_length, 1))
            y = np.array(y)
            
            if len(X) < 50:
                self.log_test("LSTM Data Preparation", False, "Insufficient sequence data")
                return
            
            # Split data | 分割數據
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            self.log_test("LSTM Data Preparation", True,
                         f"Sequences: {X.shape}, Train: {len(X_train)}")
            
            # Test training (with short epochs for testing) | 測試訓練（用短周期進行測試）
            training_history = model.train(
                X_train, y_train, 
                X_val=X_test, y_val=y_test,
                epochs=2,  # Short for testing | 測試用短周期
                batch_size=16,
                verbose=0
            )
            
            if training_history and model.is_trained:
                self.log_test("LSTM Training", True, "Model trained successfully")
            else:
                self.log_test("LSTM Training", False, "Training failed")
                return
            
            # Test prediction | 測試預測
            predictions = model.predict(X_test)
            if len(predictions) == len(X_test):
                self.log_test("LSTM Prediction", True, f"Predicted {len(predictions)} samples")
            else:
                self.log_test("LSTM Prediction", False, "Prediction size mismatch")
                
        except Exception as e:
            self.log_test("LSTM Model", False, f"Error: {str(e)}")
            # Note: TensorFlow might not be available in all environments
            # 注意：TensorFlow可能在所有環境中都不可用
            if "tensorflow" in str(e).lower() or "keras" in str(e).lower():
                self.warnings.append("TensorFlow/Keras required for LSTM - install with: pip install tensorflow")
    
    def test_training_pipeline(self):
        """Test training pipeline | 測試訓練管道"""
        print(f"\n{Colors.BOLD}6. Training Pipeline Tests | 訓練管道測試{Colors.END}")
        
        try:
            # Test import | 測試導入
            from training.model_pipeline import ModelTrainingPipeline
            self.log_test("Training Pipeline Import", True)
            
            # Test pipeline creation | 測試管道創建
            pipeline = ModelTrainingPipeline()
            self.log_test("Training Pipeline Creation", True)
            
            # Create sample data for pipeline | 為管道創建樣本數據
            sample_data = self.create_sample_data(300)
            
            # Add technical indicators using Phase 1 infrastructure | 使用第一階段基礎設施添加技術指標
            from utils.technical_indicators import TechnicalIndicators
            from utils.data_preprocessor import DataPreprocessor
            
            ti = TechnicalIndicators()
            preprocessor = DataPreprocessor()
            
            # Add indicators | 添加指標
            data_with_indicators = ti.add_all_indicators(sample_data)
            processed_data = preprocessor.preprocess_data(data_with_indicators)
            
            if len(processed_data) < 100:
                self.log_test("Pipeline Data Preparation", False, "Insufficient processed data")
                return
            
            # Get features and target | 獲取特徵和目標
            X, y = preprocessor.get_features_and_target(processed_data)
            
            if len(X) == 0 or len(y) == 0:
                self.log_test("Pipeline Data Preparation", False, "No features or targets generated")
                return
                
            self.log_test("Pipeline Data Preparation", True,
                         f"Features: {X.shape}, Target: {len(y)}")
            
            # Test data splitting | 測試數據分割
            splits = pipeline.prepare_data_splits(X, y, test_size=0.2, val_size=0.2)
            if splits and len(splits) == 3:
                self.log_test("Pipeline Data Splitting", True, "Train/Val/Test splits created")
            else:
                self.log_test("Pipeline Data Splitting", False, "Data splitting failed")
            
        except Exception as e:
            self.log_test("Training Pipeline", False, f"Error: {str(e)}")
            traceback.print_exc()
    
    def test_performance_metrics(self):
        """Test performance metrics | 測試性能指標"""
        print(f"\n{Colors.BOLD}7. Performance Metrics Tests | 性能指標測試{Colors.END}")
        
        try:
            # Test import | 測試導入
            from evaluation.performance_metrics import PerformanceMetrics
            self.log_test("Performance Metrics Import", True)
            
            # Create sample predictions and actual values | 創建樣本預測和實際值
            np.random.seed(42)  # For reproducible results | 為可重現結果
            n_samples = 100
            
            # Generate sample data | 生成樣本數據
            y_true = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
            y_pred = np.random.choice([0, 1], size=n_samples, p=[0.55, 0.45])
            y_pred_proba = np.random.random(n_samples)
            
            # Create sample returns for trading metrics | 為交易指標創建樣本回報
            returns = np.random.normal(0.001, 0.02, n_samples)
            
            # Test performance metrics | 測試性能指標
            metrics = PerformanceMetrics()
            
            # Test classification metrics | 測試分類指標
            classification_results = metrics.calculate_classification_metrics(y_true, y_pred, y_pred_proba)
            if classification_results and 'accuracy' in classification_results:
                self.log_test("Classification Metrics", True, 
                             f"Accuracy: {classification_results['accuracy']:.3f}")
            else:
                self.log_test("Classification Metrics", False, "Classification metrics failed")
            
            # Test trading metrics | 測試交易指標
            trading_results = metrics.calculate_trading_metrics(returns, y_pred)
            if trading_results and 'total_return' in trading_results:
                self.log_test("Trading Metrics", True, 
                             f"Total return: {trading_results['total_return']:.3f}")
            else:
                self.log_test("Trading Metrics", False, "Trading metrics failed")
            
            # Test directional accuracy | 測試方向準確度
            direction_accuracy = metrics.calculate_directional_accuracy(y_true, y_pred)
            if direction_accuracy is not None:
                self.log_test("Directional Accuracy", True, f"Direction accuracy: {direction_accuracy:.3f}")
            else:
                self.log_test("Directional Accuracy", False, "Direction accuracy calculation failed")
            
        except Exception as e:
            self.log_test("Performance Metrics", False, f"Error: {str(e)}")
    
    def test_model_management(self):
        """Test model management system | 測試模型管理系統"""
        print(f"\n{Colors.BOLD}8. Model Management Tests | 模型管理測試{Colors.END}")
        
        try:
            # Test import | 測試導入
            from services.model_manager import ModelManager
            self.log_test("Model Manager Import", True)
            
            # Test manager creation | 測試管理器創建
            manager = ModelManager()
            self.log_test("Model Manager Creation", True)
            
            # Test model versioning | 測試模型版本控制
            version_info = manager.get_version_info()
            if version_info and 'version' in version_info:
                self.log_test("Model Versioning", True, f"Version: {version_info['version']}")
            else:
                self.log_test("Model Versioning", False, "Version info not available")
            
            # Test model registry access | 測試模型註冊表訪問
            available_models = manager.list_available_models()
            if isinstance(available_models, list):
                self.log_test("Model Registry Access", True, f"Found {len(available_models)} model types")
            else:
                self.log_test("Model Registry Access", False, "Registry access failed")
            
        except Exception as e:
            self.log_test("Model Management", False, f"Error: {str(e)}")
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI for testing | 計算RSI用於測試"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_report(self):
        """Generate final test report | 生成最終測試報告"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}🤖 AIFX Phase 2 Test Report | AIFX第二階段測試報告{Colors.END}")
        print("="*80)
        
        # Summary statistics | 摘要統計
        pass_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"{Colors.BOLD}Test Summary | 測試摘要:{Colors.END}")
        print(f"  Total Tests: {self.total_tests}")
        print(f"  Passed: {Colors.GREEN}{self.passed_tests}{Colors.END}")
        print(f"  Failed: {Colors.RED}{self.failed_tests}{Colors.END}")
        print(f"  Pass Rate: {Colors.GREEN if pass_rate >= 80 else Colors.YELLOW}{pass_rate:.1f}%{Colors.END}")
        print(f"  Duration: {duration:.2f}s")
        
        if self.warnings:
            print(f"\n{Colors.YELLOW}⚠️ Warnings ({len(self.warnings)}):{Colors.END}")
            for warning in self.warnings[:5]:  # Show first 5 warnings | 顯示前5個警告
                print(f"  • {warning}")
            if len(self.warnings) > 5:
                print(f"  ... and {len(self.warnings) - 5} more")
        
        # Overall assessment | 總體評估
        print(f"\n{Colors.BOLD}Overall Assessment | 總體評估:{Colors.END}")
        
        if pass_rate >= 90:
            status = f"{Colors.GREEN}✅ EXCELLENT{Colors.END}"
            message = "Phase 2 AI models are fully functional | 第二階段AI模型完全正常"
        elif pass_rate >= 80:
            status = f"{Colors.YELLOW}⚠️ GOOD{Colors.END}"
            message = "Phase 2 AI models are mostly functional with minor issues | 第二階段AI模型大部分正常，有輕微問題"
        elif pass_rate >= 60:
            status = f"{Colors.YELLOW}⚠️ ACCEPTABLE{Colors.END}"
            message = "Phase 2 AI models have some issues that should be addressed | 第二階段AI模型有一些應該解決的問題"
        else:
            status = f"{Colors.RED}❌ NEEDS WORK{Colors.END}"
            message = "Phase 2 AI models have significant issues | 第二階段AI模型有重大問題"
        
        print(f"  Status: {status}")
        print(f"  {message}")
        
        # Next steps | 下一步
        print(f"\n{Colors.BOLD}Next Steps | 下一步:{Colors.END}")
        if self.failed_tests > 0:
            print(f"  1. Review and fix {self.failed_tests} failed tests | 檢查並修復{self.failed_tests}個失敗的測試")
        if self.warnings:
            print(f"  2. Address {len(self.warnings)} warnings if needed | 如有需要，處理{len(self.warnings)}個警告")
        
        if pass_rate >= 80:
            print(f"  3. {Colors.GREEN}Ready for Phase 1-2 Integration Testing{Colors.END}")
            print(f"     {Colors.GREEN}準備進行第一階段-第二階段整合測試{Colors.END}")
        else:
            print(f"  3. {Colors.YELLOW}Recommend fixing issues before integration{Colors.END}")
            print(f"     {Colors.YELLOW}建議在整合前修復問題{Colors.END}")
        
        return pass_rate >= 80

def main():
    """Main test execution | 主要測試執行"""
    tester = Phase2Tester()
    
    try:
        # Run all test suites | 運行所有測試套件
        tester.test_ai_dependencies()
        tester.test_base_model_framework()
        tester.test_xgboost_model()
        tester.test_random_forest_model()
        tester.test_lstm_model()
        tester.test_training_pipeline()
        tester.test_performance_metrics()
        tester.test_model_management()
        
        # Generate final report | 生成最終報告
        success = tester.generate_report()
        
        # Return appropriate exit code | 返回適當的退出代碼
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user | 用戶中斷測試{Colors.END}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Fatal error during testing | 測試期間致命錯誤: {str(e)}{Colors.END}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()