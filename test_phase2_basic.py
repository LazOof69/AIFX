"""
AIFX Phase 2 Basic Test Suite | AIFX第二階段基礎測試套件

Simplified Phase 2 testing focused on core functionality without intensive hyperparameter optimization.
簡化的第二階段測試，專注於核心功能，不進行密集超參數優化。

Usage | 使用方法:
    python test_phase2_basic.py
"""

import sys
import os
import traceback
import warnings
from datetime import datetime
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
    END = '\033[0m'

class Phase2BasicTester:
    """Basic Phase 2 testing class | 基礎第二階段測試類"""
    
    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = []
        self.start_time = datetime.now()
        
        print(f"{Colors.BOLD}{Colors.BLUE}🤖 AIFX Phase 2 Basic Test Suite{Colors.END}")
        print(f"{Colors.CYAN}Testing core AI model functionality... | 正在測試核心AI模型功能...{Colors.END}\n")
    
    def log_test(self, test_name: str, passed: bool, message: str = "", warning: str = ""):
        """Log test result | 記錄測試結果"""
        self.total_tests += 1
        
        if passed:
            self.passed_tests += 1
            status = f"{Colors.GREEN}✅ PASS{Colors.END}"
        else:
            self.failed_tests += 1
            status = f"{Colors.RED}❌ FAIL{Colors.END}"
        
        print(f"{status} {test_name}")
        if message:
            print(f"    {Colors.WHITE}{message}{Colors.END}")
        if warning:
            print(f"    {Colors.YELLOW}⚠️ {warning}{Colors.END}")
            self.warnings.append(warning)
    
    def create_sample_data(self, n_samples=200):
        """Create sample data for testing | 創建測試樣本數據"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=n_samples, freq='H')
        
        # Simple price simulation | 簡單價格模擬
        prices = 1.1000 + np.cumsum(np.random.normal(0, 0.001, n_samples))
        
        data = pd.DataFrame({
            'Open': prices + np.random.normal(0, 0.0001, n_samples),
            'High': prices + abs(np.random.normal(0, 0.0003, n_samples)),
            'Low': prices - abs(np.random.normal(0, 0.0003, n_samples)),
            'Close': prices,
            'Volume': np.random.uniform(1000, 5000, n_samples)
        }, index=dates)
        
        return data
    
    def test_ai_imports(self):
        """Test AI model imports | 測試AI模型導入"""
        print(f"{Colors.BOLD}1. AI Model Imports | AI模型導入{Colors.END}")
        
        # Test base model | 測試基礎模型
        try:
            from models.base_model import BaseModel, ModelRegistry
            self.log_test("Base Model Import", True)
        except Exception as e:
            self.log_test("Base Model Import", False, f"Error: {str(e)}")
        
        # Test XGBoost model | 測試XGBoost模型
        try:
            from models.xgboost_model import XGBoostModel
            self.log_test("XGBoost Model Import", True)
        except Exception as e:
            self.log_test("XGBoost Model Import", False, f"Error: {str(e)}")
        
        # Test Random Forest model | 測試隨機森林模型
        try:
            from models.random_forest_model import RandomForestModel
            self.log_test("Random Forest Model Import", True)
        except Exception as e:
            self.log_test("Random Forest Model Import", False, f"Error: {str(e)}")
        
        # Test LSTM model | 測試LSTM模型
        try:
            from models.lstm_model import LSTMModel
            self.log_test("LSTM Model Import", True, "", "TensorFlow required for full LSTM functionality")
        except Exception as e:
            self.log_test("LSTM Model Import", False, f"TensorFlow not available: {str(e)}")
        
        # Test training pipeline | 測試訓練管道
        try:
            from training.model_pipeline import ModelTrainingPipeline
            self.log_test("Training Pipeline Import", True)
        except Exception as e:
            self.log_test("Training Pipeline Import", False, f"Error: {str(e)}")
        
        # Test performance metrics | 測試性能指標
        try:
            from evaluation.performance_metrics import PerformanceMetrics
            self.log_test("Performance Metrics Import", True)
        except Exception as e:
            self.log_test("Performance Metrics Import", False, f"Error: {str(e)}")
        
        # Test model manager | 測試模型管理器
        try:
            from services.model_manager import ModelManager
            self.log_test("Model Manager Import", True)
        except Exception as e:
            self.log_test("Model Manager Import", False, f"Error: {str(e)}")
    
    def test_basic_functionality(self):
        """Test basic model functionality without intensive training | 測試基本模型功能，不進行密集訓練"""
        print(f"\n{Colors.BOLD}2. Basic Model Functionality | 基本模型功能{Colors.END}")
        
        try:
            # Create simple test data | 創建簡單測試數據
            sample_data = self.create_sample_data(100)
            
            # Add basic features | 添加基本特徵
            sample_data['returns'] = sample_data['Close'].pct_change()
            sample_data['sma_5'] = sample_data['Close'].rolling(5).mean()
            sample_data = sample_data.dropna()
            
            # Prepare ML data | 準備機器學習數據
            X = sample_data[['returns', 'sma_5']].fillna(method='ffill')
            y = (sample_data['Close'].shift(-1) > sample_data['Close']).astype(int)[:-1]
            X = X[:-1]
            
            # Test XGBoost basic functionality | 測試XGBoost基本功能
            try:
                from models.xgboost_model import XGBoostModel
                xgb_model = XGBoostModel()
                
                # Train without hyperparameter optimization | 不進行超參數優化的訓練
                training_history = xgb_model.train(X, y, optimize_hyperparameters=False)
                if training_history:
                    self.log_test("XGBoost Basic Training", True, "No hyperparameter optimization")
                else:
                    self.log_test("XGBoost Basic Training", False, "Training failed")
                
                # Test prediction | 測試預測
                if xgb_model.is_trained:
                    predictions = xgb_model.predict(X[-10:])
                    if len(predictions) == 10:
                        self.log_test("XGBoost Prediction", True, "Basic prediction working")
                    else:
                        self.log_test("XGBoost Prediction", False, "Prediction size mismatch")
                
            except Exception as e:
                self.log_test("XGBoost Basic Test", False, f"Error: {str(e)}")
            
            # Test Random Forest basic functionality | 測試隨機森林基本功能
            try:
                from models.random_forest_model import RandomForestModel
                rf_model = RandomForestModel()
                
                # Train without hyperparameter optimization | 不進行超參數優化的訓練
                training_history = rf_model.train(X, y, optimize_hyperparameters=False)
                if training_history:
                    self.log_test("Random Forest Basic Training", True, "No hyperparameter optimization")
                else:
                    self.log_test("Random Forest Basic Training", False, "Training failed")
                
                # Test prediction | 測試預測
                if rf_model.is_trained:
                    predictions = rf_model.predict(X[-10:])
                    if len(predictions) == 10:
                        self.log_test("Random Forest Prediction", True, "Basic prediction working")
                    else:
                        self.log_test("Random Forest Prediction", False, "Prediction size mismatch")
                
            except Exception as e:
                self.log_test("Random Forest Basic Test", False, f"Error: {str(e)}")
            
        except Exception as e:
            self.log_test("Basic Model Functionality", False, f"Error: {str(e)}")
    
    def test_integration_with_phase1(self):
        """Test integration with Phase 1 components | 測試與第一階段組件的集成"""
        print(f"\n{Colors.BOLD}3. Phase 1-2 Integration | 第一階段-第二階段集成{Colors.END}")
        
        try:
            # Test Phase 1 data pipeline integration | 測試第一階段數據管道集成
            from utils.data_loader import DataLoader
            from utils.data_preprocessor import DataPreprocessor
            from utils.technical_indicators import TechnicalIndicators
            
            # Initialize Phase 1 components | 初始化第一階段組件
            data_loader = DataLoader()
            preprocessor = DataPreprocessor()
            indicators = TechnicalIndicators()
            
            self.log_test("Phase 1 Components Init", True, "All Phase 1 components loaded")
            
            # Create and process data using Phase 1 pipeline | 使用第一階段管道創建和處理數據
            sample_data = self.create_sample_data(150)
            
            # Add technical indicators | 添加技術指標
            data_with_indicators = indicators.add_all_indicators(sample_data)
            
            # Preprocess data | 預處理數據
            processed_data = preprocessor.preprocess_data(data_with_indicators, 
                                                        normalize=False, 
                                                        remove_outliers=False)
            
            if len(processed_data) > 50:
                self.log_test("Phase 1 Data Pipeline", True, 
                             f"Processed {len(processed_data)} samples with {processed_data.shape[1]} features")
            else:
                self.log_test("Phase 1 Data Pipeline", False, "Insufficient processed data")
            
            # Test Phase 2 models with Phase 1 data | 使用第一階段數據測試第二階段模型
            try:
                X, y = preprocessor.get_features_and_target(processed_data)
                
                if len(X) > 0 and len(y) > 0:
                    self.log_test("Phase 1-2 Data Integration", True, 
                                 f"Features: {X.shape}, Target: {len(y)}")
                    
                    # Quick test with XGBoost | 使用XGBoost快速測試
                    from models.xgboost_model import XGBoostModel
                    xgb_model = XGBoostModel()
                    
                    # Use subset for quick test | 使用子集進行快速測試
                    X_subset = X[:50] if len(X) > 50 else X
                    y_subset = y[:50] if len(y) > 50 else y
                    
                    training_history = xgb_model.train(X_subset, y_subset, optimize_hyperparameters=False)
                    if training_history:
                        self.log_test("Phase 1-2 Model Integration", True, "XGBoost trained on Phase 1 data")
                    else:
                        self.log_test("Phase 1-2 Model Integration", False, "Model training failed")
                
                else:
                    self.log_test("Phase 1-2 Data Integration", False, "No features or targets extracted")
            
            except Exception as e:
                self.log_test("Phase 1-2 Model Integration", False, f"Model integration error: {str(e)}")
            
        except Exception as e:
            self.log_test("Phase 1-2 Integration", False, f"Integration error: {str(e)}")
    
    def test_performance_metrics_basic(self):
        """Test basic performance metrics | 測試基本性能指標"""
        print(f"\n{Colors.BOLD}4. Performance Metrics Basic | 基本性能指標{Colors.END}")
        
        try:
            from evaluation.performance_metrics import PerformanceMetrics
            
            # Create sample predictions | 創建樣本預測
            np.random.seed(42)
            y_true = np.random.choice([0, 1], size=50, p=[0.6, 0.4])
            y_pred = np.random.choice([0, 1], size=50, p=[0.55, 0.45])
            y_pred_proba = np.random.random(50)
            
            metrics = PerformanceMetrics()
            
            # Test classification metrics | 測試分類指標
            results = metrics.calculate_classification_metrics(y_true, y_pred, y_pred_proba)
            if results and 'accuracy' in results:
                self.log_test("Classification Metrics", True, f"Accuracy: {results['accuracy']:.3f}")
            else:
                self.log_test("Classification Metrics", False, "Metrics calculation failed")
            
            # Test directional accuracy | 測試方向準確度
            direction_acc = metrics.calculate_directional_accuracy(y_true, y_pred)
            if direction_acc is not None:
                self.log_test("Directional Accuracy", True, f"Direction accuracy: {direction_acc:.3f}")
            else:
                self.log_test("Directional Accuracy", False, "Direction accuracy failed")
            
        except Exception as e:
            self.log_test("Performance Metrics Basic", False, f"Error: {str(e)}")
    
    def generate_report(self):
        """Generate final test report | 生成最終測試報告"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}🤖 AIFX Phase 2 Basic Test Report{Colors.END}")
        print("="*60)
        
        pass_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"{Colors.BOLD}Test Summary | 測試摘要:{Colors.END}")
        print(f"  Total Tests: {self.total_tests}")
        print(f"  Passed: {Colors.GREEN}{self.passed_tests}{Colors.END}")
        print(f"  Failed: {Colors.RED}{self.failed_tests}{Colors.END}")
        print(f"  Pass Rate: {Colors.GREEN if pass_rate >= 70 else Colors.YELLOW}{pass_rate:.1f}%{Colors.END}")
        print(f"  Duration: {duration:.2f}s")
        
        if self.warnings:
            print(f"\n{Colors.YELLOW}⚠️ Warnings ({len(self.warnings)}):{Colors.END}")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        # Assessment | 評估
        if pass_rate >= 80:
            status = f"{Colors.GREEN}✅ GOOD{Colors.END}"
            message = "Phase 2 basic functionality is working | 第二階段基本功能正常工作"
        elif pass_rate >= 60:
            status = f"{Colors.YELLOW}⚠️ ACCEPTABLE{Colors.END}"
            message = "Phase 2 has some issues but core functionality works | 第二階段有一些問題但核心功能工作正常"
        else:
            status = f"{Colors.RED}❌ NEEDS WORK{Colors.END}"
            message = "Phase 2 has significant issues | 第二階段有重大問題"
        
        print(f"\n{Colors.BOLD}Status: {status}{Colors.END}")
        print(f"  {message}")
        
        return pass_rate >= 60

def main():
    """Main test execution | 主要測試執行"""
    tester = Phase2BasicTester()
    
    try:
        tester.test_ai_imports()
        tester.test_basic_functionality()
        tester.test_integration_with_phase1()
        tester.test_performance_metrics_basic()
        
        success = tester.generate_report()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n{Colors.RED}Fatal error: {str(e)}{Colors.END}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()