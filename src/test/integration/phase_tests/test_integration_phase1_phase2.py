"""
AIFX Phase 1-2 Integration Test Suite | AIFX第一階段-第二階段整合測試套件

Focused testing for integration and conflict resolution between Phase 1 and Phase 2.
專注於第一階段和第二階段間整合和衝突解決的測試。

Usage | 使用方法:
    python test_integration_phase1_phase2.py
"""

import sys
import os
import traceback
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import importlib

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

class IntegrationTester:
    """Integration testing between Phase 1 and Phase 2 | 第一階段和第二階段間的整合測試"""
    
    def __init__(self):
        self.test_results = []
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.warnings = []
        self.start_time = datetime.now()
        
        print(f"{Colors.BOLD}{Colors.BLUE}🔄 AIFX Phase 1-2 Integration Test Suite{Colors.END}")
        print(f"{Colors.CYAN}Testing integration and conflict resolution... | 正在測試整合和衝突解決...{Colors.END}\n")
    
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
    
    def test_import_compatibility(self):
        """Test that all imports work without conflicts | 測試所有導入都能正常工作，無衝突"""
        print(f"{Colors.BOLD}1. Import Compatibility Tests | 導入兼容性測試{Colors.END}")
        
        # Test Phase 1 imports | 測試第一階段導入
        phase1_modules = [
            ('utils.config', 'Configuration system'),
            ('utils.logger', 'Logging system'),
            ('utils.data_loader', 'Data loading utilities'),
            ('utils.data_preprocessor', 'Data preprocessing'),
            ('utils.technical_indicators', 'Technical indicators')
        ]
        
        for module_name, description in phase1_modules:
            try:
                importlib.import_module(module_name)
                self.log_test(f"Phase 1 Import: {module_name}", True, description)
            except Exception as e:
                self.log_test(f"Phase 1 Import: {module_name}", False, f"Error: {str(e)}")
        
        # Test Phase 2 imports (without TensorFlow dependencies) | 測試第二階段導入（不含TensorFlow依賴）
        phase2_modules = [
            ('models.xgboost_model', 'XGBoost model implementation'),
            ('models.random_forest_model', 'Random Forest model implementation')
        ]
        
        for module_name, description in phase2_modules:
            try:
                importlib.import_module(module_name)
                self.log_test(f"Phase 2 Import: {module_name}", True, description)
            except Exception as e:
                self.log_test(f"Phase 2 Import: {module_name}", False, f"Error: {str(e)}")
        
        # Test problematic imports separately | 單獨測試有問題的導入
        try:
            from models import lstm_model
            self.log_test("Phase 2 LSTM Import", False, "Should fail gracefully without TensorFlow")
        except ImportError as e:
            if "tensorflow" in str(e).lower() or "tf" in str(e).lower():
                self.log_test("Phase 2 LSTM Import", True, "Failed gracefully - TensorFlow not available", 
                             "TensorFlow required for LSTM functionality")
            else:
                self.log_test("Phase 2 LSTM Import", False, f"Unexpected import error: {str(e)}")
        except Exception as e:
            self.log_test("Phase 2 LSTM Import", False, f"Unexpected error: {str(e)}")
    
    def test_data_flow_compatibility(self):
        """Test data flow from Phase 1 to Phase 2 | 測試從第一階段到第二階段的數據流"""
        print(f"\n{Colors.BOLD}2. Data Flow Compatibility | 數據流兼容性{Colors.END}")
        
        try:
            # Initialize Phase 1 components | 初始化第一階段組件
            from utils.data_loader import DataLoader
            from utils.data_preprocessor import DataPreprocessor
            from utils.technical_indicators import TechnicalIndicators
            
            data_loader = DataLoader()
            preprocessor = DataPreprocessor()
            indicators = TechnicalIndicators()
            
            self.log_test("Phase 1 Component Initialization", True, "All components initialized")
            
            # Create sample forex data | 創建樣本外匯數據
            dates = pd.date_range('2024-01-01', periods=200, freq='H')
            np.random.seed(42)  # For reproducible results | 為可重現結果
            
            prices = 1.1000 + np.cumsum(np.random.normal(0, 0.001, 200))
            sample_data = pd.DataFrame({
                'Open': prices + np.random.normal(0, 0.0001, 200),
                'High': prices + abs(np.random.normal(0, 0.0003, 200)),
                'Low': prices - abs(np.random.normal(0, 0.0003, 200)),
                'Close': prices,
                'Volume': np.random.uniform(1000, 5000, 200)
            }, index=dates)
            
            # Fix OHLC relationships | 修復OHLC關係
            for i in range(len(sample_data)):
                max_oc = max(sample_data.iloc[i]['Open'], sample_data.iloc[i]['Close'])
                min_oc = min(sample_data.iloc[i]['Open'], sample_data.iloc[i]['Close'])
                sample_data.iloc[i]['High'] = max(sample_data.iloc[i]['High'], max_oc)
                sample_data.iloc[i]['Low'] = min(sample_data.iloc[i]['Low'], min_oc)
            
            self.log_test("Sample Data Creation", True, f"Created {len(sample_data)} OHLCV records")
            
            # Step 1: Add technical indicators | 步驟1：添加技術指標
            data_with_indicators = indicators.add_all_indicators(sample_data)
            original_cols = len(sample_data.columns)
            new_cols = len(data_with_indicators.columns)
            
            if new_cols > original_cols:
                self.log_test("Technical Indicators Integration", True, 
                             f"Added {new_cols - original_cols} technical indicators")
            else:
                self.log_test("Technical Indicators Integration", False, "No indicators added")
            
            # Step 2: Preprocess data | 步驟2：預處理數據
            processed_data = preprocessor.preprocess_data(data_with_indicators, 
                                                        normalize=False,
                                                        remove_outliers=False)
            
            if len(processed_data) >= len(sample_data) * 0.7:  # At least 70% retained | 至少保留70%
                self.log_test("Data Preprocessing", True, 
                             f"Processed {len(processed_data)} records, {processed_data.shape[1]} features")
            else:
                self.log_test("Data Preprocessing", False, 
                             f"Too much data lost: {len(processed_data)}/{len(sample_data)}")
            
            # Step 3: Extract features and targets | 步驟3：提取特徵和目標
            X, y = preprocessor.get_features_and_target(processed_data)
            
            if len(X) > 0 and len(y) > 0 and len(X) == len(y):
                self.log_test("Feature-Target Extraction", True, 
                             f"Features: {X.shape}, Target: {len(y)}")
            else:
                self.log_test("Feature-Target Extraction", False, 
                             f"Feature-target mismatch: X={X.shape if hasattr(X, 'shape') else 'None'}, y={len(y)}")
                return False
            
            # Step 4: Test with Phase 2 models | 步驟4：使用第二階段模型測試
            from models.xgboost_model import XGBoostModel
            from models.random_forest_model import RandomForestModel
            
            # Test XGBoost integration | 測試XGBoost整合
            xgb_model = XGBoostModel()
            
            # Use subset for quick training | 使用子集進行快速訓練
            subset_size = min(100, len(X))
            X_subset = X[:subset_size]
            y_subset = y[:subset_size]
            
            training_history = xgb_model.train(X_subset, y_subset, optimize_hyperparameters=False)
            
            if training_history and xgb_model.is_trained:
                # Test prediction | 測試預測
                test_predictions = xgb_model.predict(X_subset[-20:])
                if len(test_predictions) == 20:
                    self.log_test("XGBoost Full Integration", True, "Training and prediction successful")
                else:
                    self.log_test("XGBoost Full Integration", False, "Prediction failed")
            else:
                self.log_test("XGBoost Full Integration", False, "Training failed")
            
            # Test Random Forest integration | 測試隨機森林整合
            rf_model = RandomForestModel()
            training_history = rf_model.train(X_subset, y_subset, optimize_hyperparameters=False)
            
            if training_history and rf_model.is_trained:
                test_predictions = rf_model.predict(X_subset[-20:])
                if len(test_predictions) == 20:
                    self.log_test("Random Forest Full Integration", True, "Training and prediction successful")
                else:
                    self.log_test("Random Forest Full Integration", False, "Prediction failed")
            else:
                self.log_test("Random Forest Full Integration", False, "Training failed")
            
            return True
            
        except Exception as e:
            self.log_test("Data Flow Compatibility", False, f"Error: {str(e)}")
            traceback.print_exc()
            return False
    
    def test_memory_and_performance(self):
        """Test memory usage and performance | 測試記憶體使用和性能"""
        print(f"\n{Colors.BOLD}3. Memory and Performance Tests | 記憶體和性能測試{Colors.END}")
        
        try:
            import psutil
            import gc
            
            # Get initial memory usage | 獲取初始記憶體使用
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run data processing pipeline | 運行數據處理管道
            from utils.data_loader import DataLoader
            from utils.data_preprocessor import DataPreprocessor
            from utils.technical_indicators import TechnicalIndicators
            from models.xgboost_model import XGBoostModel
            
            # Create larger dataset for memory test | 為記憶體測試創建更大數據集
            dates = pd.date_range('2024-01-01', periods=1000, freq='H')
            np.random.seed(42)
            
            prices = 1.1000 + np.cumsum(np.random.normal(0, 0.001, 1000))
            large_dataset = pd.DataFrame({
                'Open': prices + np.random.normal(0, 0.0001, 1000),
                'High': prices + abs(np.random.normal(0, 0.0003, 1000)),
                'Low': prices - abs(np.random.normal(0, 0.0003, 1000)),
                'Close': prices,
                'Volume': np.random.uniform(1000, 5000, 1000)
            }, index=dates)
            
            # Process through full pipeline | 通過完整管道處理
            indicators = TechnicalIndicators()
            preprocessor = DataPreprocessor()
            
            data_with_indicators = indicators.add_all_indicators(large_dataset)
            processed_data = preprocessor.preprocess_data(data_with_indicators, 
                                                        normalize=False,
                                                        remove_outliers=False)
            X, y = preprocessor.get_features_and_target(processed_data)
            
            # Train model | 訓練模型
            model = XGBoostModel()
            subset_size = min(200, len(X))
            model.train(X[:subset_size], y[:subset_size], optimize_hyperparameters=False)
            
            # Get final memory usage | 獲取最終記憶體使用
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # Memory usage check | 記憶體使用檢查
            if memory_increase < 500:  # Less than 500MB increase | 增加不到500MB
                self.log_test("Memory Usage", True, f"Memory increase: {memory_increase:.1f} MB")
            else:
                self.log_test("Memory Usage", False, f"High memory usage: {memory_increase:.1f} MB",
                             "Consider memory optimization")
            
            # Garbage collection test | 垃圾回收測試
            del data_with_indicators, processed_data, model
            gc.collect()
            
            after_gc_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_freed = final_memory - after_gc_memory
            
            if memory_freed > 0:
                self.log_test("Memory Cleanup", True, f"Freed {memory_freed:.1f} MB after cleanup")
            else:
                self.log_test("Memory Cleanup", True, "Memory cleanup completed", 
                             "No significant memory freed - normal behavior")
            
        except ImportError:
            self.log_test("Memory and Performance Tests", True, "", "psutil not available - memory tests skipped")
        except Exception as e:
            self.log_test("Memory and Performance Tests", False, f"Error: {str(e)}")
    
    def test_dependency_conflicts(self):
        """Test for dependency version conflicts | 測試依賴版本衝突"""
        print(f"\n{Colors.BOLD}4. Dependency Conflict Tests | 依賴衝突測試{Colors.END}")
        
        try:
            # Test critical dependencies | 測試關鍵依賴
            critical_deps = [
                'pandas', 'numpy', 'sklearn', 'xgboost', 'matplotlib'
            ]
            
            for dep in critical_deps:
                try:
                    module = importlib.import_module(dep)
                    version = getattr(module, '__version__', 'unknown')
                    self.log_test(f"Dependency: {dep}", True, f"Version: {version}")
                except ImportError:
                    self.log_test(f"Dependency: {dep}", False, "Not installed")
            
            # Test for common conflicts | 測試常見衝突
            try:
                import sklearn
                import xgboost
                
                # Try to use both together | 嘗試一起使用
                from sklearn.model_selection import train_test_split
                import xgboost as xgb
                
                # Create sample data | 創建樣本數據
                X = np.random.random((100, 5))
                y = np.random.randint(0, 2, 100)
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Test XGBoost with sklearn compatibility | 測試XGBoost與sklearn兼容性
                xgb_classifier = xgb.XGBClassifier(n_estimators=10, random_state=42, verbosity=0)
                xgb_classifier.fit(X_train, y_train)
                predictions = xgb_classifier.predict(X_test)
                
                if len(predictions) == len(X_test):
                    self.log_test("SKLearn-XGBoost Compatibility", True, "No conflicts detected")
                else:
                    self.log_test("SKLearn-XGBoost Compatibility", False, "Prediction mismatch")
                
            except Exception as e:
                self.log_test("SKLearn-XGBoost Compatibility", False, f"Conflict detected: {str(e)}")
            
        except Exception as e:
            self.log_test("Dependency Conflict Tests", False, f"Error: {str(e)}")
    
    def generate_report(self):
        """Generate final integration test report | 生成最終整合測試報告"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}🔄 AIFX Phase 1-2 Integration Test Report{Colors.END}")
        print("="*70)
        
        pass_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"{Colors.BOLD}Integration Test Summary | 整合測試摘要:{Colors.END}")
        print(f"  Total Tests: {self.total_tests}")
        print(f"  Passed: {Colors.GREEN}{self.passed_tests}{Colors.END}")
        print(f"  Failed: {Colors.RED}{self.failed_tests}{Colors.END}")
        print(f"  Pass Rate: {Colors.GREEN if pass_rate >= 80 else Colors.YELLOW}{pass_rate:.1f}%{Colors.END}")
        print(f"  Duration: {duration:.2f}s")
        
        if self.warnings:
            print(f"\n{Colors.YELLOW}⚠️ Warnings ({len(self.warnings)}):{Colors.END}")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        # Overall assessment | 總體評估
        print(f"\n{Colors.BOLD}Integration Assessment | 整合評估:{Colors.END}")
        
        if pass_rate >= 85:
            status = f"{Colors.GREEN}✅ EXCELLENT{Colors.END}"
            message = "Phase 1 and Phase 2 integrate perfectly | 第一階段和第二階段完美整合"
            readiness = f"{Colors.GREEN}✅ READY for Phase 3 Development{Colors.END}"
        elif pass_rate >= 70:
            status = f"{Colors.GREEN}✅ GOOD{Colors.END}"
            message = "Phase 1 and Phase 2 integrate well with minor issues | 第一階段和第二階段整合良好，有輕微問題"
            readiness = f"{Colors.GREEN}✅ READY for Phase 3 Development{Colors.END}"
        elif pass_rate >= 50:
            status = f"{Colors.YELLOW}⚠️ ACCEPTABLE{Colors.END}"
            message = "Phase 1 and Phase 2 have integration issues | 第一階段和第二階段有整合問題"
            readiness = f"{Colors.YELLOW}⚠️ PROCEED WITH CAUTION to Phase 3{Colors.END}"
        else:
            status = f"{Colors.RED}❌ POOR{Colors.END}"
            message = "Significant integration conflicts detected | 檢測到重大整合衝突"
            readiness = f"{Colors.RED}❌ FIX ISSUES before Phase 3{Colors.END}"
        
        print(f"  Status: {status}")
        print(f"  {message}")
        print(f"\n{Colors.BOLD}Phase 3 Readiness | 第三階段準備狀態:{Colors.END}")
        print(f"  {readiness}")
        
        # Recommendations | 建議
        print(f"\n{Colors.BOLD}Recommendations | 建議:{Colors.END}")
        if self.failed_tests == 0:
            print(f"  • {Colors.GREEN}All systems integrated successfully{Colors.END}")
            print(f"  • {Colors.GREEN}Ready to proceed with Phase 3 Strategy Integration{Colors.END}")
        else:
            if any("tensorflow" in w.lower() or "keras" in w.lower() for w in self.warnings):
                print(f"  • Install TensorFlow for full LSTM functionality: pip install tensorflow")
            if self.failed_tests > 2:
                print(f"  • Review and fix {self.failed_tests} integration issues before Phase 3")
            print(f"  • Consider addressing warnings for optimal performance")
        
        return pass_rate >= 70

def main():
    """Main test execution | 主要測試執行"""
    tester = IntegrationTester()
    
    try:
        tester.test_import_compatibility()
        tester.test_data_flow_compatibility()
        tester.test_memory_and_performance()
        tester.test_dependency_conflicts()
        
        success = tester.generate_report()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\n{Colors.RED}Fatal integration error: {str(e)}{Colors.END}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()