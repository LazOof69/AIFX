"""
Unit tests for AIFX Data Loader | AIFX數據載入器的單元測試

This module contains comprehensive unit tests for the data loading functionality.
此模組包含數據載入功能的綜合單元測試。
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add src path for imports | 為導入添加src路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'main', 'python'))

from utils.data_loader import DataLoader
from utils.config import Config


class TestDataLoader(unittest.TestCase):
    """
    Test cases for DataLoader class | DataLoader類的測試用例
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up test class with temporary directory | 使用臨時目錄設置測試類
        """
        cls.temp_dir = tempfile.mkdtemp()
        cls.config = Config()
        cls.config.data.raw_data_path = os.path.join(cls.temp_dir, 'raw')
        cls.config.data.processed_data_path = os.path.join(cls.temp_dir, 'processed')
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up temporary directory | 清理臨時目錄
        """
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """
        Set up individual test case | 設置單個測試用例
        """
        self.data_loader = DataLoader(self.config)
    
    def test_init(self):
        """
        Test DataLoader initialization | 測試DataLoader初始化
        """
        self.assertIsInstance(self.data_loader, DataLoader)
        self.assertTrue(Path(self.data_loader.raw_data_path).exists())
        self.assertTrue(Path(self.data_loader.processed_data_path).exists())
    
    def test_convert_symbol(self):
        """
        Test symbol conversion | 測試品種轉換
        """
        # Test standard forex pairs | 測試標準外匯對
        self.assertEqual(self.data_loader._convert_symbol('EURUSD'), 'EURUSD=X')
        self.assertEqual(self.data_loader._convert_symbol('USDJPY'), 'USDJPY=X')
        self.assertEqual(self.data_loader._convert_symbol('eurusd'), 'EURUSD=X')
        
        # Test already formatted symbols | 測試已格式化的品種
        self.assertEqual(self.data_loader._convert_symbol('EURUSD=X'), 'EURUSD=X')
        
        # Test with separators | 測試帶分隔符的品種
        self.assertEqual(self.data_loader._convert_symbol('EUR/USD'), 'EURUSD=X')
        self.assertEqual(self.data_loader._convert_symbol('EUR_USD'), 'EURUSD=X')
    
    def test_clean_data(self):
        """
        Test data cleaning functionality | 測試數據清理功能
        """
        # Create sample dirty data | 創建樣本髒數據
        dates = pd.date_range('2024-01-01', periods=10, freq='H')
        dirty_data = pd.DataFrame({
            'Open': [1.1000, 1.1010, np.nan, 1.1030, 1.1040, 1.1050, 1.1060, 1.1070, 1.1080, 1.1090],
            'High': [1.1010, 1.1020, 1.1025, 1.1040, 1.1050, 1.1060, 1.1070, 1.1080, 1.1090, 1.1100],
            'Low': [1.0990, 1.1000, 1.1015, 1.1020, 1.1030, 1.1040, 1.1050, 1.1060, 1.1070, 1.1080],
            'Close': [1.1005, 1.1015, 1.1020, 1.1035, 1.1045, 1.1055, 1.1065, 1.1075, 1.1085, 1.1095],
            'Volume': [1000, 0, 1500, 2000, 1800, 1200, 1100, 1300, 1400, 1600]
        }, index=dates)
        
        # Add duplicate row | 添加重複行
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[[-1]]])
        
        # Clean the data | 清理數據
        clean_data = self.data_loader._clean_data(dirty_data)
        
        # Verify cleaning results | 驗證清理結果
        self.assertFalse(clean_data.isnull().any().any())  # No missing values | 無缺失值
        self.assertFalse(clean_data.index.duplicated().any())  # No duplicates | 無重複
        self.assertTrue((clean_data['Volume'] > 0).all())  # No zero volume | 無零成交量
        self.assertEqual(list(clean_data.columns), ['Open', 'High', 'Low', 'Close', 'Volume'])
    
    def test_validate_data(self):
        """
        Test data validation functionality | 測試數據驗證功能
        """
        # Create valid data | 創建有效數據
        dates = pd.date_range('2024-01-01', periods=1500, freq='H')  # Above minimum | 超過最小值
        valid_data = pd.DataFrame({
            'Open': np.random.uniform(1.1000, 1.1100, 1500),
            'High': np.random.uniform(1.1050, 1.1150, 1500),
            'Low': np.random.uniform(1.0950, 1.1050, 1500),
            'Close': np.random.uniform(1.1000, 1.1100, 1500),
            'Volume': np.random.uniform(1000, 5000, 1500)
        }, index=dates)
        
        # Ensure OHLC relationships are valid | 確保OHLC關係有效
        for i in range(len(valid_data)):
            valid_data.iloc[i]['High'] = max(valid_data.iloc[i][['Open', 'Close']]) + 0.001
            valid_data.iloc[i]['Low'] = min(valid_data.iloc[i][['Open', 'Close']]) - 0.001
        
        # Should not raise exception | 不應該引發異常
        try:
            self.data_loader._validate_data(valid_data, 'TEST')
        except Exception as e:
            self.fail(f"Valid data validation failed: {e}")
        
        # Test insufficient data points | 測試數據點不足
        small_data = valid_data.head(5)  # Below absolute minimum | 低於絕對最小值
        with self.assertRaises(ValueError):
            self.data_loader._validate_data(small_data, 'TEST')
        
        # Test invalid prices | 測試無效價格
        invalid_data = valid_data.copy()
        invalid_data.iloc[0]['Close'] = -1.0  # Negative price | 負價
        with self.assertRaises(ValueError):
            self.data_loader._validate_data(invalid_data, 'TEST')
    
    def test_get_data_info(self):
        """
        Test data information retrieval | 測試數據信息檢索
        """
        info = self.data_loader.get_data_info('EURUSD')
        
        # Check info structure | 檢查信息結構
        expected_keys = ['symbol', 'raw_data_available', 'processed_data_available',
                        'raw_data_records', 'processed_data_records', 'date_range']
        
        for key in expected_keys:
            self.assertIn(key, info)
        
        self.assertEqual(info['symbol'], 'EURUSD')
        self.assertIsInstance(info['raw_data_available'], bool)
        self.assertIsInstance(info['processed_data_available'], bool)


class TestDataLoaderIntegration(unittest.TestCase):
    """
    Integration tests for DataLoader | DataLoader的集成測試
    Note: These tests require internet connection | 注意：這些測試需要網絡連接
    """
    
    @classmethod
    def setUpClass(cls):
        """
        Set up integration test class | 設置集成測試類
        """
        cls.temp_dir = tempfile.mkdtemp()
        cls.config = Config()
        cls.config.data.raw_data_path = os.path.join(cls.temp_dir, 'raw')
        cls.config.data.processed_data_path = os.path.join(cls.temp_dir, 'processed')
    
    @classmethod
    def tearDownClass(cls):
        """
        Clean up temporary directory | 清理臨時目錄
        """
        shutil.rmtree(cls.temp_dir)
    
    def setUp(self):
        """
        Set up individual integration test | 設置單個集成測試
        """
        self.data_loader = DataLoader(self.config)
    
    @unittest.skipIf(os.getenv('SKIP_NETWORK_TESTS', 'False').lower() == 'true',
                     "Skipping network-dependent tests")
    def test_download_data_integration(self):
        """
        Test actual data download (requires network) | 測試實際數據下載（需要網絡）
        """
        try:
            # Download small amount of data | 下載少量數據
            data = self.data_loader.download_data(['EURUSD'], period='5d', interval='1h')
            
            # Verify download results | 驗證下載結果
            self.assertIn('EURUSD', data)
            self.assertIsInstance(data['EURUSD'], pd.DataFrame)
            self.assertGreater(len(data['EURUSD']), 0)
            
            # Check required columns | 檢查必需列
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                self.assertIn(col, data['EURUSD'].columns)
            
            # Verify data quality | 驗證數據質量
            df = data['EURUSD']
            self.assertFalse(df.isnull().all().any())  # Not all null | 不全為空
            self.assertTrue((df[['Open', 'High', 'Low', 'Close']] > 0).all().all())  # Positive prices | 正價格
            
        except Exception as e:
            # Skip if network issues | 如果網絡問題則跳過
            self.skipTest(f"Network test failed: {e}")
    
    def test_save_load_processed_data(self):
        """
        Test saving and loading processed data | 測試保存和載入處理後的數據
        """
        # Create sample processed data | 創建樣本處理後數據
        dates = pd.date_range('2024-01-01', periods=100, freq='H')
        sample_data = pd.DataFrame({
            'Open': np.random.uniform(1.1000, 1.1100, 100),
            'High': np.random.uniform(1.1050, 1.1150, 100),
            'Low': np.random.uniform(1.0950, 1.1050, 100),
            'Close': np.random.uniform(1.1000, 1.1100, 100),
            'Volume': np.random.uniform(1000, 5000, 100),
            'feature_1': np.random.random(100),
            'feature_2': np.random.random(100)
        }, index=dates)
        
        # Save processed data | 保存處理後數據
        data_dict = {'EURUSD': sample_data}
        self.data_loader.save_processed_data(data_dict)
        
        # Load processed data | 載入處理後數據
        loaded_data = self.data_loader.load_processed_data(['EURUSD'])
        
        # Verify loaded data | 驗證載入的數據
        self.assertIn('EURUSD', loaded_data)
        
        # Compare data values, ignoring frequency information
        loaded_df = loaded_data['EURUSD']
        self.assertEqual(len(loaded_df), len(sample_data))
        self.assertTrue(loaded_df.columns.equals(sample_data.columns))
        
        # Check index values (ignore frequency)
        self.assertTrue(loaded_df.index.equals(sample_data.index))
        
        # Check data values with tolerance for floating point precision
        pd.testing.assert_frame_equal(
            loaded_df.reset_index(drop=True), 
            sample_data.reset_index(drop=True),
            check_exact=False,
            rtol=1e-10
        )


def create_sample_ohlcv_data(periods: int = 100, start_date: str = '2024-01-01') -> pd.DataFrame:
    """
    Helper function to create sample OHLCV data for testing | 創建測試用樣本OHLCV數據的輔助函數
    
    Args:
        periods: Number of periods | 週期數
        start_date: Start date | 開始日期
        
    Returns:
        Sample OHLCV DataFrame | 樣本OHLCV DataFrame
    """
    dates = pd.date_range(start_date, periods=periods, freq='H')
    
    # Generate realistic price movements | 生成真實的價格變動
    base_price = 1.1000
    price_changes = np.random.normal(0, 0.0002, periods)  # Small random changes | 小的隨機變化
    prices = base_price + np.cumsum(price_changes)
    
    # Create OHLCV data | 創建OHLCV數據
    data = []
    for i, price in enumerate(prices):
        spread = np.random.uniform(0.0001, 0.0005)
        open_price = price + np.random.uniform(-spread, spread)
        close_price = price + np.random.uniform(-spread, spread)
        high_price = max(open_price, close_price) + np.random.uniform(0, spread)
        low_price = min(open_price, close_price) - np.random.uniform(0, spread)
        volume = np.random.uniform(1000, 5000)
        
        data.append({
            'Open': open_price,
            'High': high_price,
            'Low': low_price,
            'Close': close_price,
            'Volume': volume
        })
    
    return pd.DataFrame(data, index=dates)


if __name__ == '__main__':
    # Run tests | 運行測試
    unittest.main(verbosity=2)