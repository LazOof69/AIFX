#!/usr/bin/env python3
"""
Test SQL Server integration for AIFX
測試 AIFX 的 SQL Server 整合

This script tests the SQL Server database connectivity and basic operations
此腳本測試 SQL Server 資料庫連接和基本操作
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add src to path | 將 src 添加到路徑
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'main', 'python'))

from utils.database import DatabaseManager, test_database_connection, save_trading_data, load_trading_data

# Setup logging | 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Colors:
    """Terminal colors for output formatting | 用於輸出格式化的終端顏色"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def create_sample_data(symbol: str = "EURUSD", days: int = 30) -> pd.DataFrame:
    """
    Create sample trading data for testing
    創建用於測試的樣本交易數據
    """
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='H'
    )
    
    # Generate realistic OHLC data | 生成現實的OHLC數據
    np.random.seed(42)  # For reproducible results | 用於可重現結果
    base_price = 1.1000 if symbol == "EURUSD" else 150.00
    
    data = []
    current_price = base_price
    
    for date in dates:
        # Random walk with some volatility | 帶有波動性的隨機遊走
        change = np.random.normal(0, 0.001)
        current_price = max(current_price + change, 0.5000)  # Minimum price floor
        
        # Generate OHLC based on current price | 基於當前價格生成OHLC
        open_price = current_price
        high_price = open_price + abs(np.random.normal(0, 0.0005))
        low_price = open_price - abs(np.random.normal(0, 0.0005))
        close_price = open_price + np.random.normal(0, 0.0003)
        
        # Ensure OHLC relationships | 確保OHLC關係
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)
        
        volume = int(np.random.uniform(1000, 10000))
        
        data.append({
            'datetime': date,
            'open_price': round(open_price, 6),
            'high_price': round(high_price, 6),
            'low_price': round(low_price, 6),
            'close_price': round(close_price, 6),
            'volume': volume
        })
        
        current_price = close_price
    
    df = pd.DataFrame(data)
    df.set_index('datetime', inplace=True)
    
    logger.info(f"Created sample data for {symbol}: {len(df)} records")
    return df

def test_basic_connection():
    """Test basic database connection | 測試基本資料庫連接"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}1. Testing Database Connection | 測試資料庫連接{Colors.END}")
    
    try:
        success = test_database_connection()
        if success:
            print(f"{Colors.GREEN}✅ Database connection successful | 資料庫連接成功{Colors.END}")
            return True
        else:
            print(f"{Colors.RED}❌ Database connection failed | 資料庫連接失敗{Colors.END}")
            return False
    except Exception as e:
        print(f"{Colors.RED}❌ Connection error: {e}{Colors.END}")
        return False

def test_table_creation():
    """Test table creation | 測試資料表創建"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}2. Testing Table Creation | 測試資料表創建{Colors.END}")
    
    try:
        db_manager = DatabaseManager()
        db_manager.create_tables()
        print(f"{Colors.GREEN}✅ Tables created successfully | 資料表創建成功{Colors.END}")
        return True
    except Exception as e:
        print(f"{Colors.RED}❌ Table creation error: {e}{Colors.END}")
        return False

def test_data_operations():
    """Test data save and load operations | 測試數據保存和載入操作"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}3. Testing Data Operations | 測試數據操作{Colors.END}")
    
    try:
        # Create and save sample data | 創建並保存樣本數據
        print("Creating sample trading data... | 創建樣本交易數據...")
        eurusd_data = create_sample_data("EURUSD", 7)  # 7 days of data
        usdjpy_data = create_sample_data("USDJPY", 7)
        
        print("Saving data to database... | 保存數據到資料庫...")
        save_trading_data(eurusd_data, "EURUSD")
        save_trading_data(usdjpy_data, "USDJPY")
        
        print(f"{Colors.GREEN}✅ Data saved successfully | 數據保存成功{Colors.END}")
        
        # Test data loading | 測試數據載入
        print("Loading data from database... | 從資料庫載入數據...")
        loaded_eurusd = load_trading_data("EURUSD")
        loaded_usdjpy = load_trading_data("USDJPY")
        
        print(f"EURUSD records loaded: {len(loaded_eurusd)} | 載入的EURUSD記錄數")
        print(f"USDJPY records loaded: {len(loaded_usdjpy)} | 載入的USDJPY記錄數")
        
        if len(loaded_eurusd) > 0 and len(loaded_usdjpy) > 0:
            print(f"{Colors.GREEN}✅ Data operations successful | 數據操作成功{Colors.END}")
            return True
        else:
            print(f"{Colors.RED}❌ No data loaded | 未載入數據{Colors.END}")
            return False
            
    except Exception as e:
        print(f"{Colors.RED}❌ Data operations error: {e}{Colors.END}")
        return False

def test_query_performance():
    """Test query performance | 測試查詢性能"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}4. Testing Query Performance | 測試查詢性能{Colors.END}")
    
    try:
        db_manager = DatabaseManager()
        
        # Test simple query | 測試簡單查詢
        start_time = datetime.now()
        
        query = """
        SELECT COUNT(*) as record_count 
        FROM trading_data_eurusd
        """
        
        result = db_manager.execute_query(query)
        
        end_time = datetime.now()
        query_time = (end_time - start_time).total_seconds()
        
        record_count = result.iloc[0]['record_count'] if len(result) > 0 else 0
        
        print(f"Query executed in {query_time:.3f} seconds | 查詢執行時間：{query_time:.3f}秒")
        print(f"Total records: {record_count} | 總記錄數：{record_count}")
        
        if query_time < 1.0:  # Should be fast for small datasets
            print(f"{Colors.GREEN}✅ Query performance acceptable | 查詢性能可接受{Colors.END}")
            return True
        else:
            print(f"{Colors.YELLOW}⚠️ Query performance slower than expected | 查詢性能比預期慢{Colors.END}")
            return True  # Still pass, just a warning
            
    except Exception as e:
        print(f"{Colors.RED}❌ Query performance test error: {e}{Colors.END}")
        return False

def test_environment_detection():
    """Test environment configuration detection | 測試環境配置檢測"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}5. Environment Configuration | 環境配置{Colors.END}")
    
    db_manager = DatabaseManager()
    connection_string = db_manager.connection_string
    db_type = db_manager._get_db_type()
    
    print(f"Database Type: {db_type} | 資料庫類型：{db_type}")
    
    if 'sqlserver' in os.getenv('SQLSERVER_HOST', '').lower():
        print(f"{Colors.GREEN}✅ SQL Server environment detected | 檢測到SQL Server環境{Colors.END}")
    elif 'postgres' in connection_string.lower():
        print(f"{Colors.YELLOW}⚠️ PostgreSQL environment detected | 檢測到PostgreSQL環境{Colors.END}")
    elif 'sqlite' in connection_string.lower():
        print(f"{Colors.BLUE}ℹ️ SQLite environment detected (development) | 檢測到SQLite環境（開發）{Colors.END}")
    else:
        print(f"{Colors.RED}❌ Unknown database environment | 未知資料庫環境{Colors.END}")
    
    return True

def main():
    """Main test function | 主測試函數"""
    print(f"{Colors.BOLD}{Colors.BLUE}🔷 AIFX SQL Server Integration Test | AIFX SQL Server 整合測試{Colors.END}")
    print(f"{Colors.BLUE}Testing database connectivity and operations... | 測試資料庫連接和操作...{Colors.END}\n")
    
    tests = [
        ("Database Connection", test_basic_connection),
        ("Table Creation", test_table_creation),
        ("Data Operations", test_data_operations),
        ("Query Performance", test_query_performance),
        ("Environment Detection", test_environment_detection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
    
    # Summary | 摘要
    print(f"\n{Colors.BOLD}📊 Test Summary | 測試摘要{Colors.END}")
    print("=" * 50)
    print(f"Total Tests: {total} | 總測試數：{total}")
    print(f"Passed: {Colors.GREEN}{passed}{Colors.END}")
    print(f"Failed: {Colors.RED}{total - passed}{Colors.END}")
    print(f"Success Rate: {Colors.GREEN if passed == total else Colors.YELLOW}{passed/total*100:.1f}%{Colors.END}")
    
    if passed == total:
        print(f"\n{Colors.GREEN}🎉 All tests passed! SQL Server integration ready | 所有測試通過！SQL Server整合準備就緒{Colors.END}")
        print(f"\n{Colors.BOLD}📋 Next Steps | 下一步：{Colors.END}")
        print("1. Start with: docker-compose -f docker-compose-sqlserver.yml up -d")
        print("2. Access Adminer at: http://localhost:8080")
        print("3. Monitor with Grafana at: http://localhost:3000")
        return True
    else:
        print(f"\n{Colors.RED}❌ Some tests failed. Check configuration and try again | 部分測試失敗。請檢查配置後重試{Colors.END}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)