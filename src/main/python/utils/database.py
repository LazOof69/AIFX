"""
Database utilities for AIFX trading system
AIFX交易系統資料庫工具

Supports multiple database backends:
支援多種資料庫後端：
- SQL Server Express (free)
- PostgreSQL
- SQLite (development)
"""

import os
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

# Base class for SQLAlchemy models | SQLAlchemy模型基類
Base = declarative_base()

# Logger setup | 日誌設置
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Database connection and session management
    資料庫連接與會話管理
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize database manager | 初始化資料庫管理器
        
        Args:
            connection_string: Database connection string | 資料庫連接字串
        """
        self.connection_string = connection_string or self._get_connection_string()
        self.engine = None
        self.SessionLocal = None
        self._initialize_engine()
    
    def _get_connection_string(self) -> str:
        """
        Get database connection string from environment
        從環境變數取得資料庫連接字串
        """
        # Check for different database types | 檢查不同的資料庫類型
        
        # SQL Server (preferred) | SQL Server（首選）
        if os.getenv('SQLSERVER_HOST'):
            server = os.getenv('SQLSERVER_HOST', 'localhost')
            database = os.getenv('SQLSERVER_DATABASE', 'aifx')
            username = os.getenv('SQLSERVER_USERNAME', 'sa')
            password = os.getenv('SQLSERVER_PASSWORD', 'YourPassword123!')
            driver = os.getenv('SQLSERVER_DRIVER', 'ODBC Driver 17 for SQL Server')
            
            return f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}"
        
        # PostgreSQL fallback | PostgreSQL備用
        elif os.getenv('POSTGRES_HOST') or os.getenv('DATABASE_URL'):
            if os.getenv('DATABASE_URL'):
                return os.getenv('DATABASE_URL')
            else:
                host = os.getenv('POSTGRES_HOST', 'localhost')
                port = os.getenv('POSTGRES_PORT', '5432')
                database = os.getenv('POSTGRES_DB', 'aifx')
                username = os.getenv('POSTGRES_USER', 'aifx')
                password = os.getenv('POSTGRES_PASSWORD', 'password')
                
                return f"postgresql://{username}:{password}@{host}:{port}/{database}"
        
        # SQLite for development | SQLite用於開發
        else:
            db_path = os.path.join(os.getcwd(), 'data', 'aifx.db')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            return f"sqlite:///{db_path}"
    
    def _initialize_engine(self):
        """Initialize SQLAlchemy engine | 初始化SQLAlchemy引擎"""
        try:
            # Engine configuration based on database type | 根據資料庫類型配置引擎
            if 'mssql' in self.connection_string:
                # SQL Server specific settings | SQL Server特定設置
                self.engine = create_engine(
                    self.connection_string,
                    echo=False,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    connect_args={
                        "TrustServerCertificate": "yes",
                        "Connection Timeout": 30
                    }
                )
            elif 'postgresql' in self.connection_string:
                # PostgreSQL settings | PostgreSQL設置
                self.engine = create_engine(
                    self.connection_string,
                    echo=False,
                    pool_pre_ping=True,
                    pool_recycle=3600
                )
            else:
                # SQLite settings | SQLite設置
                self.engine = create_engine(
                    self.connection_string,
                    echo=False,
                    connect_args={"check_same_thread": False}
                )
            
            # Create session factory | 創建會話工廠
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            logger.info(f"Database engine initialized: {self._get_db_type()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise
    
    def _get_db_type(self) -> str:
        """Get database type string | 取得資料庫類型字串"""
        if 'mssql' in self.connection_string:
            return "SQL Server"
        elif 'postgresql' in self.connection_string:
            return "PostgreSQL"
        else:
            return "SQLite"
    
    @contextmanager
    def get_session(self):
        """
        Context manager for database sessions
        資料庫會話的上下文管理器
        """
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def test_connection(self) -> bool:
        """
        Test database connection
        測試資料庫連接
        """
        try:
            with self.engine.connect() as connection:
                if 'mssql' in self.connection_string:
                    result = connection.execute(text("SELECT @@VERSION"))
                elif 'postgresql' in self.connection_string:
                    result = connection.execute(text("SELECT version()"))
                else:
                    result = connection.execute(text("SELECT sqlite_version()"))
                
                version_info = result.fetchone()[0]
                logger.info(f"Database connection successful: {version_info}")
                return True
                
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def create_tables(self):
        """
        Create all tables defined in models
        創建模型中定義的所有資料表
        """
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute SQL query and return DataFrame
        執行SQL查詢並返回DataFrame
        """
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql_query(query, connection, params=params)
                return df
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def save_dataframe(self, df: pd.DataFrame, table_name: str, 
                      if_exists: str = 'replace', index: bool = True):
        """
        Save DataFrame to database table
        將DataFrame保存到資料庫表
        """
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=index)
            logger.info(f"DataFrame saved to table '{table_name}': {len(df)} rows")
        except Exception as e:
            logger.error(f"Failed to save DataFrame: {e}")
            raise


# Global database manager instance | 全域資料庫管理器實例
db_manager = None


def get_database_manager() -> DatabaseManager:
    """
    Get or create global database manager
    取得或創建全域資料庫管理器
    """
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager


def init_database(connection_string: Optional[str] = None) -> DatabaseManager:
    """
    Initialize database with custom connection string
    使用自定義連接字串初始化資料庫
    """
    global db_manager
    db_manager = DatabaseManager(connection_string)
    return db_manager


# Convenience functions | 便利函數
def test_database_connection() -> bool:
    """Test current database connection | 測試當前資料庫連接"""
    return get_database_manager().test_connection()


def save_trading_data(df: pd.DataFrame, symbol: str):
    """
    Save trading data to database
    將交易數據保存到資料庫
    """
    table_name = f"trading_data_{symbol.lower().replace('=', '').replace('x', '')}"
    get_database_manager().save_dataframe(df, table_name)


def load_trading_data(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Load trading data from database
    從資料庫載入交易數據
    """
    table_name = f"trading_data_{symbol.lower().replace('=', '').replace('x', '')}"
    
    query = f"SELECT * FROM {table_name}"
    params = {}
    
    if start_date and end_date:
        query += " WHERE date BETWEEN %(start_date)s AND %(end_date)s"
        params = {"start_date": start_date, "end_date": end_date}
    
    return get_database_manager().execute_query(query, params)