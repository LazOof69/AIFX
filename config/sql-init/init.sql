-- AIFX Database Initialization Script for SQL Server
-- AIFX SQL Server 資料庫初始化腳本

USE master;
GO

-- Create AIFX database if it doesn't exist | 如果不存在則創建AIFX資料庫
IF NOT EXISTS (SELECT name FROM sys.databases WHERE name = 'aifx')
BEGIN
    CREATE DATABASE aifx;
    PRINT 'Database [aifx] created successfully';
END
ELSE
BEGIN
    PRINT 'Database [aifx] already exists';
END
GO

USE aifx;
GO

-- Create trading data table | 創建交易數據表
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='trading_data_eurusd' AND xtype='U')
BEGIN
    CREATE TABLE trading_data_eurusd (
        id BIGINT IDENTITY(1,1) PRIMARY KEY,
        datetime DATETIME2 NOT NULL,
        open_price DECIMAL(18,6) NOT NULL,
        high_price DECIMAL(18,6) NOT NULL,
        low_price DECIMAL(18,6) NOT NULL,
        close_price DECIMAL(18,6) NOT NULL,
        volume BIGINT NOT NULL DEFAULT 0,
        created_at DATETIME2 DEFAULT GETDATE(),
        updated_at DATETIME2 DEFAULT GETDATE(),
        INDEX IX_trading_data_eurusd_datetime (datetime)
    );
    PRINT 'Table [trading_data_eurusd] created successfully';
END
GO

-- Create trading data table for USD/JPY | 創建USD/JPY交易數據表
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='trading_data_usdjpy' AND xtype='U')
BEGIN
    CREATE TABLE trading_data_usdjpy (
        id BIGINT IDENTITY(1,1) PRIMARY KEY,
        datetime DATETIME2 NOT NULL,
        open_price DECIMAL(18,6) NOT NULL,
        high_price DECIMAL(18,6) NOT NULL,
        low_price DECIMAL(18,6) NOT NULL,
        close_price DECIMAL(18,6) NOT NULL,
        volume BIGINT NOT NULL DEFAULT 0,
        created_at DATETIME2 DEFAULT GETDATE(),
        updated_at DATETIME2 DEFAULT GETDATE(),
        INDEX IX_trading_data_usdjpy_datetime (datetime)
    );
    PRINT 'Table [trading_data_usdjpy] created successfully';
END
GO

-- Create signals table | 創建信號表
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='trading_signals' AND xtype='U')
BEGIN
    CREATE TABLE trading_signals (
        id BIGINT IDENTITY(1,1) PRIMARY KEY,
        symbol NVARCHAR(20) NOT NULL,
        datetime DATETIME2 NOT NULL,
        signal_type NVARCHAR(10) NOT NULL CHECK (signal_type IN ('BUY', 'SELL', 'HOLD')),
        confidence DECIMAL(5,4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
        strength DECIMAL(5,4) NOT NULL CHECK (strength >= 0 AND strength <= 1),
        source NVARCHAR(50) NOT NULL,
        metadata NVARCHAR(MAX),
        created_at DATETIME2 DEFAULT GETDATE(),
        INDEX IX_trading_signals_symbol_datetime (symbol, datetime),
        INDEX IX_trading_signals_datetime (datetime)
    );
    PRINT 'Table [trading_signals] created successfully';
END
GO

-- Create model performance table | 創建模型性能表
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='model_performance' AND xtype='U')
BEGIN
    CREATE TABLE model_performance (
        id BIGINT IDENTITY(1,1) PRIMARY KEY,
        model_name NVARCHAR(100) NOT NULL,
        symbol NVARCHAR(20) NOT NULL,
        accuracy DECIMAL(5,4),
        precision_score DECIMAL(5,4),
        recall_score DECIMAL(5,4),
        f1_score DECIMAL(5,4),
        training_date DATETIME2 NOT NULL,
        evaluation_date DATETIME2 NOT NULL,
        parameters NVARCHAR(MAX),
        created_at DATETIME2 DEFAULT GETDATE(),
        INDEX IX_model_performance_model_symbol (model_name, symbol)
    );
    PRINT 'Table [model_performance] created successfully';
END
GO

-- Create configuration table | 創建配置表
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='system_config' AND xtype='U')
BEGIN
    CREATE TABLE system_config (
        id BIGINT IDENTITY(1,1) PRIMARY KEY,
        config_key NVARCHAR(100) NOT NULL UNIQUE,
        config_value NVARCHAR(MAX) NOT NULL,
        config_type NVARCHAR(20) NOT NULL DEFAULT 'string',
        description NVARCHAR(500),
        created_at DATETIME2 DEFAULT GETDATE(),
        updated_at DATETIME2 DEFAULT GETDATE()
    );
    PRINT 'Table [system_config] created successfully';
END
GO

-- Insert default configuration | 插入預設配置
INSERT INTO system_config (config_key, config_value, config_type, description) VALUES
('trading.symbols', '["EURUSD=X", "USDJPY=X"]', 'json', 'Default trading symbols'),
('trading.timeframe', '1h', 'string', 'Default trading timeframe'),
('models.retrain_frequency', '7', 'integer', 'Model retraining frequency in days'),
('risk.max_position_size', '0.02', 'float', 'Maximum position size as percentage of account'),
('signals.min_confidence', '0.6', 'float', 'Minimum signal confidence threshold');

PRINT 'Default configuration inserted successfully';
GO

-- Create stored procedure for signal insertion | 創建信號插入存儲過程
CREATE OR ALTER PROCEDURE sp_InsertTradingSignal
    @symbol NVARCHAR(20),
    @datetime DATETIME2,
    @signal_type NVARCHAR(10),
    @confidence DECIMAL(5,4),
    @strength DECIMAL(5,4),
    @source NVARCHAR(50),
    @metadata NVARCHAR(MAX) = NULL
AS
BEGIN
    SET NOCOUNT ON;
    
    INSERT INTO trading_signals (symbol, datetime, signal_type, confidence, strength, source, metadata)
    VALUES (@symbol, @datetime, @signal_type, @confidence, @strength, @source, @metadata);
    
    SELECT SCOPE_IDENTITY() as signal_id;
END
GO

PRINT 'Stored procedure [sp_InsertTradingSignal] created successfully';

-- Create view for latest signals | 創建最新信號視圖
CREATE OR ALTER VIEW v_latest_signals AS
SELECT 
    s1.symbol,
    s1.datetime,
    s1.signal_type,
    s1.confidence,
    s1.strength,
    s1.source
FROM trading_signals s1
WHERE s1.datetime = (
    SELECT MAX(s2.datetime)
    FROM trading_signals s2
    WHERE s2.symbol = s1.symbol
);
GO

PRINT 'View [v_latest_signals] created successfully';

-- Display summary | 顯示摘要
PRINT '=== AIFX Database Setup Complete ===';
PRINT 'Database: aifx';
PRINT 'Tables created: trading_data_eurusd, trading_data_usdjpy, trading_signals, model_performance, system_config';
PRINT 'Stored procedures: sp_InsertTradingSignal';
PRINT 'Views: v_latest_signals';
PRINT '=====================================';

GO