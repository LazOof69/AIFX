-- AIFX Database Initialization Script
-- AIFX 資料庫初始化腳本

-- Create AIFX database if not exists
-- 如果不存在則創建AIFX資料庫
CREATE DATABASE IF NOT EXISTS aifx;

-- Create aifx user if not exists
-- 如果不存在則創建aifx用戶
DO
$do$
BEGIN
   IF NOT EXISTS (
      SELECT FROM pg_catalog.pg_roles
      WHERE  rolname = 'aifx') THEN
      
      CREATE ROLE aifx LOGIN PASSWORD 'password';
   END IF;
END
$do$;

-- Grant privileges to aifx user
-- 授予aifx用戶權限
GRANT ALL PRIVILEGES ON DATABASE aifx TO aifx;

-- Connect to aifx database
-- 連接到aifx資料庫
\c aifx;

-- Create basic tables for AIFX application
-- 為AIFX應用程式創建基本表

-- Trading data table
CREATE TABLE IF NOT EXISTS trading_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open_price DECIMAL(10,5),
    high_price DECIMAL(10,5),
    low_price DECIMAL(10,5),
    close_price DECIMAL(10,5),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- AI model predictions table
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL,
    predicted_direction INTEGER, -- 1 for up, -1 for down, 0 for sideways
    confidence DECIMAL(5,4),
    features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading signals table
CREATE TABLE IF NOT EXISTS trading_signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL,
    signal_timestamp TIMESTAMP NOT NULL,
    signal_type VARCHAR(10) NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    technical_score DECIMAL(5,4),
    ai_score DECIMAL(5,4),
    combined_score DECIMAL(5,4),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trading performance table
CREATE TABLE IF NOT EXISTS trading_performance (
    id SERIAL PRIMARY KEY,
    strategy_name VARCHAR(100) NOT NULL,
    backtest_start DATE NOT NULL,
    backtest_end DATE NOT NULL,
    total_return DECIMAL(10,6),
    sharpe_ratio DECIMAL(8,6),
    max_drawdown DECIMAL(8,6),
    win_rate DECIMAL(5,4),
    total_trades INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
-- 創建索引以提高性能
CREATE INDEX IF NOT EXISTS idx_trading_data_symbol_timestamp ON trading_data(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_model_predictions_symbol_timestamp ON model_predictions(symbol, prediction_timestamp);
CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_timestamp ON trading_signals(symbol, signal_timestamp);
CREATE INDEX IF NOT EXISTS idx_trading_performance_strategy ON trading_performance(strategy_name, backtest_start);

-- Grant permissions on tables
-- 授予表權限
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aifx;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aifx;

-- Success message
-- 成功消息
SELECT 'AIFX database initialization completed successfully | AIFX資料庫初始化成功完成' as status;