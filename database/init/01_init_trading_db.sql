-- AIFX 24/7 Trading System - Database Initialization
-- AIFX 24/7 交易系統 - 資料庫初始化
--
-- This script initializes the PostgreSQL database for 24/7 USD/JPY trading
-- with all necessary tables, indexes, and functions for optimal performance.
--
-- 此腳本初始化PostgreSQL資料庫用於24/7美元/日圓交易，
-- 包含所有必要的表、索引和函數以實現最佳性能。

-- ============================================================================
-- DATABASE SETUP | 資料庫設置
-- ============================================================================

-- Set timezone for consistent trading hours
-- 設置時區以保持一致的交易時間
SET timezone = 'UTC';

-- Create extensions for better performance
-- 創建擴展以提高性能
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ============================================================================
-- TRADING TABLES | 交易表
-- ============================================================================

-- Historical price data table (main data storage)
-- 歷史價格數據表（主要數據存儲）
CREATE TABLE IF NOT EXISTS historical_prices (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL DEFAULT 'USD/JPY',
    open DECIMAL(10, 5) NOT NULL,
    high DECIMAL(10, 5) NOT NULL,
    low DECIMAL(10, 5) NOT NULL,
    close DECIMAL(10, 5) NOT NULL,
    volume BIGINT DEFAULT 0,
    spread DECIMAL(8, 5),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT valid_ohlc CHECK (high >= low AND high >= open AND high >= close AND low <= open AND low <= close),
    CONSTRAINT positive_prices CHECK (open > 0 AND high > 0 AND low > 0 AND close > 0)
);

-- Trading signals table
-- 交易信號表
CREATE TABLE IF NOT EXISTS trading_signals (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(20) NOT NULL DEFAULT 'USD/JPY',
    signal_type VARCHAR(20) NOT NULL, -- BUY, SELL, HOLD
    confidence DECIMAL(5, 4) NOT NULL,

    -- AI model predictions
    xgb_prediction DECIMAL(5, 4),
    rf_prediction DECIMAL(5, 4),
    lstm_prediction DECIMAL(5, 4),
    ensemble_prediction DECIMAL(5, 4),

    -- Technical indicators
    technical_score DECIMAL(5, 4),
    rsi_value DECIMAL(5, 2),
    macd_signal DECIMAL(8, 5),
    ma_trend VARCHAR(20),

    -- Signal metadata
    signal_strength VARCHAR(20), -- WEAK, MODERATE, STRONG
    execution_status VARCHAR(20) DEFAULT 'PENDING', -- PENDING, EXECUTED, CANCELLED
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT valid_signal CHECK (signal_type IN ('BUY', 'SELL', 'HOLD')),
    CONSTRAINT valid_confidence CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT valid_strength CHECK (signal_strength IN ('WEAK', 'MODERATE', 'STRONG'))
);

-- Active trading positions
-- 活躍交易倉位
CREATE TABLE IF NOT EXISTS trading_positions (
    id BIGSERIAL PRIMARY KEY,
    position_id UUID DEFAULT uuid_generate_v4() UNIQUE,

    -- Position details
    symbol VARCHAR(20) NOT NULL DEFAULT 'USD/JPY',
    side VARCHAR(10) NOT NULL, -- BUY, SELL
    quantity DECIMAL(12, 2) NOT NULL,

    -- Entry information
    entry_time TIMESTAMPTZ NOT NULL,
    entry_price DECIMAL(10, 5) NOT NULL,
    entry_signal_id BIGINT REFERENCES trading_signals(id),

    -- Exit information (nullable until closed)
    exit_time TIMESTAMPTZ,
    exit_price DECIMAL(10, 5),
    exit_reason VARCHAR(50), -- TAKE_PROFIT, STOP_LOSS, MANUAL, TIMEOUT

    -- Risk management
    stop_loss DECIMAL(10, 5),
    take_profit DECIMAL(10, 5),
    trailing_stop DECIMAL(10, 5),

    -- P&L calculation
    unrealized_pnl DECIMAL(15, 2) DEFAULT 0,
    realized_pnl DECIMAL(15, 2),
    pnl_percentage DECIMAL(8, 4),

    -- Position status
    status VARCHAR(20) DEFAULT 'OPEN', -- OPEN, CLOSED, CANCELLED

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT valid_side CHECK (side IN ('BUY', 'SELL')),
    CONSTRAINT positive_quantity CHECK (quantity > 0),
    CONSTRAINT valid_status CHECK (status IN ('OPEN', 'CLOSED', 'CANCELLED')),
    CONSTRAINT valid_prices CHECK (entry_price > 0 AND (exit_price IS NULL OR exit_price > 0))
);

-- Performance metrics and analytics
-- 性能指標和分析
CREATE TABLE IF NOT EXISTS performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    metric_date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL DEFAULT 'USD/JPY',

    -- Daily performance
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate DECIMAL(5, 2) DEFAULT 0,

    -- P&L metrics
    daily_pnl DECIMAL(15, 2) DEFAULT 0,
    cumulative_pnl DECIMAL(15, 2) DEFAULT 0,
    max_drawdown DECIMAL(15, 2) DEFAULT 0,
    max_profit DECIMAL(15, 2) DEFAULT 0,

    -- Risk metrics
    sharpe_ratio DECIMAL(8, 4),
    sortino_ratio DECIMAL(8, 4),
    calmar_ratio DECIMAL(8, 4),
    volatility DECIMAL(8, 4),

    -- Trading volume
    total_volume DECIMAL(18, 2) DEFAULT 0,
    avg_position_size DECIMAL(12, 2),

    -- System metrics
    uptime_percentage DECIMAL(5, 2) DEFAULT 100,
    signal_accuracy DECIMAL(5, 2),
    execution_latency DECIMAL(8, 3), -- milliseconds

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT unique_daily_metric UNIQUE (metric_date, symbol),
    CONSTRAINT valid_percentages CHECK (win_rate >= 0 AND win_rate <= 100 AND uptime_percentage >= 0 AND uptime_percentage <= 100)
);

-- System logs and events
-- 系統日誌和事件
CREATE TABLE IF NOT EXISTS system_logs (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    log_level VARCHAR(20) NOT NULL, -- DEBUG, INFO, WARNING, ERROR, CRITICAL
    component VARCHAR(50) NOT NULL, -- TRADER, ANALYZER, DATABASE, API, etc.
    message TEXT NOT NULL,
    details JSONB,
    session_id UUID,
    user_id VARCHAR(50),

    -- Constraints
    CONSTRAINT valid_log_level CHECK (log_level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'))
);

-- Feature importance tracking (for AI models)
-- 特徵重要性跟蹤（用於AI模型）
CREATE TABLE IF NOT EXISTS feature_importance (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    importance_score DECIMAL(8, 6) NOT NULL,
    rank_position INTEGER,
    training_date DATE NOT NULL,

    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CONSTRAINT unique_model_feature UNIQUE (model_name, feature_name, training_date)
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE | 性能索引
-- ============================================================================

-- Historical prices indexes
CREATE INDEX IF NOT EXISTS idx_historical_prices_timestamp ON historical_prices (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_historical_prices_symbol_timestamp ON historical_prices (symbol, timestamp DESC);

-- Trading signals indexes
CREATE INDEX IF NOT EXISTS idx_trading_signals_timestamp ON trading_signals (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_timestamp ON trading_signals (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trading_signals_type ON trading_signals (signal_type);
CREATE INDEX IF NOT EXISTS idx_trading_signals_status ON trading_signals (execution_status);

-- Trading positions indexes
CREATE INDEX IF NOT EXISTS idx_trading_positions_symbol ON trading_positions (symbol);
CREATE INDEX IF NOT EXISTS idx_trading_positions_status ON trading_positions (status);
CREATE INDEX IF NOT EXISTS idx_trading_positions_entry_time ON trading_positions (entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trading_positions_position_id ON trading_positions (position_id);

-- Performance metrics indexes
CREATE INDEX IF NOT EXISTS idx_performance_metrics_date ON performance_metrics (metric_date DESC);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_symbol_date ON performance_metrics (symbol, metric_date DESC);

-- System logs indexes
CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs (log_level);
CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs (component);

-- ============================================================================
-- FUNCTIONS AND TRIGGERS | 函數和觸發器
-- ============================================================================

-- Function to update the updated_at timestamp
-- 更新updated_at時間戳的函數
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
-- 為updated_at創建觸發器
CREATE TRIGGER update_historical_prices_updated_at
    BEFORE UPDATE ON historical_prices
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trading_positions_updated_at
    BEFORE UPDATE ON trading_positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to calculate position P&L
-- 計算倉位盈虧的函數
CREATE OR REPLACE FUNCTION calculate_position_pnl(
    p_side VARCHAR,
    p_entry_price DECIMAL,
    p_current_price DECIMAL,
    p_quantity DECIMAL
)
RETURNS DECIMAL AS $$
BEGIN
    IF p_side = 'BUY' THEN
        RETURN (p_current_price - p_entry_price) * p_quantity;
    ELSE
        RETURN (p_entry_price - p_current_price) * p_quantity;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function to get latest price for a symbol
-- 獲取品種最新價格的函數
CREATE OR REPLACE FUNCTION get_latest_price(p_symbol VARCHAR DEFAULT 'USD/JPY')
RETURNS DECIMAL AS $$
DECLARE
    latest_price DECIMAL;
BEGIN
    SELECT close INTO latest_price
    FROM historical_prices
    WHERE symbol = p_symbol
    ORDER BY timestamp DESC
    LIMIT 1;

    RETURN COALESCE(latest_price, 0);
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PARTITIONING SETUP (for better performance with large data)
-- 分區設置（為大數據提供更好性能）
-- ============================================================================

-- Enable automatic partitioning for historical_prices by month
-- 為historical_prices按月啟用自動分區
-- (This would require pg_partman extension in production)

-- ============================================================================
-- INITIAL DATA SETUP | 初始數據設置
-- ============================================================================

-- Insert initial system configuration
-- 插入初始系統配置
INSERT INTO system_logs (log_level, component, message) VALUES
('INFO', 'DATABASE', 'AIFX 24/7 Trading Database Initialized Successfully');

-- ============================================================================
-- GRANTS AND PERMISSIONS | 授權和權限
-- ============================================================================

-- Grant permissions to the aifx user
-- 為aifx用戶授予權限
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO aifx;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO aifx;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO aifx;

-- ============================================================================
-- DATABASE MAINTENANCE | 資料庫維護
-- ============================================================================

-- Enable automatic statistics collection
-- 啟用自動統計收集
ALTER SYSTEM SET track_activities = on;
ALTER SYSTEM SET track_counts = on;
ALTER SYSTEM SET track_io_timing = on;

COMMIT;

-- Log successful completion
-- 記錄成功完成
DO $$
BEGIN
    RAISE NOTICE 'AIFX 24/7 Trading Database initialization completed successfully!';
    RAISE NOTICE 'Database is ready for USD/JPY trading operations.';
END $$;