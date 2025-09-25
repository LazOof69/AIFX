"""
AIFX Logging System | AIFX日誌系統

This module provides structured logging capabilities for the AIFX trading system.
此模組為AIFX交易系統提供結構化日誌功能。

Features | 功能:
- Structured logging with context | 帶上下文的結構化日誌
- Multiple output formats (JSON, console) | 多種輸出格式（JSON、控制台）
- Log level management | 日誌級別管理
- Trading-specific log events | 交易特定的日誌事件
"""

import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum


class LogLevel(Enum):
    """
    Log level enumeration | 日誌級別枚舉
    """
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class TradingEventType(Enum):
    """
    Trading event types for specialized logging | 交易事件類型用於專門日誌記錄
    """
    DATA_DOWNLOAD = "DATA_DOWNLOAD"          # Data download events | 數據下載事件
    SIGNAL_GENERATED = "SIGNAL_GENERATED"    # Trading signal events | 交易信號事件
    ORDER_PLACED = "ORDER_PLACED"            # Order placement | 訂單下達
    ORDER_FILLED = "ORDER_FILLED"            # Order execution | 訂單執行
    POSITION_OPENED = "POSITION_OPENED"      # Position opening | 倉位開啟
    POSITION_CLOSED = "POSITION_CLOSED"      # Position closing | 倉位關閉
    RISK_EVENT = "RISK_EVENT"                # Risk management events | 風險管理事件
    MODEL_PREDICTION = "MODEL_PREDICTION"    # AI model predictions | AI模型預測
    BACKTEST_START = "BACKTEST_START"        # Backtesting start | 回測開始
    BACKTEST_END = "BACKTEST_END"            # Backtesting end | 回測結束
    ERROR_OCCURRED = "ERROR_OCCURRED"        # Error events | 錯誤事件


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging | 用於結構化日誌的自定義JSON格式器"""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }
        
        # Add any extra attributes | 添加任何額外屬性
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
                          'module', 'exc_info', 'exc_text', 'stack_info', 'lineno', 'funcName',
                          'created', 'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage']:
                log_obj[key] = value
        
        return json.dumps(log_obj)


def setup_logging(
    log_level: str = "INFO",
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_dir: Optional[str] = None
) -> logging.Logger:
    """
    Setup structured logging for AIFX system | 為AIFX系統設置結構化日誌
    
    Args:
        log_level: Logging level | 日誌級別
        log_to_file: Enable file logging | 啟用文件日誌
        log_to_console: Enable console logging | 啟用控制台日誌
        log_dir: Log directory path | 日誌目錄路徑
        
    Returns:
        Configured logger instance | 配置好的日誌器實例
    """
    # Get root logger | 獲取根日誌器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers | 清除現有處理器
    root_logger.handlers = []
    
    # Console handler | 控制台處理器
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Console formatter | 控制台格式器
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler | 文件處理器
    if log_to_file:
        if log_dir is None:
            log_dir = "logs"
        
        # Create log directory | 創建日誌目錄
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create log file with timestamp | 創建帶時間戳的日誌文件
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"aifx_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # JSON formatter for files | 文件的JSON格式器
        file_formatter = JSONFormatter()
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Return configured logger | 返回配置好的日誌器
    return logging.getLogger("AIFX")


class AIFXLogger:
    """
    AIFX specialized logger class | AIFX專用日誌器類
    """
    
    def __init__(self, logger_name: str = "AIFX"):
        """
        Initialize AIFX logger | 初始化AIFX日誌器
        
        Args:
            logger_name: Name of the logger | 日誌器名稱
        """
        self.logger = logging.getLogger(logger_name)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def log_trading_event(
        self,
        event_type: TradingEventType,
        message: str,
        symbol: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log trading-specific events | 記錄交易特定事件
        
        Args:
            event_type: Type of trading event | 交易事件類型
            message: Log message | 日誌消息
            symbol: Trading symbol | 交易品種
            **kwargs: Additional context data | 額外上下文數據
        """
        context = {
            "event_type": event_type.value,
            "session_id": self.session_id,
            **kwargs
        }
        
        if symbol:
            context["symbol"] = symbol
        
        # Create log record with extra attributes | 創建帶額外屬性的日誌記錄
        self.logger.info(message, extra=context)
    
    def log_data_event(
        self,
        operation: str,
        symbol: str,
        timeframe: str,
        records_count: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Log data processing events | 記錄數據處理事件
        
        Args:
            operation: Data operation type | 數據操作類型
            symbol: Trading symbol | 交易品種
            timeframe: Data timeframe | 數據時間框架
            records_count: Number of records | 記錄數量
            **kwargs: Additional context | 額外上下文
        """
        context = {
            "event_type": TradingEventType.DATA_DOWNLOAD.value,
            "operation": operation,
            "symbol": symbol,
            "timeframe": timeframe,
            "session_id": self.session_id,
            **kwargs
        }
        
        if records_count is not None:
            context["records_count"] = records_count
            
        message = f"Data {operation}: {symbol} [{timeframe}]"
        if records_count:
            message += f" - {records_count} records"
            
        self.logger.info(message, extra=context)
    
    def log_model_prediction(
        self,
        model_name: str,
        symbol: str,
        prediction: Any,
        confidence: Optional[float] = None,
        features_count: Optional[int] = None,
        **kwargs
    ) -> None:
        """
        Log AI model predictions | 記錄AI模型預測
        
        Args:
            model_name: Name of the model | 模型名稱
            symbol: Trading symbol | 交易品種
            prediction: Model prediction | 模型預測
            confidence: Prediction confidence | 預測信心度
            features_count: Number of features used | 使用的特徵數量
            **kwargs: Additional context | 額外上下文
        """
        context = {
            "event_type": TradingEventType.MODEL_PREDICTION.value,
            "model_name": model_name,
            "symbol": symbol,
            "prediction": str(prediction),
            "session_id": self.session_id,
            **kwargs
        }
        
        if confidence is not None:
            context["confidence"] = confidence
            
        if features_count is not None:
            context["features_count"] = features_count
            
        message = f"Model prediction: {model_name} -> {symbol} = {prediction}"
        if confidence:
            message += f" (confidence: {confidence:.3f})"
            
        self.logger.info(message, **context)
    
    def log_trade_signal(
        self,
        signal_type: str,  # BUY, SELL, HOLD
        symbol: str,
        price: float,
        confidence: float,
        indicators: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> None:
        """
        Log trading signals | 記錄交易信號
        
        Args:
            signal_type: Type of signal (BUY/SELL/HOLD) | 信號類型
            symbol: Trading symbol | 交易品種
            price: Current price | 當前價格
            confidence: Signal confidence | 信號信心度
            indicators: Technical indicators values | 技術指標值
            **kwargs: Additional context | 額外上下文
        """
        context = {
            "event_type": TradingEventType.SIGNAL_GENERATED.value,
            "signal_type": signal_type,
            "symbol": symbol,
            "price": price,
            "confidence": confidence,
            "session_id": self.session_id,
            **kwargs
        }
        
        if indicators:
            context["indicators"] = indicators
            
        message = f"Trading signal: {signal_type} {symbol} @ {price:.5f} (confidence: {confidence:.3f})"
        self.logger.info(message, **context)
    
    def log_risk_event(
        self,
        risk_type: str,
        symbol: str,
        current_value: float,
        threshold: float,
        action_taken: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Log risk management events | 記錄風險管理事件
        
        Args:
            risk_type: Type of risk event | 風險事件類型
            symbol: Trading symbol | 交易品種
            current_value: Current risk value | 當前風險值
            threshold: Risk threshold | 風險閾值
            action_taken: Action taken | 採取的行動
            **kwargs: Additional context | 額外上下文
        """
        context = {
            "event_type": TradingEventType.RISK_EVENT.value,
            "risk_type": risk_type,
            "symbol": symbol,
            "current_value": current_value,
            "threshold": threshold,
            "session_id": self.session_id,
            **kwargs
        }
        
        if action_taken:
            context["action_taken"] = action_taken
            
        message = f"Risk event: {risk_type} {symbol} - {current_value:.3f} (threshold: {threshold:.3f})"
        if action_taken:
            message += f" -> {action_taken}"
            
        self.logger.warning(message, extra=context)
    
    def log_backtest_summary(
        self,
        start_date: str,
        end_date: str,
        total_trades: int,
        win_rate: float,
        total_return: float,
        max_drawdown: float,
        sharpe_ratio: float,
        **kwargs
    ) -> None:
        """
        Log backtesting summary | 記錄回測摘要
        
        Args:
            start_date: Backtest start date | 回測開始日期
            end_date: Backtest end date | 回測結束日期
            total_trades: Total number of trades | 總交易次數
            win_rate: Win rate percentage | 勝率百分比
            total_return: Total return percentage | 總回報百分比
            max_drawdown: Maximum drawdown | 最大回撤
            sharpe_ratio: Sharpe ratio | 夏普比率
            **kwargs: Additional metrics | 額外指標
        """
        context = {
            "event_type": TradingEventType.BACKTEST_END.value,
            "start_date": start_date,
            "end_date": end_date,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "session_id": self.session_id,
            **kwargs
        }
        
        message = (
            f"Backtest completed: {start_date} to {end_date} | "
            f"Trades: {total_trades}, Win Rate: {win_rate:.1f}%, "
            f"Return: {total_return:.2f}%, Drawdown: {max_drawdown:.2f}%, "
            f"Sharpe: {sharpe_ratio:.2f}"
        )
        
        self.logger.info(message, **context)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message | 記錄錯誤消息"""
        kwargs['event_type'] = TradingEventType.ERROR_OCCURRED.value
        self.logger.error(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message | 記錄警告消息"""
        self.logger.warning(message, extra=kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message | 記錄信息消息"""
        self.logger.info(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message | 記錄調試消息"""
        self.logger.debug(message, extra=kwargs)

    def log_exception(
        self,
        exception: Exception,
        component: str = "unknown",
        operation: str = "unknown",
        additional_context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log exceptions with structured context | 使用結構化上下文記錄異常

        Args:
            exception: The exception to log
            component: Component where error occurred
            operation: Operation being performed
            additional_context: Additional context data
        """
        from .exceptions import AIFXBaseException, ErrorSeverity, ErrorCategory

        # Prepare context data
        context = {
            "event_type": TradingEventType.ERROR_OCCURRED.value,
            "component": component,
            "operation": operation,
            "session_id": self.session_id
        }

        if additional_context:
            context.update(additional_context)

        if isinstance(exception, AIFXBaseException):
            # Use structured data from AIFX exceptions
            error_data = exception.to_dict()
            context.update(error_data)
            message = f"AIFX Exception in {component}.{operation}: {exception.message}"
        else:
            # Handle standard exceptions
            context.update({
                "error_type": exception.__class__.__name__,
                "error_message": str(exception),
                "severity": ErrorSeverity.MEDIUM.value,
                "category": ErrorCategory.SYSTEM.value
            })
            message = f"Exception in {component}.{operation}: {str(exception)}"

        self.logger.error(message, extra=context, exc_info=True)

    def log_error_with_recovery(
        self,
        error_message: str,
        component: str,
        operation: str,
        recovery_action: str,
        success: bool = True,
        **kwargs
    ) -> None:
        """
        Log errors with recovery attempts | 記錄帶恢復嘗試的錯誤

        Args:
            error_message: Description of the error
            component: Component where error occurred
            operation: Operation that failed
            recovery_action: Action taken to recover
            success: Whether recovery was successful
            **kwargs: Additional context
        """
        context = {
            "event_type": "ERROR_RECOVERY",
            "component": component,
            "operation": operation,
            "recovery_action": recovery_action,
            "recovery_success": success,
            "session_id": self.session_id,
            **kwargs
        }

        level = "WARNING" if success else "ERROR"
        status = "successful" if success else "failed"
        message = f"Error recovery {status} in {component}.{operation}: {error_message}"

        if success:
            self.logger.warning(message, extra=context)
        else:
            self.logger.error(message, extra=context)

    def log_performance_warning(
        self,
        component: str,
        operation: str,
        duration: float,
        threshold: float,
        **kwargs
    ) -> None:
        """
        Log performance warnings | 記錄性能警告

        Args:
            component: Component name
            operation: Operation name
            duration: Actual duration in seconds
            threshold: Expected threshold in seconds
            **kwargs: Additional context
        """
        context = {
            "event_type": "PERFORMANCE_WARNING",
            "component": component,
            "operation": operation,
            "duration": duration,
            "threshold": threshold,
            "performance_ratio": duration / threshold,
            "session_id": self.session_id,
            **kwargs
        }

        message = f"Performance warning: {component}.{operation} took {duration:.2f}s (threshold: {threshold:.2f}s)"
        self.logger.warning(message, extra=context)


# Global logger instance | 全局日誌器實例
logger = None

def get_logger(logger_name: str = "AIFX") -> AIFXLogger:
    """
    Get or create AIFX logger instance | 獲取或創建AIFX日誌器實例
    
    Args:
        logger_name: Logger name | 日誌器名稱
        
    Returns:
        AIFXLogger instance | AIFX日誌器實例
    """
    global logger
    if logger is None:
        # Setup logging on first call | 第一次調用時設置日誌
        setup_logging()
        logger = AIFXLogger(logger_name)
    return logger


if __name__ == "__main__":
    # Example usage | 使用示例
    setup_logging(log_level="INFO")
    aifx_logger = get_logger()
    
    # Test different log types | 測試不同日誌類型
    aifx_logger.info("AIFX system started | AIFX系統啟動")
    
    aifx_logger.log_data_event(
        operation="download",
        symbol="EURUSD=X",
        timeframe="1h",
        records_count=1000
    )
    
    aifx_logger.log_trade_signal(
        signal_type="BUY",
        symbol="EURUSD=X", 
        price=1.0850,
        confidence=0.75,
        indicators={"RSI": 35.2, "MACD": 0.0012}
    )
    
    aifx_logger.log_model_prediction(
        model_name="XGBoost",
        symbol="USDJPY=X",
        prediction="UP",
        confidence=0.68,
        features_count=25
    )


def setup_logger(logger_name: str = "AIFX") -> AIFXLogger:
    """
    Setup AIFX logger instance (alias for get_logger) | 設置AIFX日誌記錄器實例（get_logger的別名）
    
    Args:
        logger_name: Name of the logger | 記錄器名稱
        
    Returns:
        AIFXLogger instance | AIFXLogger實例
    """
    return get_logger(logger_name)