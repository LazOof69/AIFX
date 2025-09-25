"""
AIFX Exception Handling System | AIFX異常處理系統

Custom exception classes and error handling utilities for the AIFX trading system.
AIFX交易系統的自定義異常類和錯誤處理工具。

Features | 功能:
- Trading-specific exceptions | 交易特定異常
- Structured error reporting | 結構化錯誤報告
- Context-aware error tracking | 上下文感知錯誤追蹤
- Error recovery mechanisms | 錯誤恢復機制
"""

import traceback
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass


class ErrorSeverity(Enum):
    """Error severity levels | 錯誤嚴重級別"""
    LOW = "low"           # Non-critical errors that don't affect trading
    MEDIUM = "medium"     # Errors that may affect performance
    HIGH = "high"         # Errors that affect trading functionality
    CRITICAL = "critical" # System-breaking errors requiring immediate attention


class ErrorCategory(Enum):
    """Error categories for better classification | 錯誤分類以便更好地分類"""
    DATA = "data"                    # Data-related errors
    MODEL = "model"                  # AI/ML model errors
    TRADING = "trading"              # Trading execution errors
    BROKER = "broker"                # Broker connection/API errors
    RISK = "risk"                    # Risk management errors
    SYSTEM = "system"                # System-level errors
    CONFIGURATION = "configuration"  # Configuration errors
    NETWORK = "network"              # Network connectivity errors


@dataclass
class ErrorContext:
    """
    Error context information | 錯誤上下文信息
    """
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    component: str
    operation: str
    additional_data: Dict[str, Any]


class AIFXBaseException(Exception):
    """
    Base exception class for AIFX system | AIFX系統基礎異常類
    """

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        component: str = "unknown",
        operation: str = "unknown",
        **additional_data
    ):
        super().__init__(message)
        self.message = message
        self.context = ErrorContext(
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            component=component,
            operation=operation,
            additional_data=additional_data
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging | 將異常轉換為字典用於日誌記錄"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'timestamp': self.context.timestamp.isoformat(),
            'category': self.context.category.value,
            'severity': self.context.severity.value,
            'component': self.context.component,
            'operation': self.context.operation,
            'additional_data': self.context.additional_data,
            'traceback': traceback.format_exc()
        }


# Data-related exceptions | 數據相關異常
class DataException(AIFXBaseException):
    """Base class for data-related errors | 數據相關錯誤的基類"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.DATA, **kwargs)


class DataSourceException(DataException):
    """Data source connection or retrieval errors | 數據源連接或檢索錯誤"""
    pass


class DataValidationException(DataException):
    """Data validation and quality errors | 數據驗證和質量錯誤"""
    pass


class DataProcessingException(DataException):
    """Data preprocessing and feature generation errors | 數據預處理和特徵生成錯誤"""
    pass


# Model-related exceptions | 模型相關異常
class ModelException(AIFXBaseException):
    """Base class for model-related errors | 模型相關錯誤的基類"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.MODEL, **kwargs)


class ModelTrainingException(ModelException):
    """Model training errors | 模型訓練錯誤"""
    pass


class ModelPredictionException(ModelException):
    """Model prediction errors | 模型預測錯誤"""
    pass


class ModelLoadException(ModelException):
    """Model loading and serialization errors | 模型加載和序列化錯誤"""
    pass


# Trading-related exceptions | 交易相關異常
class TradingException(AIFXBaseException):
    """Base class for trading-related errors | 交易相關錯誤的基類"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.TRADING, **kwargs)


class OrderException(TradingException):
    """Order placement and management errors | 訂單下達和管理錯誤"""
    pass


class PositionException(TradingException):
    """Position management errors | 倉位管理錯誤"""
    pass


class ExecutionException(TradingException):
    """Trade execution errors | 交易執行錯誤"""
    pass


# Broker-related exceptions | 券商相關異常
class BrokerException(AIFXBaseException):
    """Base class for broker-related errors | 券商相關錯誤的基類"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.BROKER, **kwargs)


class BrokerConnectionException(BrokerException):
    """Broker connection errors | 券商連接錯誤"""
    pass


class BrokerAPIException(BrokerException):
    """Broker API errors | 券商API錯誤"""
    pass


class BrokerAuthenticationException(BrokerException):
    """Broker authentication errors | 券商認證錯誤"""
    pass


# Risk management exceptions | 風險管理異常
class RiskException(AIFXBaseException):
    """Base class for risk management errors | 風險管理錯誤的基類"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.RISK, severity=ErrorSeverity.HIGH, **kwargs)


class RiskLimitException(RiskException):
    """Risk limit violations | 風險限制違規"""
    pass


class RiskCalculationException(RiskException):
    """Risk calculation errors | 風險計算錯誤"""
    pass


# Configuration exceptions | 配置異常
class ConfigurationException(AIFXBaseException):
    """Base class for configuration errors | 配置錯誤的基類"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)


class ConfigurationLoadException(ConfigurationException):
    """Configuration loading errors | 配置加載錯誤"""
    pass


class ConfigurationValidationException(ConfigurationException):
    """Configuration validation errors | 配置驗證錯誤"""
    pass


# Network exceptions | 網絡異常
class NetworkException(AIFXBaseException):
    """Base class for network-related errors | 網絡相關錯誤的基類"""

    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, **kwargs)


class NetworkConnectionException(NetworkException):
    """Network connectivity errors | 網絡連接錯誤"""
    pass


class NetworkTimeoutException(NetworkException):
    """Network timeout errors | 網絡超時錯誤"""
    pass


class ErrorHandler:
    """
    Centralized error handling and reporting | 集中錯誤處理和報告
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
        self.error_history: List[Dict[str, Any]] = []

    def handle_exception(
        self,
        exception: Exception,
        context: Optional[Dict[str, Any]] = None,
        reraise: bool = True
    ) -> None:
        """
        Handle and log exceptions | 處理和記錄異常

        Args:
            exception: The exception to handle
            context: Additional context information
            reraise: Whether to reraise the exception
        """
        if isinstance(exception, AIFXBaseException):
            error_data = exception.to_dict()
            if context:
                error_data['context'] = context
        else:
            # Convert standard exceptions to our format
            error_data = {
                'error_type': exception.__class__.__name__,
                'message': str(exception),
                'timestamp': datetime.now().isoformat(),
                'category': ErrorCategory.SYSTEM.value,
                'severity': ErrorSeverity.MEDIUM.value,
                'component': context.get('component', 'unknown') if context else 'unknown',
                'operation': context.get('operation', 'unknown') if context else 'unknown',
                'additional_data': context or {},
                'traceback': traceback.format_exc()
            }

        # Log the error
        self.logger.error(f"Exception handled: {error_data['error_type']}", extra=error_data)

        # Track error statistics
        error_type = error_data['error_type']
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1

        # Store in error history (keep last 100)
        self.error_history.append(error_data)
        if len(self.error_history) > 100:
            self.error_history.pop(0)

        if reraise:
            raise exception

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics | 獲取錯誤統計信息"""
        return {
            'total_errors': sum(self.error_counts.values()),
            'error_counts': self.error_counts.copy(),
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }

    def clear_error_history(self) -> None:
        """Clear error history | 清除錯誤歷史"""
        self.error_history.clear()
        self.error_counts.clear()


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_errors(component: str = "unknown", operation: str = "unknown"):
    """
    Decorator for automatic error handling | 自動錯誤處理裝飾器

    Args:
        component: Component name for error tracking
        operation: Operation name for error tracking
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'component': component,
                    'operation': operation,
                    'function': func.__name__,
                    'args': str(args)[:100],  # Truncate for logging
                    'kwargs': str(kwargs)[:100]
                }
                global_error_handler.handle_exception(e, context, reraise=True)
        return wrapper
    return decorator


# Convenience functions for quick error raising
def raise_data_error(message: str, **kwargs):
    """Raise a data-related error | 拋出數據相關錯誤"""
    raise DataException(message, **kwargs)


def raise_model_error(message: str, **kwargs):
    """Raise a model-related error | 拋出模型相關錯誤"""
    raise ModelException(message, **kwargs)


def raise_trading_error(message: str, **kwargs):
    """Raise a trading-related error | 拋出交易相關錯誤"""
    raise TradingException(message, **kwargs)


def raise_broker_error(message: str, **kwargs):
    """Raise a broker-related error | 拋出券商相關錯誤"""
    raise BrokerException(message, **kwargs)


def raise_risk_error(message: str, **kwargs):
    """Raise a risk management error | 拋出風險管理錯誤"""
    raise RiskException(message, **kwargs)


def raise_config_error(message: str, **kwargs):
    """Raise a configuration error | 拋出配置錯誤"""
    raise ConfigurationException(message, **kwargs)


def raise_network_error(message: str, **kwargs):
    """Raise a network-related error | 拋出網絡相關錯誤"""
    raise NetworkException(message, **kwargs)