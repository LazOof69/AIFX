"""
AIFX - Main Application Entry Point
AIFX - 主應用程式入口點

This module creates and configures the FastAPI application for production deployment.
此模組創建並配置用於生產部署的FastAPI應用程式。
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Add the project root to Python path | 將專案根目錄添加到Python路徑
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.main.python.utils.logger import setup_logger
from src.main.python.utils.config import load_config
from src.main.python.core.ai_signal_combiner import AISignalCombiner
from src.main.python.services.model_manager import ModelManager

# Initialize logger | 初始化日誌記錄器
logger = setup_logger(__name__)

# Global application state | 全域應用程式狀態
app_state = {
    "model_manager": None,
    "signal_combiner": None,
    "startup_time": None,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    應用程式生命週期管理器，用於啟動和關閉事件。
    """
    # Startup | 啟動
    logger.info("🚀 Starting AIFX Application...")
    logger.info("🚀 啟動AIFX應用程式...")
    
    try:
        # Initialize configuration | 初始化配置
        config = load_config()
        logger.info("✅ Configuration loaded successfully")
        
        # Initialize Model Manager | 初始化模型管理器
        app_state["model_manager"] = ModelManager()
        logger.info("✅ Model Manager initialized")
        
        # Initialize AI Signal Combiner | 初始化AI信號組合器
        app_state["signal_combiner"] = AISignalCombiner()
        logger.info("✅ AI Signal Combiner initialized")
        
        # Record startup time | 記錄啟動時間
        import time
        app_state["startup_time"] = time.time()
        
        logger.info("🎯 AIFX Application startup completed successfully!")
        logger.info("🎯 AIFX應用程式啟動成功完成！")
        
    except Exception as e:
        logger.error(f"❌ Failed to start AIFX Application: {e}")
        logger.error(f"❌ AIFX應用程式啟動失敗：{e}")
        raise
    
    yield
    
    # Shutdown | 關閉
    logger.info("🛑 Shutting down AIFX Application...")
    logger.info("🛑 關閉AIFX應用程式...")
    
    # Cleanup resources | 清理資源
    if app_state["model_manager"]:
        # Clean up model manager resources | 清理模型管理器資源
        logger.info("🧹 Cleaning up Model Manager...")
    
    if app_state["signal_combiner"]:
        # Clean up signal combiner resources | 清理信號組合器資源
        logger.info("🧹 Cleaning up Signal Combiner...")
    
    logger.info("✅ AIFX Application shutdown completed")
    logger.info("✅ AIFX應用程式關閉完成")

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    創建並配置FastAPI應用程式。
    
    Returns:
        FastAPI: Configured FastAPI application instance
    """
    
    # Create FastAPI application | 創建FastAPI應用程式
    app = FastAPI(
        title="AIFX - AI-Enhanced Forex Trading System",
        description="Professional quantitative trading system with AI models for EUR/USD and USD/JPY",
        version="4.0.0",
        docs_url="/docs" if os.getenv("AIFX_ENV") != "production" else None,
        redoc_url="/redoc" if os.getenv("AIFX_ENV") != "production" else None,
        lifespan=lifespan
    )
    
    # Add middleware | 添加中間件
    
    # CORS middleware | CORS中間件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production | 為生產環境適當配置
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Gzip compression middleware | Gzip壓縮中間件
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Exception handlers | 異常處理器
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", None)
            }
        )
    
    # Health check endpoint | 健康檢查端點
    @app.get("/health")
    async def health_check():
        """
        Health check endpoint for load balancers and monitoring.
        用於負載均衡器和監控的健康檢查端點。
        """
        import time
        current_time = time.time()
        uptime = current_time - app_state.get("startup_time", current_time)
        
        return {
            "status": "healthy",
            "timestamp": current_time,
            "uptime_seconds": uptime,
            "environment": os.getenv("AIFX_ENV", "unknown"),
            "version": "4.0.0",
            "components": {
                "model_manager": "ready" if app_state.get("model_manager") else "not_ready",
                "signal_combiner": "ready" if app_state.get("signal_combiner") else "not_ready",
            }
        }
    
    # Readiness check endpoint | 就緒檢查端點
    @app.get("/ready")
    async def readiness_check():
        """
        Readiness check endpoint to verify all components are ready.
        就緒檢查端點以驗證所有組件是否就緒。
        """
        ready = (
            app_state.get("model_manager") is not None and
            app_state.get("signal_combiner") is not None
        )
        
        if not ready:
            raise HTTPException(
                status_code=503,
                detail="Service not ready - components still initializing"
            )
        
        return {
            "status": "ready",
            "timestamp": time.time(),
            "message": "All components are ready"
        }
    
    # Metrics endpoint | 指標端點
    @app.get("/metrics")
    async def get_metrics():
        """
        Prometheus metrics endpoint.
        Prometheus指標端點。
        """
        # This would integrate with prometheus_client
        # 這將與prometheus_client整合
        return {"message": "Metrics endpoint - integrate with prometheus_client"}
    
    # Trading API endpoints | 交易API端點
    @app.get("/api/v1/signal")
    async def get_trading_signal(symbol: str = "EURUSD"):
        """
        Get current trading signal for specified symbol.
        獲取指定品種的當前交易信號。
        """
        try:
            if not app_state.get("signal_combiner"):
                raise HTTPException(status_code=503, detail="Signal combiner not ready")
            
            # This would integrate with the actual signal combiner
            # 這將與實際的信號組合器整合
            return {
                "symbol": symbol,
                "signal": "HOLD",  # Placeholder | 佔位符
                "confidence": 0.5,
                "timestamp": time.time(),
                "message": "Trading signal endpoint - integrate with signal combiner"
            }
            
        except Exception as e:
            logger.error(f"Error getting trading signal: {e}")
            raise HTTPException(status_code=500, detail="Failed to get trading signal")
    
    # Model status endpoint | 模型狀態端點
    @app.get("/api/v1/models/status")
    async def get_model_status():
        """
        Get status of all AI models.
        獲取所有AI模型的狀態。
        """
        try:
            if not app_state.get("model_manager"):
                raise HTTPException(status_code=503, detail="Model manager not ready")
            
            # This would integrate with the actual model manager
            # 這將與實際的模型管理器整合
            return {
                "models": {
                    "xgboost": {"status": "ready", "version": "1.0.0"},
                    "random_forest": {"status": "ready", "version": "1.0.0"},
                    "lstm": {"status": "ready", "version": "1.0.0"}
                },
                "timestamp": time.time(),
                "message": "Model status endpoint - integrate with model manager"
            }
            
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            raise HTTPException(status_code=500, detail="Failed to get model status")
    
    logger.info("✅ FastAPI application created and configured")
    return app

# Create the application instance | 創建應用程式實例
app = create_app()

def main():
    """
    Main entry point for running the application directly.
    直接運行應用程式的主入口點。
    """
    import time
    
    # Development server configuration | 開發服務器配置
    config = {
        "app": "src.main.python.app:app",
        "host": "0.0.0.0",
        "port": 8000,
        "reload": os.getenv("AIFX_ENV") == "development",
        "log_level": "info",
        "access_log": True,
    }
    
    logger.info("🚀 Starting AIFX development server...")
    logger.info(f"📋 Environment: {os.getenv('AIFX_ENV', 'development')}")
    logger.info(f"📍 Server: http://{config['host']}:{config['port']}")
    
    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()