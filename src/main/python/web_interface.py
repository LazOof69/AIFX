"""
AIFX - Simplified Web Trading Signals Interface
AIFX - 簡化網頁交易信號介面

Simplified 24/7 web interface that displays only entry and exit signals.
簡化的24小時網頁介面，只顯示入場和出場信號。
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import core components - adjusted for Docker container paths
from src.main.python.core.ai_signal_combiner import AISignalCombiner
from src.main.python.utils.data_loader import DataLoader
from src.main.python.utils.feature_generator import FeatureGenerator
from src.main.python.services.model_manager import ModelLifecycleManager
from src.main.python.services.lightweight_signal_service import LightweightSignalService, SignalServiceConfig

logger = logging.getLogger(__name__)


class TradingSignalData:
    """Trading signal data structure for web interface"""

    def __init__(self):
        self.current_signals = {
            'EURUSD': {'action': 'HOLD', 'confidence': 0.0, 'timestamp': None},
            'USDJPY': {'action': 'HOLD', 'confidence': 0.0, 'timestamp': None}
        }
        self.open_positions = {}
        self.daily_stats = {
            'total_signals': 0,
            'entry_signals': 0,
            'exit_signals': 0,
            'successful_trades': 0
        }


class SimplifiedTradingWebApp:
    """
    Simplified 24/7 Trading Web Interface
    簡化的24小時交易網頁介面

    Features:
    - Real-time entry/exit signal display
    - Simple clean interface
    - WebSocket updates
    - 24/7 continuous operation
    """

    def __init__(self):
        self.app = FastAPI(title="AIFX - Trading Signals", version="1.0.0")
        self.signal_data = TradingSignalData()
        self.websocket_connections: List[WebSocket] = []

        # Core trading components - Using lightweight service
        self.signal_service = None
        self.model_manager = None  # Optional for compatibility
        self.signal_combiner = None  # Optional for compatibility

        # Monitoring state
        self.is_running = False
        self.last_signal_time = None
        self.update_interval = 30  # 30 seconds

        self._setup_routes()
        self._setup_static_files()

    def _setup_static_files(self):
        """Setup static files and templates"""
        # Create static directory if it doesn't exist
        static_dir = Path("src/main/resources/static")
        static_dir.mkdir(parents=True, exist_ok=True)

        templates_dir = Path("src/main/resources/templates")
        templates_dir.mkdir(parents=True, exist_ok=True)

        # Mount static files
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        self.templates = Jinja2Templates(directory=str(templates_dir))

    def _setup_routes(self):
        """Setup web routes"""

        @self.app.get("/", response_class=HTMLResponse)
        async def get_trading_interface(request: Request):
            """Main trading signals interface"""
            return self.templates.TemplateResponse("trading_signals.html", {
                "request": request,
                "title": "AIFX Trading Signals"
            })

        @self.app.get("/api/signals")
        async def get_current_signals():
            """Get current trading signals"""
            return JSONResponse({
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "signals": self.signal_data.current_signals,
                "positions": self.signal_data.open_positions,
                "stats": self.signal_data.daily_stats
            })

        @self.app.get("/api/health")
        async def health_check():
            """Simple health check"""
            return JSONResponse({
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "is_monitoring": self.is_running,
                "last_signal_time": self.last_signal_time.isoformat() if self.last_signal_time else None
            })

        @self.app.websocket("/ws/signals")
        async def websocket_signals(websocket: WebSocket):
            """WebSocket endpoint for real-time signal updates"""
            await websocket.accept()
            self.websocket_connections.append(websocket)

            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)

    async def initialize_trading_components(self):
        """Initialize core trading components"""
        try:
            logger.info("🔧 Initializing lightweight trading components...")

            # Create optimized configuration for 24/7 operation
            config = SignalServiceConfig(
                data_refresh_interval=300,  # 5 minutes
                signal_generation_interval=30,  # 30 seconds for web
                lookback_days=7,  # Minimal data for speed
                feature_count=15,  # Reduced features
                entry_threshold=0.65,  # Higher threshold for quality
                confidence_threshold=0.6,
                max_memory_mb=300  # Limit memory usage
            )

            # Initialize Lightweight Signal Service
            self.signal_service = LightweightSignalService(config)
            logger.info("✅ Lightweight Signal Service initialized")

            logger.info("🎯 All trading components initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize trading components: {e}")
            raise

    async def start_monitoring(self):
        """Start 24/7 signal monitoring"""
        if self.is_running:
            logger.warning("Monitoring already running")
            return

        self.is_running = True
        logger.info("🚀 Starting 24/7 signal monitoring...")

        # Start lightweight signal service
        if self.signal_service:
            await self.signal_service.start()
            logger.info("✅ Lightweight signal service started")

        # Start web interface monitoring task
        asyncio.create_task(self._web_monitoring_loop())

    async def _web_monitoring_loop(self):
        """Web interface monitoring loop - polls lightweight service for updates"""
        logger.info("🌐 Web monitoring loop started")

        while self.is_running:
            try:
                # Get signals from lightweight service
                if self.signal_service and self.signal_service.is_running:
                    await self._sync_with_signal_service()

                # Broadcast updates to all connected clients
                await self._broadcast_signal_updates()

                # Wait for next update (faster updates for web interface)
                await asyncio.sleep(5)  # 5-second updates for web

            except Exception as e:
                logger.error(f"Error in web monitoring loop: {e}")
                await asyncio.sleep(5)  # Error recovery delay

    async def _sync_with_signal_service(self):
        """Sync data from lightweight signal service"""
        try:
            # Get current signals from the service
            signals = self.signal_service.get_current_signals()
            positions = self.signal_service.get_open_positions()
            stats = self.signal_service.get_performance_stats()

            # Update local signal data
            for symbol, signal_data in signals.items():
                self.signal_data.current_signals[symbol] = {
                    'action': signal_data['action'],
                    'confidence': signal_data['confidence'],
                    'strength': signal_data['strength'],
                    'timestamp': signal_data['timestamp'],
                    'source': 'Lightweight AI Service'
                }

            # Update positions
            self.signal_data.open_positions = positions

            # Update daily stats
            self.signal_data.daily_stats.update({
                'total_signals': stats.get('signals_generated', 0),
                'entry_signals': len([p for p in positions.values() if 'entry_time' in p]),
                'exit_signals': self.signal_data.daily_stats.get('exit_signals', 0),
                'successful_trades': self.signal_data.daily_stats.get('successful_trades', 0)
            })

            self.last_signal_time = datetime.now()

        except Exception as e:
            logger.error(f"Error syncing with signal service: {e}")

    # Legacy methods removed - now using lightweight signal service
    # The lightweight service handles all signal generation, market data fetching,
    # and signal processing internally for better performance and 24/7 operation

    async def _broadcast_signal_updates(self):
        """Broadcast signal updates to all connected WebSocket clients"""
        if not self.websocket_connections:
            return

        try:
            message = {
                "type": "signal_update",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "signals": self.signal_data.current_signals,
                    "positions": self.signal_data.open_positions,
                    "stats": self.signal_data.daily_stats
                }
            }

            # Send to all connected clients
            disconnected = []
            for websocket in self.websocket_connections:
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception:
                    disconnected.append(websocket)

            # Remove disconnected clients
            for ws in disconnected:
                self.websocket_connections.remove(ws)

        except Exception as e:
            logger.error(f"Error broadcasting signal updates: {e}")

    def get_app(self) -> FastAPI:
        """Get FastAPI application instance"""
        return self.app


# Global app instance
trading_web_app = SimplifiedTradingWebApp()


async def startup_event():
    """Startup event handler"""
    logger.info("🚀 Starting AIFX Simplified Trading Web Interface...")

    try:
        # Initialize trading components
        await trading_web_app.initialize_trading_components()

        # Start monitoring
        await trading_web_app.start_monitoring()

        logger.info("✅ AIFX Trading Web Interface started successfully")

    except Exception as e:
        logger.error(f"❌ Failed to start trading web interface: {e}")
        raise


# Add startup event
trading_web_app.app.add_event_handler("startup", startup_event)

# Get app instance for uvicorn
app = trading_web_app.get_app()


def main():
    """Main entry point for running the simplified web interface"""
    logger.info("🌐 Starting AIFX Simplified Trading Signals Web Server...")

    config = {
        "app": "src.main.python.web_interface:app",
        "host": "0.0.0.0",
        "port": 8080,
        "reload": False,  # Disable reload for 24/7 operation
        "log_level": "info",
    }

    logger.info(f"📍 Server: http://{config['host']}:{config['port']}")
    logger.info("🔄 24/7 Continuous Operation Mode")

    try:
        uvicorn.run(**config)
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")


if __name__ == "__main__":
    main()