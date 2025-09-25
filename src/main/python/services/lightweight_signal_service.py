"""
AIFX - Lightweight Signal Generation Service
AIFX - 輕量級信號生成服務

Optimized 24/7 service for generating trading signals with minimal resource usage.
優化的24小時服務，使用最少資源生成交易信號。
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

# Core components
from utils.data_loader import DataLoader
from utils.feature_generator import FeatureGenerator
from utils.mock_data_generator import create_mock_data_for_service
from utils.performance import profile_performance, cached, performance_context
from core.signal_combiner import TradingSignal, SignalType
from core.signal_detector import EntrySignalDetector, ExitSignalDetector, DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class LightweightSignal:
    """Lightweight signal data structure"""
    symbol: str
    action: str  # 'ENTER_LONG', 'ENTER_SHORT', 'EXIT', 'HOLD'
    confidence: float  # 0.0 to 1.0
    strength: float    # 0.0 to 1.0
    timestamp: datetime
    reasoning: str = ""


@dataclass
class SignalServiceConfig:
    """Configuration for lightweight signal service"""
    # Update intervals
    data_refresh_interval: int = 300  # 5 minutes
    signal_generation_interval: int = 60  # 1 minute

    # Data settings
    lookback_days: int = 7  # Only 7 days of data for speed
    feature_count: int = 20  # Reduced feature set

    # Signal thresholds
    entry_threshold: float = 0.65  # Higher threshold for cleaner signals
    exit_threshold: float = 0.4    # Lower threshold for quicker exits
    confidence_threshold: float = 0.6

    # Performance settings
    max_memory_mb: int = 500  # Memory limit
    max_processing_time: int = 30  # Max processing time per cycle


class LightweightSignalService:
    """
    Lightweight 24/7 Signal Generation Service
    輕量級24小時信號生成服務

    Features:
    - Minimal resource usage
    - Optimized for speed
    - 24/7 continuous operation
    - Essential signals only
    """

    def __init__(self, config: Optional[SignalServiceConfig] = None):
        self.config = config or SignalServiceConfig()

        # Core components
        self.data_loader = DataLoader()
        self.feature_generator = self._create_lightweight_feature_generator()

        # Signal detection system
        self.entry_detector = EntrySignalDetector()
        self.exit_detector = ExitSignalDetector()

        # Service state
        self.is_running = False
        self.last_data_refresh = None
        self.cached_data = {}
        self.current_signals = {}
        self.open_positions = {}

        # Performance monitoring
        self.performance_stats = {
            'signals_generated': 0,
            'processing_time_avg': 0.0,
            'memory_usage_mb': 0.0,
            'uptime_seconds': 0.0,
            'start_time': None
        }

        # Lightweight technical indicators cache
        self.indicators_cache = {}

        logger.info("Lightweight Signal Service initialized")

    def _create_lightweight_feature_generator(self) -> FeatureGenerator:
        """Create optimized feature generator with reduced feature set"""
        fg = FeatureGenerator()

        # Override with lightweight configuration
        fg.config = {
            'price_features': True,
            'volume_features': False,  # Disable volume for speed
            'technical_indicators': True,
            'volatility_features': True,
            'momentum_features': True,
            'statistical_features': False,  # Disable complex stats
            'advanced_features': False,     # Disable advanced features
        }

        return fg

    async def start(self):
        """Start the lightweight signal service"""
        if self.is_running:
            logger.warning("Signal service already running")
            return

        self.is_running = True
        self.performance_stats['start_time'] = datetime.now()

        logger.info("🚀 Starting Lightweight Signal Service...")

        # Initialize data cache
        await self._initialize_data_cache()

        # Start service loops
        asyncio.create_task(self._data_refresh_loop())
        asyncio.create_task(self._signal_generation_loop())
        asyncio.create_task(self._performance_monitoring_loop())

        logger.info("✅ Lightweight Signal Service started")

    async def stop(self):
        """Stop the signal service"""
        self.is_running = False
        logger.info("🛑 Lightweight Signal Service stopped")

    async def _initialize_data_cache(self):
        """Initialize data cache with recent market data"""
        logger.info("📊 Initializing data cache...")

        try:
            symbols = ['EURUSD=X', 'USDJPY=X']

            for symbol in symbols:
                await self._refresh_symbol_data(symbol)

            logger.info(f"✅ Data cache initialized for {len(symbols)} symbols")

        except Exception as e:
            logger.error(f"❌ Failed to initialize data cache: {e}")

    async def _data_refresh_loop(self):
        """Background loop for refreshing market data"""
        logger.info("📈 Data refresh loop started")

        while self.is_running:
            try:
                start_time = time.time()

                # Refresh data for all symbols
                symbols = ['EURUSD=X', 'USDJPY=X']

                for symbol in symbols:
                    await self._refresh_symbol_data(symbol)

                processing_time = time.time() - start_time
                logger.debug(f"Data refresh completed in {processing_time:.2f}s")

                self.last_data_refresh = datetime.now()

                # Wait for next refresh
                await asyncio.sleep(self.config.data_refresh_interval)

            except Exception as e:
                logger.error(f"Error in data refresh loop: {e}")
                await asyncio.sleep(30)  # Error recovery delay

    @profile_performance
    async def _refresh_symbol_data(self, symbol: str):
        """Refresh data for a specific symbol"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.lookback_days)

            # Load data using correct method
            data_dict = self.data_loader.download_data(
                symbols=symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval='1h'
            )

            # Extract data for this symbol
            data = data_dict.get(symbol, pd.DataFrame())

            if not data.empty:
                self.cached_data[symbol] = data
                logger.info(f"✅ Refreshed real data for {symbol}: {len(data)} records")
            else:
                # Use mock data as fallback
                logger.warning(f"⚠️ Real data failed for {symbol}, using mock data")
                mock_data = create_mock_data_for_service([symbol], days=self.config.lookback_days)
                if symbol in mock_data and not mock_data[symbol].empty:
                    self.cached_data[symbol] = mock_data[symbol]
                    logger.info(f"🎭 Using mock data for {symbol}: {len(mock_data[symbol])} records")

        except Exception as e:
            logger.error(f"❌ Error refreshing data for {symbol}: {e}")
            # Try mock data as last resort
            try:
                logger.info(f"🎭 Attempting mock data for {symbol} after error")
                mock_data = create_mock_data_for_service([symbol], days=self.config.lookback_days)
                if symbol in mock_data and not mock_data[symbol].empty:
                    self.cached_data[symbol] = mock_data[symbol]
                    logger.info(f"✅ Emergency mock data loaded for {symbol}: {len(mock_data[symbol])} records")
            except Exception as mock_error:
                logger.error(f"❌ Even mock data failed for {symbol}: {mock_error}")

    async def _signal_generation_loop(self):
        """Main loop for generating trading signals"""
        logger.info("🎯 Signal generation loop started")

        while self.is_running:
            try:
                start_time = time.time()

                # Generate signals for all symbols
                await self._generate_all_signals()

                processing_time = time.time() - start_time
                self._update_performance_stats(processing_time)

                logger.debug(f"Signal generation completed in {processing_time:.2f}s")

                # Wait for next generation cycle
                await asyncio.sleep(self.config.signal_generation_interval)

            except Exception as e:
                logger.error(f"Error in signal generation loop: {e}")
                await asyncio.sleep(10)  # Error recovery delay

    async def _generate_all_signals(self):
        """Generate signals for all tracked symbols"""
        symbols = ['EURUSD', 'USDJPY']  # Display symbols
        yahoo_symbols = ['EURUSD=X', 'USDJPY=X']  # Yahoo Finance symbols

        for i, symbol in enumerate(symbols):
            yahoo_symbol = yahoo_symbols[i]

            if yahoo_symbol not in self.cached_data:
                logger.warning(f"No cached data available for {yahoo_symbol}")
                continue

            try:
                signal = await self._generate_symbol_signal(symbol, yahoo_symbol)
                if signal:
                    self.current_signals[symbol] = signal
                    logger.debug(f"Generated signal for {symbol}: {signal.action}")

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")

    @profile_performance
    @cached(ttl=30)  # Cache signals for 30 seconds
    async def _generate_symbol_signal(self, symbol: str, yahoo_symbol: str) -> Optional[LightweightSignal]:
        """Generate trading signal for a specific symbol"""
        try:
            # Get cached data
            data = self.cached_data[yahoo_symbol]

            if data.empty or len(data) < 50:  # Need minimum data
                logger.warning(f"Insufficient data for {symbol}")
                return None

            # Generate lightweight features
            features = await self._generate_lightweight_features(data)

            if features.empty:
                logger.warning(f"No features generated for {symbol}")
                return None

            # Generate signal using lightweight algorithm
            signal = self._calculate_lightweight_signal(symbol, features, data)

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None

    @profile_performance
    @cached(ttl=60)  # Cache features for 1 minute
    async def _generate_lightweight_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate optimized feature set"""
        try:
            # Use only recent data for speed
            recent_data = data.tail(100) if len(data) > 100 else data

            # Generate essential features only
            features = self._calculate_essential_features(recent_data)

            return features

        except Exception as e:
            logger.error(f"Error generating lightweight features: {e}")
            return pd.DataFrame()

    def _calculate_essential_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate only essential features for signal generation"""
        try:
            features = pd.DataFrame(index=data.index)

            # Price-based features
            features['price_change'] = data['Close'].pct_change()
            features['price_sma_5'] = data['Close'].rolling(5).mean()
            features['price_sma_20'] = data['Close'].rolling(20).mean()

            # Technical indicators (simplified)
            features['rsi'] = self._calculate_rsi(data['Close'], 14)
            features['macd'] = self._calculate_macd(data['Close'])
            features['bb_upper'], features['bb_lower'] = self._calculate_bollinger_bands(data['Close'])

            # Momentum features
            features['momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
            features['momentum_10'] = data['Close'] / data['Close'].shift(10) - 1

            # Volatility features
            features['volatility'] = data['Close'].pct_change().rolling(20).std()
            features['atr'] = self._calculate_atr(data)

            # Signal strength
            features['volume_sma'] = data['Volume'].rolling(20).mean() if 'Volume' in data.columns else 0

            # Remove NaN values
            features = features.fillna(method='bfill').fillna(0)

            return features.tail(1)  # Return only latest features

        except Exception as e:
            logger.error(f"Error calculating essential features: {e}")
            return pd.DataFrame()

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        return macd

    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    def _calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window).mean()
        return atr

    def _calculate_lightweight_signal(self, symbol: str, features: pd.DataFrame, price_data: pd.DataFrame) -> LightweightSignal:
        """Calculate trading signal using specialized entry/exit detectors"""
        try:
            if features.empty:
                return LightweightSignal(symbol, 'HOLD', 0.0, 0.0, datetime.now(), "No features")

            current_position = self.open_positions.get(symbol)

            # Check for exit signals first if position is open
            if current_position:
                exit_signal = self.exit_detector.detect_exit_signals(
                    symbol, current_position, price_data, features
                )

                if exit_signal:
                    action = 'EXIT'
                    if symbol in self.open_positions:
                        del self.open_positions[symbol]

                    return LightweightSignal(
                        symbol=symbol,
                        action=action,
                        confidence=exit_signal.confidence,
                        strength=exit_signal.confidence,  # Use confidence as strength for exits
                        timestamp=datetime.now(),
                        reasoning=exit_signal.reasoning
                    )
                else:
                    # Hold current position
                    position_type = current_position.get('type', 'UNKNOWN')
                    action = f'HOLD_{position_type}'

                    return LightweightSignal(
                        symbol=symbol,
                        action=action,
                        confidence=0.6,  # Moderate confidence for holding
                        strength=0.6,
                        timestamp=datetime.now(),
                        reasoning=f"Holding {position_type} position"
                    )

            # Look for entry signals if no position is open
            else:
                entry_signals = self.entry_detector.detect_entry_signals(price_data, features)

                if entry_signals:
                    # Use the highest confidence entry signal
                    best_signal = entry_signals[0]  # Already sorted by confidence

                    # Only act on high-confidence signals
                    if best_signal.confidence >= self.config.confidence_threshold:
                        if best_signal.signal_type.value in ['ENTER_LONG']:
                            action = 'ENTER_LONG'
                            self.open_positions[symbol] = {
                                'type': 'LONG',
                                'entry_time': datetime.now(),
                                'entry_price': price_data['Close'].iloc[-1]
                            }
                        elif best_signal.signal_type.value in ['ENTER_SHORT']:
                            action = 'ENTER_SHORT'
                            self.open_positions[symbol] = {
                                'type': 'SHORT',
                                'entry_time': datetime.now(),
                                'entry_price': price_data['Close'].iloc[-1]
                            }
                        else:
                            action = 'HOLD'

                        return LightweightSignal(
                            symbol=symbol,
                            action=action,
                            confidence=best_signal.confidence,
                            strength=best_signal.confidence,
                            timestamp=datetime.now(),
                            reasoning=best_signal.reasoning
                        )

                # No strong entry signals found
                return LightweightSignal(
                    symbol=symbol,
                    action='HOLD',
                    confidence=0.5,
                    strength=0.5,
                    timestamp=datetime.now(),
                    reasoning="No clear entry signal detected"
                )

        except Exception as e:
            logger.error(f"Error calculating signal for {symbol}: {e}")
            return LightweightSignal(symbol, 'HOLD', 0.0, 0.0, datetime.now(), f"Error: {str(e)}")

    async def _performance_monitoring_loop(self):
        """Monitor service performance and resource usage"""
        while self.is_running:
            try:
                # Update performance statistics
                if self.performance_stats['start_time']:
                    self.performance_stats['uptime_seconds'] = (
                        datetime.now() - self.performance_stats['start_time']
                    ).total_seconds()

                # Monitor memory usage (simplified)
                import psutil
                process = psutil.Process()
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                self.performance_stats['memory_usage_mb'] = memory_usage

                # Log performance stats every 5 minutes
                if int(time.time()) % 300 == 0:
                    logger.info(f"📊 Performance: {self.performance_stats['signals_generated']} signals, "
                              f"{memory_usage:.1f}MB memory, "
                              f"{self.performance_stats['uptime_seconds']/3600:.1f}h uptime")

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                await asyncio.sleep(60)

    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        self.performance_stats['signals_generated'] += 1

        # Update average processing time
        current_avg = self.performance_stats['processing_time_avg']
        count = self.performance_stats['signals_generated']
        self.performance_stats['processing_time_avg'] = (
            (current_avg * (count - 1) + processing_time) / count
        )

    def get_current_signals(self) -> Dict[str, Dict[str, Any]]:
        """Get current trading signals"""
        signals = {}

        for symbol, signal in self.current_signals.items():
            signals[symbol] = {
                'action': signal.action,
                'confidence': round(signal.confidence * 100, 1),
                'strength': signal.strength,
                'timestamp': signal.timestamp.isoformat(),
                'reasoning': signal.reasoning
            }

        return signals

    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current open positions"""
        return self.open_positions.copy()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get service performance statistics"""
        return self.performance_stats.copy()

    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        return {
            'is_running': self.is_running,
            'last_data_refresh': self.last_data_refresh.isoformat() if self.last_data_refresh else None,
            'cached_symbols': list(self.cached_data.keys()),
            'current_signals_count': len(self.current_signals),
            'open_positions_count': len(self.open_positions),
            'performance_stats': self.get_performance_stats(),
            'config': {
                'data_refresh_interval': self.config.data_refresh_interval,
                'signal_generation_interval': self.config.signal_generation_interval,
                'entry_threshold': self.config.entry_threshold,
                'confidence_threshold': self.config.confidence_threshold
            }
        }


# Factory function
def create_lightweight_signal_service(config: Optional[SignalServiceConfig] = None) -> LightweightSignalService:
    """Create a lightweight signal service instance"""
    return LightweightSignalService(config)