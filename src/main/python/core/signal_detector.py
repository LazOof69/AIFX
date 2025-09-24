"""
AIFX - Entry/Exit Signal Detection System
AIFX - 入場/出場信號檢測系統

Specialized system for detecting precise entry and exit signals for 24/7 web operation.
用於24小時網頁運作的精確入場和出場信號檢測專業系統。
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, NamedTuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Signal types for trading actions"""
    ENTER_LONG = "ENTER_LONG"
    ENTER_SHORT = "ENTER_SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    HOLD = "HOLD"


class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4


@dataclass
class DetectionResult:
    """Result of signal detection"""
    signal_type: SignalType
    strength: SignalStrength
    confidence: float
    price_target: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class TechnicalPattern(NamedTuple):
    """Technical pattern detection result"""
    pattern_name: str
    signal_type: SignalType
    confidence: float
    reasoning: str


class EntrySignalDetector:
    """
    Entry Signal Detection System
    入場信號檢測系統

    Detects high-probability entry points using multiple technical indicators.
    使用多個技術指標檢測高機率入場點。
    """

    def __init__(self):
        self.name = "Entry Signal Detector"
        self.version = "1.0"

        # Detection thresholds
        self.thresholds = {
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_signal_threshold': 0.0001,
            'bb_squeeze_threshold': 0.02,
            'momentum_threshold': 0.015,
            'volume_spike_threshold': 1.5
        }

        # Pattern detection parameters
        self.pattern_lookback = 20
        self.confirmation_periods = 3

        logger.info("Entry Signal Detector initialized")

    def detect_entry_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> List[DetectionResult]:
        """
        Detect entry signals from market data and features
        從市場數據和特徵檢測入場信號

        Args:
            data: OHLCV market data
            features: Technical features and indicators

        Returns:
            List of detected entry signals
        """
        try:
            if data.empty or features.empty:
                return []

            signals = []

            # RSI-based entry signals
            rsi_signals = self._detect_rsi_entry_signals(features)
            signals.extend(rsi_signals)

            # MACD-based entry signals
            macd_signals = self._detect_macd_entry_signals(features)
            signals.extend(macd_signals)

            # Bollinger Bands breakout signals
            bb_signals = self._detect_bollinger_breakout_signals(data, features)
            signals.extend(bb_signals)

            # Momentum-based signals
            momentum_signals = self._detect_momentum_entry_signals(features)
            signals.extend(momentum_signals)

            # Pattern-based signals
            pattern_signals = self._detect_technical_patterns(data)
            signals.extend(pattern_signals)

            # Filter and rank signals
            filtered_signals = self._filter_and_rank_signals(signals)

            logger.debug(f"Detected {len(filtered_signals)} entry signals from {len(signals)} candidates")
            return filtered_signals

        except Exception as e:
            logger.error(f"Error detecting entry signals: {e}")
            return []

    def _detect_rsi_entry_signals(self, features: pd.DataFrame) -> List[DetectionResult]:
        """Detect RSI-based entry signals"""
        signals = []

        try:
            if 'rsi' not in features.columns:
                return signals

            latest_rsi = features['rsi'].iloc[-1]
            prev_rsi = features['rsi'].iloc[-2] if len(features) > 1 else latest_rsi

            # RSI oversold reversal (potential long entry)
            if (prev_rsi <= self.thresholds['rsi_oversold'] and
                latest_rsi > self.thresholds['rsi_oversold']):

                confidence = min(0.8, (self.thresholds['rsi_oversold'] - prev_rsi) / 10)

                signals.append(DetectionResult(
                    signal_type=SignalType.ENTER_LONG,
                    strength=SignalStrength.MODERATE,
                    confidence=confidence,
                    reasoning=f"RSI oversold reversal: {prev_rsi:.1f} → {latest_rsi:.1f}"
                ))

            # RSI overbought reversal (potential short entry)
            elif (prev_rsi >= self.thresholds['rsi_overbought'] and
                  latest_rsi < self.thresholds['rsi_overbought']):

                confidence = min(0.8, (prev_rsi - self.thresholds['rsi_overbought']) / 10)

                signals.append(DetectionResult(
                    signal_type=SignalType.ENTER_SHORT,
                    strength=SignalStrength.MODERATE,
                    confidence=confidence,
                    reasoning=f"RSI overbought reversal: {prev_rsi:.1f} → {latest_rsi:.1f}"
                ))

        except Exception as e:
            logger.error(f"Error in RSI signal detection: {e}")

        return signals

    def _detect_macd_entry_signals(self, features: pd.DataFrame) -> List[DetectionResult]:
        """Detect MACD-based entry signals"""
        signals = []

        try:
            if 'macd' not in features.columns:
                return signals

            macd_values = features['macd'].tail(5)
            if len(macd_values) < 3:
                return signals

            latest_macd = macd_values.iloc[-1]
            prev_macd = macd_values.iloc[-2]
            prev_2_macd = macd_values.iloc[-3]

            # MACD bullish crossover
            if (prev_2_macd < 0 and prev_macd < 0 and
                latest_macd > 0 and latest_macd > self.thresholds['macd_signal_threshold']):

                confidence = min(0.75, abs(latest_macd) * 10000)  # Scale MACD value

                signals.append(DetectionResult(
                    signal_type=SignalType.ENTER_LONG,
                    strength=SignalStrength.STRONG,
                    confidence=confidence,
                    reasoning=f"MACD bullish crossover: {latest_macd:.4f}"
                ))

            # MACD bearish crossover
            elif (prev_2_macd > 0 and prev_macd > 0 and
                  latest_macd < 0 and abs(latest_macd) > self.thresholds['macd_signal_threshold']):

                confidence = min(0.75, abs(latest_macd) * 10000)

                signals.append(DetectionResult(
                    signal_type=SignalType.ENTER_SHORT,
                    strength=SignalStrength.STRONG,
                    confidence=confidence,
                    reasoning=f"MACD bearish crossover: {latest_macd:.4f}"
                ))

        except Exception as e:
            logger.error(f"Error in MACD signal detection: {e}")

        return signals

    def _detect_bollinger_breakout_signals(self, data: pd.DataFrame, features: pd.DataFrame) -> List[DetectionResult]:
        """Detect Bollinger Bands breakout signals"""
        signals = []

        try:
            required_cols = ['bb_upper', 'bb_lower']
            if not all(col in features.columns for col in required_cols):
                return signals

            latest_price = data['Close'].iloc[-1]
            bb_upper = features['bb_upper'].iloc[-1]
            bb_lower = features['bb_lower'].iloc[-1]

            # Bullish breakout above upper band
            if latest_price > bb_upper:
                breakout_strength = (latest_price - bb_upper) / bb_upper
                confidence = min(0.7, breakout_strength * 100)

                signals.append(DetectionResult(
                    signal_type=SignalType.ENTER_LONG,
                    strength=SignalStrength.MODERATE,
                    confidence=confidence,
                    reasoning=f"Bollinger breakout above upper band: {breakout_strength:.3f}%"
                ))

            # Bearish breakout below lower band
            elif latest_price < bb_lower:
                breakout_strength = (bb_lower - latest_price) / bb_lower
                confidence = min(0.7, breakout_strength * 100)

                signals.append(DetectionResult(
                    signal_type=SignalType.ENTER_SHORT,
                    strength=SignalStrength.MODERATE,
                    confidence=confidence,
                    reasoning=f"Bollinger breakout below lower band: {breakout_strength:.3f}%"
                ))

        except Exception as e:
            logger.error(f"Error in Bollinger Bands signal detection: {e}")

        return signals

    def _detect_momentum_entry_signals(self, features: pd.DataFrame) -> List[DetectionResult]:
        """Detect momentum-based entry signals"""
        signals = []

        try:
            momentum_cols = [col for col in features.columns if 'momentum' in col]
            if not momentum_cols:
                return signals

            # Use the strongest momentum indicator available
            momentum_col = momentum_cols[0]
            momentum_values = features[momentum_col].tail(3)

            if len(momentum_values) < 2:
                return signals

            latest_momentum = momentum_values.iloc[-1]
            prev_momentum = momentum_values.iloc[-2]

            # Strong positive momentum acceleration
            if (latest_momentum > self.thresholds['momentum_threshold'] and
                latest_momentum > prev_momentum):

                confidence = min(0.8, abs(latest_momentum) * 50)

                signals.append(DetectionResult(
                    signal_type=SignalType.ENTER_LONG,
                    strength=SignalStrength.STRONG,
                    confidence=confidence,
                    reasoning=f"Strong positive momentum: {latest_momentum:.3f}"
                ))

            # Strong negative momentum acceleration
            elif (latest_momentum < -self.thresholds['momentum_threshold'] and
                  latest_momentum < prev_momentum):

                confidence = min(0.8, abs(latest_momentum) * 50)

                signals.append(DetectionResult(
                    signal_type=SignalType.ENTER_SHORT,
                    strength=SignalStrength.STRONG,
                    confidence=confidence,
                    reasoning=f"Strong negative momentum: {latest_momentum:.3f}"
                ))

        except Exception as e:
            logger.error(f"Error in momentum signal detection: {e}")

        return signals

    def _detect_technical_patterns(self, data: pd.DataFrame) -> List[DetectionResult]:
        """Detect technical chart patterns"""
        signals = []

        try:
            if len(data) < self.pattern_lookback:
                return signals

            recent_data = data.tail(self.pattern_lookback)

            # Support/Resistance breakout detection
            pattern_signals = self._detect_support_resistance_breakout(recent_data)
            signals.extend(pattern_signals)

            # Price consolidation breakout
            consolidation_signals = self._detect_consolidation_breakout(recent_data)
            signals.extend(consolidation_signals)

        except Exception as e:
            logger.error(f"Error in pattern detection: {e}")

        return signals

    def _detect_support_resistance_breakout(self, data: pd.DataFrame) -> List[DetectionResult]:
        """Detect support/resistance level breakouts"""
        signals = []

        try:
            # Calculate recent highs and lows
            recent_high = data['High'].max()
            recent_low = data['Low'].min()
            latest_price = data['Close'].iloc[-1]

            # Resistance breakout
            if latest_price > recent_high * 1.001:  # 0.1% buffer
                confidence = min(0.7, (latest_price - recent_high) / recent_high * 100)

                signals.append(DetectionResult(
                    signal_type=SignalType.ENTER_LONG,
                    strength=SignalStrength.MODERATE,
                    confidence=confidence,
                    reasoning=f"Resistance breakout at {recent_high:.5f}"
                ))

            # Support breakdown
            elif latest_price < recent_low * 0.999:  # 0.1% buffer
                confidence = min(0.7, (recent_low - latest_price) / recent_low * 100)

                signals.append(DetectionResult(
                    signal_type=SignalType.ENTER_SHORT,
                    strength=SignalStrength.MODERATE,
                    confidence=confidence,
                    reasoning=f"Support breakdown at {recent_low:.5f}"
                ))

        except Exception as e:
            logger.error(f"Error in support/resistance detection: {e}")

        return signals

    def _detect_consolidation_breakout(self, data: pd.DataFrame) -> List[DetectionResult]:
        """Detect breakout from price consolidation"""
        signals = []

        try:
            # Calculate price range and volatility
            price_range = data['High'].max() - data['Low'].min()
            avg_price = data['Close'].mean()
            relative_range = price_range / avg_price

            # Check for tight consolidation (low volatility)
            if relative_range < 0.02:  # 2% range
                latest_price = data['Close'].iloc[-1]
                consolidation_high = data['High'].max()
                consolidation_low = data['Low'].min()

                # Upward breakout
                if latest_price > consolidation_high:
                    confidence = 0.65

                    signals.append(DetectionResult(
                        signal_type=SignalType.ENTER_LONG,
                        strength=SignalStrength.MODERATE,
                        confidence=confidence,
                        reasoning=f"Consolidation breakout upward: {relative_range:.1%} range"
                    ))

                # Downward breakout
                elif latest_price < consolidation_low:
                    confidence = 0.65

                    signals.append(DetectionResult(
                        signal_type=SignalType.ENTER_SHORT,
                        strength=SignalStrength.MODERATE,
                        confidence=confidence,
                        reasoning=f"Consolidation breakdown: {relative_range:.1%} range"
                    ))

        except Exception as e:
            logger.error(f"Error in consolidation breakout detection: {e}")

        return signals

    def _filter_and_rank_signals(self, signals: List[DetectionResult]) -> List[DetectionResult]:
        """Filter and rank signals by strength and confidence"""
        try:
            # Remove duplicate signal types
            signal_by_type = {}
            for signal in signals:
                if signal.signal_type not in signal_by_type:
                    signal_by_type[signal.signal_type] = signal
                else:
                    # Keep the signal with higher confidence
                    if signal.confidence > signal_by_type[signal.signal_type].confidence:
                        signal_by_type[signal.signal_type] = signal

            # Sort by confidence and strength
            filtered_signals = list(signal_by_type.values())
            filtered_signals.sort(
                key=lambda x: (x.confidence, x.strength.value),
                reverse=True
            )

            # Return top signals only
            return filtered_signals[:2]  # Max 2 signals

        except Exception as e:
            logger.error(f"Error filtering signals: {e}")
            return signals


class ExitSignalDetector:
    """
    Exit Signal Detection System
    出場信號檢測系統

    Detects optimal exit points to protect profits and minimize losses.
    檢測最佳出場點以保護利潤並最小化損失。
    """

    def __init__(self):
        self.name = "Exit Signal Detector"
        self.version = "1.0"

        # Exit detection parameters
        self.exit_thresholds = {
            'profit_target_ratio': 2.0,  # Risk-reward ratio
            'trailing_stop_ratio': 0.5,  # Trailing stop percentage
            'max_loss_ratio': -0.02,     # Maximum loss threshold
            'rsi_extreme_exit': 80,      # RSI exit threshold
            'momentum_reversal': 0.01    # Momentum reversal threshold
        }

        logger.info("Exit Signal Detector initialized")

    def detect_exit_signals(self, symbol: str, position_info: Dict,
                          data: pd.DataFrame, features: pd.DataFrame) -> Optional[DetectionResult]:
        """
        Detect exit signals for an open position
        檢測開倉位置的出場信號

        Args:
            symbol: Trading symbol
            position_info: Open position information
            data: Current market data
            features: Technical features

        Returns:
            Exit signal if detected, None otherwise
        """
        try:
            if not position_info or data.empty:
                return None

            position_type = position_info.get('type')
            entry_time = position_info.get('entry_time')
            entry_price = position_info.get('entry_price', data['Close'].iloc[-10])  # Fallback

            current_price = data['Close'].iloc[-1]

            # Calculate position P&L
            if position_type == 'LONG':
                pnl_ratio = (current_price - entry_price) / entry_price
            else:  # SHORT
                pnl_ratio = (entry_price - current_price) / entry_price

            # Profit target exit
            profit_exit = self._check_profit_target_exit(pnl_ratio, position_type)
            if profit_exit:
                return profit_exit

            # Stop loss exit
            stop_loss_exit = self._check_stop_loss_exit(pnl_ratio, position_type)
            if stop_loss_exit:
                return stop_loss_exit

            # Technical reversal exit
            technical_exit = self._check_technical_reversal_exit(position_type, features, data)
            if technical_exit:
                return technical_exit

            # Time-based exit (if position held too long)
            time_exit = self._check_time_based_exit(entry_time, position_type)
            if time_exit:
                return time_exit

            return None

        except Exception as e:
            logger.error(f"Error detecting exit signals for {symbol}: {e}")
            return None

    def _check_profit_target_exit(self, pnl_ratio: float, position_type: str) -> Optional[DetectionResult]:
        """Check if profit target is reached"""
        try:
            if pnl_ratio > 0.02:  # 2% profit
                confidence = min(0.9, pnl_ratio * 25)  # Higher confidence for larger profits

                return DetectionResult(
                    signal_type=SignalType.EXIT_LONG if position_type == 'LONG' else SignalType.EXIT_SHORT,
                    strength=SignalStrength.STRONG,
                    confidence=confidence,
                    reasoning=f"Profit target reached: {pnl_ratio:.2%}"
                )

        except Exception as e:
            logger.error(f"Error in profit target check: {e}")

        return None

    def _check_stop_loss_exit(self, pnl_ratio: float, position_type: str) -> Optional[DetectionResult]:
        """Check if stop loss is triggered"""
        try:
            if pnl_ratio < self.exit_thresholds['max_loss_ratio']:
                confidence = 0.95  # High confidence for stop loss

                return DetectionResult(
                    signal_type=SignalType.EXIT_LONG if position_type == 'LONG' else SignalType.EXIT_SHORT,
                    strength=SignalStrength.VERY_STRONG,
                    confidence=confidence,
                    reasoning=f"Stop loss triggered: {pnl_ratio:.2%}"
                )

        except Exception as e:
            logger.error(f"Error in stop loss check: {e}")

        return None

    def _check_technical_reversal_exit(self, position_type: str, features: pd.DataFrame,
                                     data: pd.DataFrame) -> Optional[DetectionResult]:
        """Check for technical reversal signals"""
        try:
            # RSI-based exit
            if 'rsi' in features.columns:
                latest_rsi = features['rsi'].iloc[-1]

                if position_type == 'LONG' and latest_rsi > self.exit_thresholds['rsi_extreme_exit']:
                    return DetectionResult(
                        signal_type=SignalType.EXIT_LONG,
                        strength=SignalStrength.MODERATE,
                        confidence=0.7,
                        reasoning=f"RSI overbought exit: {latest_rsi:.1f}"
                    )
                elif position_type == 'SHORT' and latest_rsi < (100 - self.exit_thresholds['rsi_extreme_exit']):
                    return DetectionResult(
                        signal_type=SignalType.EXIT_SHORT,
                        strength=SignalStrength.MODERATE,
                        confidence=0.7,
                        reasoning=f"RSI oversold exit: {latest_rsi:.1f}"
                    )

            # Momentum reversal exit
            momentum_cols = [col for col in features.columns if 'momentum' in col]
            if momentum_cols:
                momentum = features[momentum_cols[0]].iloc[-1]

                if (position_type == 'LONG' and momentum < -self.exit_thresholds['momentum_reversal']):
                    return DetectionResult(
                        signal_type=SignalType.EXIT_LONG,
                        strength=SignalStrength.MODERATE,
                        confidence=0.65,
                        reasoning=f"Momentum reversal: {momentum:.3f}"
                    )
                elif (position_type == 'SHORT' and momentum > self.exit_thresholds['momentum_reversal']):
                    return DetectionResult(
                        signal_type=SignalType.EXIT_SHORT,
                        strength=SignalStrength.MODERATE,
                        confidence=0.65,
                        reasoning=f"Momentum reversal: {momentum:.3f}"
                    )

        except Exception as e:
            logger.error(f"Error in technical reversal check: {e}")

        return None

    def _check_time_based_exit(self, entry_time: datetime, position_type: str) -> Optional[DetectionResult]:
        """Check for time-based exit (if position held too long)"""
        try:
            if isinstance(entry_time, str):
                entry_time = datetime.fromisoformat(entry_time.replace('Z', '+00:00'))

            time_held = datetime.now() - entry_time
            max_hold_time = timedelta(hours=24)  # 24 hours maximum

            if time_held > max_hold_time:
                return DetectionResult(
                    signal_type=SignalType.EXIT_LONG if position_type == 'LONG' else SignalType.EXIT_SHORT,
                    strength=SignalStrength.WEAK,
                    confidence=0.6,
                    reasoning=f"Time-based exit: position held for {time_held}"
                )

        except Exception as e:
            logger.error(f"Error in time-based exit check: {e}")

        return None


# Factory functions
def create_entry_detector() -> EntrySignalDetector:
    """Create entry signal detector instance"""
    return EntrySignalDetector()


def create_exit_detector() -> ExitSignalDetector:
    """Create exit signal detector instance"""
    return ExitSignalDetector()