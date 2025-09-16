#!/usr/bin/env python3
"""
Simple USD/JPY Backtesting Script | 簡單的美元/日圓回測腳本
================================================================

A focused, simple backtesting implementation specifically for USD/JPY trading
with the AIFX system using only technical indicators (no AI models required).
專門針對美元/日圓交易的專注、簡單的回測實現，使用AIFX系統僅使用技術指標。

Usage | 使用方法:
    python simple_usdjpy_backtest.py

This script will:
1. Load USD/JPY data from the existing data file
2. Generate technical indicators and signals
3. Run a comprehensive backtest
4. Display detailed performance results
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / "src" / "main" / "python"))

try:
    from utils.data_loader import DataLoader
    from utils.data_preprocessor import DataPreprocessor
    from utils.technical_indicators import TechnicalIndicators
    from utils.logger import setup_logger
    from utils.config import Config
    from evaluation.performance_metrics import TradingPerformanceMetrics
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

class SimpleUSDJPYBacktester:
    """Simple USD/JPY Backtesting Engine | 簡單的美元/日圓回測引擎"""

    def __init__(self, initial_capital: float = 100000):
        """Initialize the backtester | 初始化回測器"""
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = []
        self.trades = []
        self.logger = setup_logger("SimpleBacktester")

        # Trading parameters | 交易參數
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.stop_loss_atr_multiplier = 2.0
        self.take_profit_atr_multiplier = 3.0
        self.min_confidence = 0.4  # Lowered to generate more trades

        print(f"🚀 Initialized Simple USD/JPY Backtester with ${initial_capital:,.2f}")

    def load_data(self, symbol: str = 'USDJPY') -> pd.DataFrame:
        """Load USD/JPY data | 載入美元/日圓數據"""
        data_file = Path("data/raw/USDJPY_1d_3y.csv")

        if not data_file.exists():
            self.logger.error(f"Data file not found: {data_file}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(data_file, index_col=0, parse_dates=True)
            self.logger.info(f"✅ Loaded {len(df)} data points from {data_file}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return pd.DataFrame()

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals using technical indicators | 使用技術指標生成交易信號"""
        try:
            # Make a copy to avoid modifying original
            df = df.copy()

            # Initialize technical indicators
            ti = TechnicalIndicators()

            # Generate basic technical indicators
            df['sma_5'] = ti.sma(df['Close'], 5)
            df['sma_10'] = ti.sma(df['Close'], 10)
            df['sma_20'] = ti.sma(df['Close'], 20)
            df['sma_50'] = ti.sma(df['Close'], 50)
            df['rsi'] = ti.rsi(df['Close'], 14)

            # MACD
            macd_result = ti.macd(df['Close'])
            df['macd'] = macd_result['macd']
            df['macd_signal'] = macd_result['signal']

            # Bollinger Bands
            bb_result = ti.bollinger_bands(df['Close'], period=20)
            df['bb_upper'] = bb_result['upper']
            df['bb_middle'] = bb_result['middle']
            df['bb_lower'] = bb_result['lower']

            # ATR
            df['atr'] = ti.atr(df['High'], df['Low'], df['Close'], 14)

            # Generate trading signals
            signals = []

            for i in range(50, len(df)):  # Start after enough data for indicators
                row = df.iloc[i]

                # Signal components
                rsi_signal = 0
                macd_signal = 0
                bb_signal = 0
                ma_signal = 0

                # RSI signals
                if row['rsi'] < 30:  # Oversold
                    rsi_signal = 1
                elif row['rsi'] > 70:  # Overbought
                    rsi_signal = -1

                # MACD signals
                if row['macd'] > row['macd_signal']:
                    macd_signal = 1
                elif row['macd'] < row['macd_signal']:
                    macd_signal = -1

                # Bollinger Bands signals
                if row['Close'] < row['bb_lower']:  # Below lower band
                    bb_signal = 1
                elif row['Close'] > row['bb_upper']:  # Above upper band
                    bb_signal = -1

                # Moving Average signals
                if row['sma_5'] > row['sma_20']:  # Short MA above long MA
                    ma_signal = 1
                elif row['sma_5'] < row['sma_20']:  # Short MA below long MA
                    ma_signal = -1

                # Combined signal
                signal_strength = (rsi_signal + macd_signal + bb_signal + ma_signal) / 4.0

                # Signal confidence based on agreement
                agreements = sum([1 for s in [rsi_signal, macd_signal, bb_signal, ma_signal] if s != 0])
                confidence = agreements / 4.0

                # Generate final signal (lowered threshold for more trades)
                if signal_strength > 0.25 and confidence >= self.min_confidence:
                    signal = 1  # BUY
                elif signal_strength < -0.25 and confidence >= self.min_confidence:
                    signal = -1  # SELL
                else:
                    signal = 0  # HOLD

                signals.append({
                    'signal': signal,
                    'confidence': confidence,
                    'strength': abs(signal_strength)
                })

            # Add signals to dataframe
            signal_df = pd.DataFrame(signals, index=df.index[50:])

            # Handle potential column overlap by using specific assignment
            for col in ['signal', 'confidence', 'strength']:
                if col in signal_df.columns:
                    df[col] = signal_df[col]

            # Fill NaN values
            df['signal'] = df.get('signal', 0).fillna(0)
            df['confidence'] = df.get('confidence', 0).fillna(0)

            buy_signals = (df['signal'] == 1).sum()
            sell_signals = (df['signal'] == -1).sum()

            self.logger.info(f"✅ Generated signals: {buy_signals} BUY, {sell_signals} SELL")
            return df

        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return df

    def execute_backtest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute the backtest | 執行回測"""
        self.current_capital = self.initial_capital
        self.positions = []
        self.trades = []

        current_position = None

        for i, (timestamp, row) in enumerate(df.iterrows()):
            if pd.isna(row.get('signal', 0)) or row.get('signal', 0) == 0:
                continue

            signal = int(row['signal'])
            confidence = row.get('confidence', 0)
            current_price = row['Close']
            atr = row.get('atr', 0.01)

            # Close existing position if signal changes
            if current_position and current_position['signal'] != signal:
                # Close position
                exit_price = current_price
                pnl = self.calculate_pnl(current_position, exit_price)

                self.current_capital += pnl

                trade = {
                    'entry_time': current_position['entry_time'],
                    'exit_time': timestamp,
                    'signal': current_position['signal'],
                    'entry_price': current_position['entry_price'],
                    'exit_price': exit_price,
                    'position_size': current_position['position_size'],
                    'pnl': pnl,
                    'return_pct': pnl / current_position['position_size'] * 100
                }

                self.trades.append(trade)
                current_position = None

            # Open new position
            if not current_position and signal != 0:
                # Calculate position size
                risk_amount = self.current_capital * self.risk_per_trade
                stop_distance = atr * self.stop_loss_atr_multiplier
                position_size = min(risk_amount / stop_distance, self.current_capital * 0.1)  # Max 10% of capital

                # Stop loss and take profit levels
                if signal == 1:  # BUY
                    stop_loss = current_price - (atr * self.stop_loss_atr_multiplier)
                    take_profit = current_price + (atr * self.take_profit_atr_multiplier)
                else:  # SELL
                    stop_loss = current_price + (atr * self.stop_loss_atr_multiplier)
                    take_profit = current_price - (atr * self.take_profit_atr_multiplier)

                current_position = {
                    'entry_time': timestamp,
                    'signal': signal,
                    'entry_price': current_price,
                    'position_size': position_size,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence
                }

        # Close final position if exists
        if current_position:
            exit_price = df['Close'].iloc[-1]
            pnl = self.calculate_pnl(current_position, exit_price)
            self.current_capital += pnl

            trade = {
                'entry_time': current_position['entry_time'],
                'exit_time': df.index[-1],
                'signal': current_position['signal'],
                'entry_price': current_position['entry_price'],
                'exit_price': exit_price,
                'position_size': current_position['position_size'],
                'pnl': pnl,
                'return_pct': pnl / current_position['position_size'] * 100
            }

            self.trades.append(trade)

        return self.calculate_performance_metrics()

    def calculate_pnl(self, position: Dict, exit_price: float) -> float:
        """Calculate P&L for a position | 計算倉位盈虧"""
        entry_price = position['entry_price']
        position_size = position['position_size']
        signal = position['signal']

        if signal == 1:  # BUY position
            pnl = (exit_price - entry_price) / entry_price * position_size
        else:  # SELL position
            pnl = (entry_price - exit_price) / entry_price * position_size

        return pnl

    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics | 計算綜合績效指標"""
        if not self.trades:
            return {
                'total_trades': 0,
                'total_return_pct': 0,
                'final_capital': self.initial_capital,
                'win_rate': 0,
                'avg_return_per_trade': 0
            }

        trades_df = pd.DataFrame(self.trades)

        # Basic metrics
        total_trades = len(self.trades)
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100
        winning_trades = (trades_df['pnl'] > 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Advanced metrics
        avg_return_per_trade = trades_df['pnl'].mean()
        max_win = trades_df['pnl'].max()
        max_loss = trades_df['pnl'].min()

        # Calculate Sharpe ratio approximation
        returns = trades_df['return_pct'].values
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0

        return {
            'total_trades': total_trades,
            'total_return_pct': total_return,
            'final_capital': self.current_capital,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return_per_trade,
            'max_win': max_win,
            'max_loss': max_loss,
            'sharpe_ratio': sharpe_ratio,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'trades_data': trades_df.to_dict('records') if not trades_df.empty else []
        }

    def display_results(self, metrics: Dict[str, Any]):
        """Display backtest results | 顯示回測結果"""
        print("\n" + "="*80)
        print("🎯 USD/JPY BACKTESTING RESULTS | 美元/日圓回測結果")
        print("="*80)

        print(f"📊 Initial Capital: ${self.initial_capital:,.2f}")
        print(f"📊 Final Capital: ${metrics['final_capital']:,.2f}")
        print(f"📈 Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"📊 Total Trades: {metrics['total_trades']}")

        if metrics['total_trades'] > 0:
            print(f"🎯 Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"✅ Winning Trades: {metrics['winning_trades']}")
            print(f"❌ Losing Trades: {metrics['losing_trades']}")
            print(f"💰 Average Return per Trade: ${metrics['avg_return_per_trade']:,.2f}")
            print(f"🚀 Best Trade: ${metrics['max_win']:,.2f}")
            print(f"📉 Worst Trade: ${metrics['max_loss']:,.2f}")
            print(f"📊 Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")

            # Show recent trades
            if len(metrics['trades_data']) > 0:
                print(f"\n📋 Recent Trades (Last 5):")
                recent_trades = metrics['trades_data'][-5:]
                for i, trade in enumerate(recent_trades, 1):
                    pnl_status = "✅" if trade['pnl'] > 0 else "❌"
                    signal_str = "BUY" if trade['signal'] == 1 else "SELL"
                    print(f"  {pnl_status} Trade {len(metrics['trades_data'])-5+i}: {signal_str} "
                          f"${trade['pnl']:,.2f} ({trade['return_pct']:.2f}%)")

        print("="*80)

def main():
    """Main execution function | 主執行函數"""
    print("🚀 Starting Simple USD/JPY Backtesting...")
    print("🚀 開始簡單的美元/日圓回測...")

    # Initialize backtester
    backtester = SimpleUSDJPYBacktester(initial_capital=100000)

    # Load data
    print("\n📊 Loading USD/JPY data...")
    df = backtester.load_data()

    if df.empty:
        print("❌ No data loaded. Exiting...")
        return

    print(f"✅ Loaded {len(df)} data points from {df.index[0]} to {df.index[-1]}")

    # Generate signals
    print("\n🔧 Generating technical signals...")
    df_with_signals = backtester.generate_signals(df)

    # Execute backtest
    print("\n🔄 Executing backtest...")
    results = backtester.execute_backtest(df_with_signals)

    # Display results
    backtester.display_results(results)

    print(f"\n✅ Backtest completed successfully!")
    print(f"✅ 回測成功完成！")

if __name__ == "__main__":
    main()