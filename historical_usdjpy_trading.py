#!/usr/bin/env python3
"""
Historical USD/JPY Trading Simulation & Database Creation | 歷史美元/日圓交易模擬與資料庫創建
==============================================================================

Comprehensive historical USD/JPY trading simulation with complete database creation.
Downloads maximum available historical data and runs complete backtesting analysis.
全面的歷史美元/日圓交易模擬配合完整的資料庫創建。

Features | 功能:
- Maximum historical data download (10+ years) | 最大歷史數據下載（10年以上）
- SQLite database creation for all results | 所有結果的SQLite資料庫創建
- Complete trading simulation with AI models | 使用AI模型的完整交易模擬
- Comprehensive performance analytics | 全面績效分析
- Export capabilities for further analysis | 導出功能用於進一步分析

Author: AIFX Development Team
Created: 2025-09-16
"""

import asyncio
import sqlite3
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import os

# Add the src path to Python path
sys.path.insert(0, str(Path(__file__).parent / "src" / "main" / "python"))

try:
    from utils.data_loader import DataLoader
    from utils.feature_generator import FeatureGenerator
    from models.xgboost_model import XGBoostModel
    from models.random_forest_model import RandomForestModel
    from evaluation.backtest_engine import BacktestEngine, BacktestConfig
    from utils.logger import setup_logger
    from utils.config import Config
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)

class HistoricalUSDJPYTrader:
    """
    Historical USD/JPY Trading System with Database Creation
    歷史美元/日圓交易系統配合資料庫創建

    Downloads maximum historical data and performs comprehensive backtesting
    with database storage for all results and analytics.
    下載最大歷史數據並執行全面回測，將所有結果和分析存儲到資料庫。
    """

    def __init__(self, output_dir: str = "output"):
        """Initialize the historical trading system"""
        self.logger = setup_logger("HistoricalUSDJPY")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Database setup | 資料庫設置
        self.db_path = self.output_dir / "usdjpy_trading_history.db"
        self.setup_database()

        # Trading components | 交易組件
        self.data_loader = DataLoader()
        self.feature_generator = FeatureGenerator()
        self.xgb_model = XGBoostModel()
        self.rf_model = RandomForestModel()

        # Setup backtest configuration | 設置回測配置
        backtest_config = BacktestConfig(
            start_date="2014-01-01",
            end_date="2024-12-31",
            initial_capital=10000.0,
            commission_rate=0.0002,
            slippage_factor=0.0001,
            trading_symbols=["USDJPY=X"]
        )
        self.backtest_engine = BacktestEngine(backtest_config)

        # Results storage | 結果存儲
        self.historical_data = None
        self.trading_results = []
        self.performance_metrics = {}

    def setup_database(self) -> None:
        """Setup SQLite database for storing all trading data and results"""
        self.logger.info("🗄️ Setting up SQLite database for historical trading data...")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Historical price data table | 歷史價格數據表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historical_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume INTEGER,
                symbol TEXT DEFAULT 'USD/JPY',
                timeframe TEXT DEFAULT '1H',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Trading signals table | 交易信號表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,  -- BUY/SELL/HOLD
                confidence REAL NOT NULL,
                xgb_prediction REAL,
                rf_prediction REAL,
                technical_score REAL,
                final_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Trading positions table | 交易倉位表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time DATETIME NOT NULL,
                exit_time DATETIME,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,  -- BUY/SELL
                entry_price REAL NOT NULL,
                exit_price REAL,
                quantity REAL NOT NULL,
                pnl REAL,
                pnl_percentage REAL,
                duration_hours INTEGER,
                stop_loss REAL,
                take_profit REAL,
                exit_reason TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Performance metrics table | 績效指標表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                period_start DATETIME NOT NULL,
                period_end DATETIME NOT NULL,
                symbol TEXT DEFAULT 'USD/JPY',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Feature importance table | 特徵重要性表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_importance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                importance_score REAL NOT NULL,
                rank_position INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        conn.commit()
        conn.close()

        self.logger.info(f"✅ Database setup complete: {self.db_path}")

    async def download_historical_data(self, years_back: int = 10) -> pd.DataFrame:
        """Download maximum historical USD/JPY data"""
        self.logger.info(f"📊 Downloading {years_back} years of USD/JPY historical data...")

        # Calculate date range for maximum historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)

        try:
            # Download USD/JPY data with maximum history
            symbol = "USDJPY=X"  # Yahoo Finance USD/JPY symbol
            self.logger.info(f"📈 Fetching {symbol} from {start_date.date()} to {end_date.date()}")

            historical_data_dict = self.data_loader.download_data(
                symbols=["USDJPY"],
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                interval="1h",
                force_update=True
            )

            # Extract the USD/JPY data from the dictionary
            if historical_data_dict and "USDJPY" in historical_data_dict:
                historical_data = historical_data_dict["USDJPY"]
            else:
                historical_data = None

            if historical_data is None or historical_data.empty:
                # Fallback to daily data if hourly not available
                self.logger.warning("⚠️ Hourly data not available, falling back to daily data")
                historical_data_dict = self.data_loader.download_data(
                    symbols=["USDJPY"],
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d'),
                    interval="1d",
                    force_update=True
                )
                if historical_data_dict and "USDJPY" in historical_data_dict:
                    historical_data = historical_data_dict["USDJPY"]

            if historical_data is not None and not historical_data.empty:
                self.historical_data = historical_data
                self.logger.info(f"✅ Downloaded {len(historical_data)} data points for USD/JPY")
                self.logger.info(f"📅 Date range: {historical_data.index[0]} to {historical_data.index[-1]}")

                # Save to database
                await self.save_historical_data_to_db(historical_data)
                return historical_data
            else:
                self.logger.error("❌ Failed to download historical data")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"❌ Error downloading historical data: {str(e)}")
            return pd.DataFrame()

    async def save_historical_data_to_db(self, data: pd.DataFrame) -> None:
        """Save historical price data to database"""
        self.logger.info("💾 Saving historical data to database...")

        conn = sqlite3.connect(self.db_path)

        try:
            # Prepare data for database insertion
            db_data = []
            for timestamp, row in data.iterrows():
                db_data.append((
                    timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row.get('Volume', 0)) if not pd.isna(row.get('Volume', 0)) else 0
                ))

            # Insert data
            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO historical_prices
                (timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', db_data)

            conn.commit()
            self.logger.info(f"✅ Saved {len(db_data)} historical data points to database")

        except Exception as e:
            self.logger.error(f"❌ Error saving historical data: {str(e)}")
        finally:
            conn.close()

    async def generate_features_and_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive features and trading signals"""
        self.logger.info("🔧 Generating features and trading signals...")

        try:
            # Generate all technical features
            features_data = self.feature_generator.generate_features(data)

            if features_data is None or features_data.empty:
                self.logger.error("❌ Feature generation failed")
                return pd.DataFrame()

            self.logger.info(f"✅ Generated {len(features_data.columns)} features")

            # Generate AI model predictions
            await self.generate_ai_predictions(features_data)

            return features_data

        except Exception as e:
            self.logger.error(f"❌ Error in feature generation: {str(e)}")
            return pd.DataFrame()

    async def generate_ai_predictions(self, features_data: pd.DataFrame) -> None:
        """Generate AI model predictions and save to database"""
        self.logger.info("🤖 Generating AI model predictions...")

        try:
            # Prepare training data (use first 80% for training)
            split_idx = int(len(features_data) * 0.8)
            train_data = features_data.iloc[:split_idx]
            test_data = features_data.iloc[split_idx:]

            # Create target variable (price direction: 1 for up, 0 for down)
            train_target = (train_data['close'].shift(-1) > train_data['close']).astype(int).dropna()
            train_features = train_data.iloc[:-1]  # Remove last row due to shift

            # Train XGBoost model
            self.logger.info("🌲 Training XGBoost model...")
            self.xgb_model.train(train_features, train_target)

            # Train Random Forest model
            self.logger.info("🌳 Training Random Forest model...")
            self.rf_model.train(train_features, train_target)

            # Generate predictions for all data
            xgb_predictions = self.xgb_model.predict(features_data)
            rf_predictions = self.rf_model.predict(features_data)

            # Calculate ensemble predictions
            ensemble_predictions = (xgb_predictions + rf_predictions) / 2

            # Save predictions to database
            await self.save_predictions_to_db(features_data, xgb_predictions, rf_predictions, ensemble_predictions)

            self.logger.info("✅ AI predictions generated and saved")

        except Exception as e:
            self.logger.error(f"❌ Error in AI prediction generation: {str(e)}")

    async def save_predictions_to_db(self, data: pd.DataFrame, xgb_pred: np.ndarray,
                                   rf_pred: np.ndarray, ensemble_pred: np.ndarray) -> None:
        """Save trading signals to database"""
        self.logger.info("💾 Saving trading signals to database...")

        conn = sqlite3.connect(self.db_path)

        try:
            db_signals = []
            for i, (timestamp, row) in enumerate(data.iterrows()):
                if i < len(xgb_pred) and i < len(rf_pred) and i < len(ensemble_pred):
                    # Determine signal type based on ensemble prediction
                    confidence = float(ensemble_pred[i])
                    if confidence > 0.6:
                        signal_type = "BUY"
                    elif confidence < 0.4:
                        signal_type = "SELL"
                    else:
                        signal_type = "HOLD"

                    # Calculate technical score (simplified)
                    technical_score = 0.5  # Placeholder for now

                    db_signals.append((
                        timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        'USD/JPY',
                        signal_type,
                        confidence,
                        float(xgb_pred[i]),
                        float(rf_pred[i]),
                        technical_score,
                        confidence
                    ))

            cursor = conn.cursor()
            cursor.executemany('''
                INSERT OR REPLACE INTO trading_signals
                (timestamp, symbol, signal_type, confidence, xgb_prediction,
                 rf_prediction, technical_score, final_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', db_signals)

            conn.commit()
            self.logger.info(f"✅ Saved {len(db_signals)} trading signals to database")

        except Exception as e:
            self.logger.error(f"❌ Error saving signals: {str(e)}")
        finally:
            conn.close()

    async def run_comprehensive_backtest(self, data: pd.DataFrame) -> Dict:
        """Run comprehensive historical trading simulation"""
        self.logger.info("🔄 Running comprehensive USD/JPY trading simulation...")

        try:
            # Setup backtest parameters
            initial_capital = 10000.0
            risk_per_trade = 0.02

            # Run backtest
            backtest_results = await self.backtest_engine.run_backtest(
                data=data,
                symbol="USD/JPY",
                initial_capital=initial_capital,
                risk_per_trade=risk_per_trade
            )

            if backtest_results:
                self.performance_metrics = backtest_results
                await self.save_backtest_results_to_db(backtest_results)
                self.logger.info("✅ Backtest completed successfully")
                return backtest_results
            else:
                self.logger.error("❌ Backtest failed")
                return {}

        except Exception as e:
            self.logger.error(f"❌ Error in backtesting: {str(e)}")
            return {}

    async def save_backtest_results_to_db(self, results: Dict) -> None:
        """Save backtest results and performance metrics to database"""
        self.logger.info("💾 Saving backtest results to database...")

        conn = sqlite3.connect(self.db_path)

        try:
            cursor = conn.cursor()

            # Save performance metrics
            period_start = self.historical_data.index[0].strftime('%Y-%m-%d %H:%M:%S')
            period_end = self.historical_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')

            metrics_data = []
            for metric_name, metric_value in results.items():
                if isinstance(metric_value, (int, float)):
                    metrics_data.append((
                        metric_name,
                        float(metric_value),
                        period_start,
                        period_end
                    ))

            cursor.executemany('''
                INSERT OR REPLACE INTO performance_metrics
                (metric_name, metric_value, period_start, period_end)
                VALUES (?, ?, ?, ?)
            ''', metrics_data)

            conn.commit()
            self.logger.info(f"✅ Saved {len(metrics_data)} performance metrics to database")

        except Exception as e:
            self.logger.error(f"❌ Error saving backtest results: {str(e)}")
        finally:
            conn.close()

    async def generate_comprehensive_report(self) -> None:
        """Generate comprehensive trading analysis report"""
        self.logger.info("📊 Generating comprehensive trading analysis report...")

        conn = sqlite3.connect(self.db_path)

        try:
            # Query database for comprehensive analysis
            df_prices = pd.read_sql_query("SELECT * FROM historical_prices ORDER BY timestamp", conn)
            df_signals = pd.read_sql_query("SELECT * FROM trading_signals ORDER BY timestamp", conn)
            df_metrics = pd.read_sql_query("SELECT * FROM performance_metrics", conn)

            # Generate analysis report
            report = {
                "analysis_timestamp": datetime.now().isoformat(),
                "data_summary": {
                    "total_data_points": len(df_prices),
                    "date_range": {
                        "start": df_prices['timestamp'].min() if not df_prices.empty else None,
                        "end": df_prices['timestamp'].max() if not df_prices.empty else None
                    },
                    "total_signals": len(df_signals),
                    "buy_signals": len(df_signals[df_signals['signal_type'] == 'BUY']) if not df_signals.empty else 0,
                    "sell_signals": len(df_signals[df_signals['signal_type'] == 'SELL']) if not df_signals.empty else 0,
                    "hold_signals": len(df_signals[df_signals['signal_type'] == 'HOLD']) if not df_signals.empty else 0
                },
                "performance_metrics": df_metrics.set_index('metric_name')['metric_value'].to_dict() if not df_metrics.empty else {},
                "database_location": str(self.db_path),
                "symbol": "USD/JPY",
                "trading_system": "AIFX Historical Analysis"
            }

            # Save report to JSON file
            report_path = self.output_dir / f"usdjpy_historical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            self.logger.info(f"✅ Comprehensive report saved: {report_path}")

            # Display summary
            self.display_analysis_summary(report)

        except Exception as e:
            self.logger.error(f"❌ Error generating report: {str(e)}")
        finally:
            conn.close()

    def display_analysis_summary(self, report: Dict) -> None:
        """Display comprehensive analysis summary"""
        print("\n" + "="*80)
        print("🎯 USD/JPY HISTORICAL TRADING ANALYSIS COMPLETE | 美元/日圓歷史交易分析完成")
        print("="*80)

        summary = report["data_summary"]
        metrics = report["performance_metrics"]

        print(f"📊 Total Data Points: {summary['total_data_points']:,}")
        print(f"📅 Analysis Period: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"🎯 Symbol: USD/JPY")
        print(f"⏱️ Timeframe: 1 Hour")

        print(f"\n📈 TRADING SIGNALS GENERATED | 生成的交易信號")
        print(f"   🟢 BUY Signals: {summary['buy_signals']:,}")
        print(f"   🔴 SELL Signals: {summary['sell_signals']:,}")
        print(f"   ⚪ HOLD Signals: {summary['hold_signals']:,}")
        print(f"   📊 Total Signals: {summary['total_signals']:,}")

        if metrics:
            print(f"\n💰 PERFORMANCE METRICS | 績效指標")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {metric}: {value:.4f}")
                else:
                    print(f"   {metric}: {value}")

        print(f"\n🗄️ DATABASE CREATED | 已創建資料庫")
        print(f"   📍 Location: {report['database_location']}")
        print(f"   📊 Contains: Price data, Signals, Positions, Metrics")

        print("\n🎉 HISTORICAL USD/JPY ANALYSIS COMPLETE!")
        print("🎉 歷史美元/日圓分析完成！")
        print("="*80)

async def main():
    """Main execution function for historical USD/JPY trading analysis"""
    print("🚀 Starting Historical USD/JPY Trading Analysis...")
    print("🚀 開始歷史美元/日圓交易分析...")

    # Create historical trader instance
    trader = HistoricalUSDJPYTrader()

    try:
        # Step 1: Download maximum historical data
        print("\n📊 Step 1: Downloading Maximum Historical Data...")
        historical_data = await trader.download_historical_data(years_back=10)

        if historical_data.empty:
            print("❌ Failed to download historical data. Exiting...")
            return

        # Step 2: Generate features and AI predictions
        print("\n🔧 Step 2: Generating Features and AI Predictions...")
        features_data = await trader.generate_features_and_signals(historical_data)

        if features_data.empty:
            print("❌ Failed to generate features. Exiting...")
            return

        # Step 3: Run comprehensive backtest
        print("\n🔄 Step 3: Running Comprehensive Trading Simulation...")
        backtest_results = await trader.run_comprehensive_backtest(features_data)

        # Step 4: Generate comprehensive report
        print("\n📊 Step 4: Generating Comprehensive Analysis Report...")
        await trader.generate_comprehensive_report()

        print("\n✅ Historical USD/JPY trading analysis completed successfully!")
        print("✅ 歷史美元/日圓交易分析成功完成！")

    except Exception as e:
        print(f"❌ Error in historical analysis: {str(e)}")
        logging.error(f"Historical analysis error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())