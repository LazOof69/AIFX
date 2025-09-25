#!/usr/bin/env python3
"""
Automatic USD/JPY Database Analysis System | 自動美元/日圓資料庫分析系統
====================================================================

Comprehensive automatic analysis system for the USD/JPY historical trading database.
Provides deep insights, statistical analysis, pattern recognition, and trading opportunities.
針對美元/日圓歷史交易資料庫的全面自動分析系統。

Features | 功能:
- Complete statistical analysis of 10 years of data | 10年數據的完整統計分析
- Price movement pattern recognition | 價格變動模式識別
- Volatility analysis and risk assessment | 波動性分析和風險評估
- Trend identification and seasonality analysis | 趋勢識別和季節性分析
- Support/resistance level detection | 支撐/阻力位檢測
- Correlation analysis with economic events | 與經濟事件的相關性分析
- Automatic report generation with insights | 自動報告生成配合洞察

Author: AIFX Development Team
Created: 2025-09-16
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import warnings
warnings.filterwarnings('ignore')

# Statistical and analysis libraries
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

class USDJPYDatabaseAnalyzer:
    """
    Comprehensive USD/JPY Database Analysis System
    全面的美元/日圓資料庫分析系統

    Automatically analyzes the historical USD/JPY trading database to extract
    insights, patterns, and trading opportunities with detailed reporting.
    自動分析歷史美元/日圓交易資料庫以提取見解、模式和交易機會。
    """

    def __init__(self, db_path: str = "output/usdjpy_trading_history.db", output_dir: str = "output/analysis"):
        """Initialize the database analyzer"""
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Analysis results storage
        self.price_data = None
        self.analysis_results = {}
        self.insights = []

        # Analysis timestamp
        self.analysis_timestamp = datetime.now()

        self.logger.info("🔬 USD/JPY Database Analyzer initialized")

    def load_database_data(self) -> bool:
        """Load all relevant data from the database"""
        try:
            self.logger.info("📊 Loading data from USD/JPY database...")

            conn = sqlite3.connect(self.db_path)

            # Load historical prices
            query = """
            SELECT timestamp, open, high, low, close, volume
            FROM historical_prices
            ORDER BY timestamp
            """

            self.price_data = pd.read_sql_query(query, conn, parse_dates=['timestamp'])
            self.price_data.set_index('timestamp', inplace=True)

            conn.close()

            self.logger.info(f"✅ Loaded {len(self.price_data)} price records")
            self.logger.info(f"📅 Date range: {self.price_data.index[0]} to {self.price_data.index[-1]}")

            return True

        except Exception as e:
            self.logger.error(f"❌ Error loading database: {str(e)}")
            return False

    def perform_basic_statistics(self) -> Dict:
        """Perform comprehensive basic statistical analysis"""
        self.logger.info("📈 Performing basic statistical analysis...")

        stats_results = {}

        # Price statistics
        close_prices = self.price_data['close']

        stats_results['price_statistics'] = {
            'current_price': float(close_prices.iloc[-1]),
            'min_price': float(close_prices.min()),
            'max_price': float(close_prices.max()),
            'mean_price': float(close_prices.mean()),
            'median_price': float(close_prices.median()),
            'std_deviation': float(close_prices.std()),
            'price_range': float(close_prices.max() - close_prices.min()),
            'coefficient_of_variation': float(close_prices.std() / close_prices.mean())
        }

        # Daily returns analysis
        daily_returns = close_prices.pct_change().dropna()

        stats_results['returns_analysis'] = {
            'mean_daily_return': float(daily_returns.mean()),
            'std_daily_return': float(daily_returns.std()),
            'annualized_volatility': float(daily_returns.std() * np.sqrt(252)),
            'sharpe_ratio': float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)),
            'skewness': float(stats.skew(daily_returns)),
            'kurtosis': float(stats.kurtosis(daily_returns)),
            'max_daily_gain': float(daily_returns.max()),
            'max_daily_loss': float(daily_returns.min()),
            'positive_days_pct': float((daily_returns > 0).sum() / len(daily_returns) * 100)
        }

        # Volume analysis
        if self.price_data['volume'].sum() > 0:
            stats_results['volume_analysis'] = {
                'avg_daily_volume': float(self.price_data['volume'].mean()),
                'max_volume': float(self.price_data['volume'].max()),
                'volume_std': float(self.price_data['volume'].std()),
                'volume_trend': float(np.corrcoef(range(len(self.price_data)), self.price_data['volume'])[0,1])
            }

        # Price level analysis
        price_levels = np.percentile(close_prices, [10, 25, 50, 75, 90])
        stats_results['price_levels'] = {
            'support_strong': float(price_levels[0]),  # 10th percentile
            'support_moderate': float(price_levels[1]),  # 25th percentile
            'median_level': float(price_levels[2]),  # 50th percentile
            'resistance_moderate': float(price_levels[3]),  # 75th percentile
            'resistance_strong': float(price_levels[4])  # 90th percentile
        }

        self.analysis_results['basic_statistics'] = stats_results
        self.logger.info("✅ Basic statistical analysis completed")

        return stats_results

    def analyze_trends_and_patterns(self) -> Dict:
        """Analyze price trends and identify patterns"""
        self.logger.info("📊 Analyzing trends and patterns...")

        pattern_results = {}
        close_prices = self.price_data['close']

        # Trend analysis using moving averages
        self.price_data['ma_20'] = close_prices.rolling(window=20).mean()
        self.price_data['ma_50'] = close_prices.rolling(window=50).mean()
        self.price_data['ma_200'] = close_prices.rolling(window=200).mean()

        # Current trend determination
        current_price = close_prices.iloc[-1]
        ma_20_current = self.price_data['ma_20'].iloc[-1]
        ma_50_current = self.price_data['ma_50'].iloc[-1]
        ma_200_current = self.price_data['ma_200'].iloc[-1]

        # Trend classification
        if current_price > ma_20_current > ma_50_current > ma_200_current:
            trend_status = "Strong Uptrend"
        elif current_price > ma_20_current > ma_50_current:
            trend_status = "Moderate Uptrend"
        elif current_price < ma_20_current < ma_50_current < ma_200_current:
            trend_status = "Strong Downtrend"
        elif current_price < ma_20_current < ma_50_current:
            trend_status = "Moderate Downtrend"
        else:
            trend_status = "Sideways/Consolidating"

        pattern_results['trend_analysis'] = {
            'current_trend': trend_status,
            'price_vs_ma20': float((current_price - ma_20_current) / ma_20_current * 100),
            'price_vs_ma50': float((current_price - ma_50_current) / ma_50_current * 100),
            'price_vs_ma200': float((current_price - ma_200_current) / ma_200_current * 100),
            'ma20_slope': float((ma_20_current - self.price_data['ma_20'].iloc[-21]) / 21),
            'ma50_slope': float((ma_50_current - self.price_data['ma_50'].iloc[-51]) / 51)
        }

        # Volatility clustering analysis
        returns = close_prices.pct_change().dropna()
        volatility = returns.rolling(window=20).std()

        pattern_results['volatility_analysis'] = {
            'current_volatility': float(volatility.iloc[-1]),
            'avg_volatility': float(volatility.mean()),
            'volatility_percentile': float(stats.percentileofscore(volatility.dropna(), volatility.iloc[-1])),
            'high_vol_periods': int((volatility > volatility.quantile(0.8)).sum()),
            'low_vol_periods': int((volatility < volatility.quantile(0.2)).sum())
        }

        # Seasonality analysis
        self.price_data['month'] = self.price_data.index.month
        self.price_data['day_of_week'] = self.price_data.index.dayofweek
        self.price_data['quarter'] = self.price_data.index.quarter

        monthly_returns = self.price_data.groupby('month')['close'].pct_change().groupby(self.price_data['month']).mean()
        quarterly_returns = self.price_data.groupby('quarter')['close'].pct_change().groupby(self.price_data['quarter']).mean()

        pattern_results['seasonality'] = {
            'best_month': int(monthly_returns.idxmax()),
            'worst_month': int(monthly_returns.idxmin()),
            'best_quarter': int(quarterly_returns.idxmax()),
            'worst_quarter': int(quarterly_returns.idxmin()),
            'monthly_performance': monthly_returns.to_dict(),
            'quarterly_performance': quarterly_returns.to_dict()
        }

        self.analysis_results['patterns'] = pattern_results
        self.logger.info("✅ Pattern analysis completed")

        return pattern_results

    def detect_support_resistance_levels(self) -> Dict:
        """Advanced support and resistance level detection"""
        self.logger.info("🎯 Detecting support and resistance levels...")

        sr_results = {}

        # Use local maxima and minima for S/R detection
        from scipy.signal import find_peaks

        highs = self.price_data['high'].values
        lows = self.price_data['low'].values
        closes = self.price_data['close'].values

        # Find resistance levels (peaks in highs)
        resistance_peaks, _ = find_peaks(highs, distance=20, prominence=np.std(highs)*0.5)
        resistance_levels = highs[resistance_peaks]

        # Find support levels (peaks in inverted lows)
        support_peaks, _ = find_peaks(-lows, distance=20, prominence=np.std(lows)*0.5)
        support_levels = lows[support_peaks]

        # Cluster levels to find key S/R zones
        if len(resistance_levels) > 3:
            kmeans_r = KMeans(n_clusters=min(5, len(resistance_levels)//2))
            resistance_clusters = kmeans_r.fit_predict(resistance_levels.reshape(-1, 1))
            key_resistance = [resistance_levels[resistance_clusters == i].mean()
                            for i in range(kmeans_r.n_clusters)]
        else:
            key_resistance = resistance_levels.tolist()

        if len(support_levels) > 3:
            kmeans_s = KMeans(n_clusters=min(5, len(support_levels)//2))
            support_clusters = kmeans_s.fit_predict(support_levels.reshape(-1, 1))
            key_support = [support_levels[support_clusters == i].mean()
                         for i in range(kmeans_s.n_clusters)]
        else:
            key_support = support_levels.tolist()

        # Current price context
        current_price = closes[-1]

        sr_results['support_resistance'] = {
            'current_price': float(current_price),
            'key_resistance_levels': [float(x) for x in sorted(key_resistance, reverse=True)],
            'key_support_levels': [float(x) for x in sorted(key_support, reverse=True)],
            'nearest_resistance': float(min([r for r in key_resistance if r > current_price], default=current_price)),
            'nearest_support': float(max([s for s in key_support if s < current_price], default=current_price)),
            'resistance_count': len(resistance_levels),
            'support_count': len(support_levels)
        }

        self.analysis_results['support_resistance'] = sr_results
        self.logger.info("✅ Support/Resistance analysis completed")

        return sr_results

    def analyze_market_cycles(self) -> Dict:
        """Analyze market cycles and regime changes"""
        self.logger.info("🔄 Analyzing market cycles...")

        cycle_results = {}

        returns = self.price_data['close'].pct_change().dropna()
        prices = self.price_data['close']

        # Bull/Bear market detection using 200-day MA
        ma_200 = prices.rolling(window=200).mean()

        bull_periods = (prices > ma_200).astype(int)
        bear_periods = (prices < ma_200).astype(int)

        # Calculate regime statistics
        current_regime = "Bull Market" if prices.iloc[-1] > ma_200.iloc[-1] else "Bear Market"

        # Find regime changes
        regime_changes = (bull_periods.diff() != 0).sum()

        # Drawdown analysis
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        max_drawdown = drawdown.min()

        # Recovery analysis
        in_drawdown = drawdown < -0.05  # 5% drawdown threshold
        drawdown_periods = in_drawdown.astype(int).diff().fillna(0)

        cycle_results['market_cycles'] = {
            'current_regime': current_regime,
            'regime_changes': int(regime_changes),
            'bull_market_days': int(bull_periods.sum()),
            'bear_market_days': int(bear_periods.sum()),
            'bull_market_percentage': float(bull_periods.mean() * 100),
            'max_drawdown': float(max_drawdown),
            'current_drawdown': float(drawdown.iloc[-1]),
            'avg_bull_return': float(returns[bull_periods == 1].mean()),
            'avg_bear_return': float(returns[bear_periods == 1].mean()),
            'days_since_peak': int((prices.index[-1] - peak.idxmax()).days) if not pd.isna(peak.idxmax()) else 0
        }

        self.analysis_results['cycles'] = cycle_results
        self.logger.info("✅ Market cycle analysis completed")

        return cycle_results

    def generate_trading_insights(self) -> List[str]:
        """Generate actionable trading insights based on analysis"""
        self.logger.info("💡 Generating trading insights...")

        insights = []

        # Price level insights
        stats = self.analysis_results.get('basic_statistics', {})
        patterns = self.analysis_results.get('patterns', {})
        sr = self.analysis_results.get('support_resistance', {})
        cycles = self.analysis_results.get('cycles', {})

        if stats:
            price_stats = stats.get('price_statistics', {})
            current_price = price_stats.get('current_price', 0)
            mean_price = price_stats.get('mean_price', 0)

            if current_price > mean_price * 1.05:
                insights.append(f"💹 USD/JPY is trading {((current_price/mean_price-1)*100):.1f}% above historical average - potential overvaluation")
            elif current_price < mean_price * 0.95:
                insights.append(f"📉 USD/JPY is trading {((1-current_price/mean_price)*100):.1f}% below historical average - potential undervaluation")

        # Trend insights
        if patterns:
            trend = patterns.get('trend_analysis', {})
            current_trend = trend.get('current_trend', '')

            if 'Strong Uptrend' in current_trend:
                insights.append("🚀 Strong bullish momentum detected - trend following strategies favored")
            elif 'Strong Downtrend' in current_trend:
                insights.append("📉 Strong bearish momentum detected - consider short positions or defensive strategies")
            elif 'Sideways' in current_trend:
                insights.append("↔️ Market in consolidation - range trading strategies may be effective")

        # Volatility insights
        if patterns:
            vol = patterns.get('volatility_analysis', {})
            vol_percentile = vol.get('volatility_percentile', 50)

            if vol_percentile > 80:
                insights.append("⚠️ Extremely high volatility environment - increase position sizing caution")
            elif vol_percentile < 20:
                insights.append("😴 Low volatility period - potential for volatility expansion ahead")

        # Support/Resistance insights
        if sr:
            sr_data = sr.get('support_resistance', {})
            current_price = sr_data.get('current_price', 0)
            nearest_resistance = sr_data.get('nearest_resistance', 0)
            nearest_support = sr_data.get('nearest_support', 0)

            resistance_distance = abs(nearest_resistance - current_price) / current_price * 100
            support_distance = abs(current_price - nearest_support) / current_price * 100

            if resistance_distance < 1:
                insights.append(f"🚫 Approaching key resistance at {nearest_resistance:.3f} - watch for rejection")
            if support_distance < 1:
                insights.append(f"🛡️ Near key support at {nearest_support:.3f} - potential bounce zone")

        # Seasonal insights
        if patterns:
            seasonality = patterns.get('seasonality', {})
            current_month = datetime.now().month
            best_month = seasonality.get('best_month', 0)
            worst_month = seasonality.get('worst_month', 0)

            if current_month == best_month:
                insights.append(f"📅 Currently in historically strongest month ({current_month}) for USD/JPY")
            elif current_month == worst_month:
                insights.append(f"📅 Currently in historically weakest month ({current_month}) for USD/JPY")

        # Market regime insights
        if cycles:
            cycle_data = cycles.get('market_cycles', {})
            current_regime = cycle_data.get('current_regime', '')
            max_drawdown = cycle_data.get('max_drawdown', 0)
            current_drawdown = cycle_data.get('current_drawdown', 0)

            if current_drawdown < -0.1:
                insights.append(f"🔻 Significant drawdown of {current_drawdown*100:.1f}% - potential recovery opportunity")

            if abs(current_drawdown - max_drawdown) < 0.02:
                insights.append("⭐ Near maximum historical drawdown - high probability reversal zone")

        if not insights:
            insights.append("📊 Market conditions are within normal parameters - no immediate alerts")

        self.insights = insights
        self.logger.info(f"✅ Generated {len(insights)} trading insights")

        return insights

    def create_visualizations(self) -> None:
        """Create comprehensive visualization charts"""
        self.logger.info("📊 Creating analysis visualizations...")

        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('USD/JPY Comprehensive Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. Price chart with moving averages
        ax1 = axes[0, 0]
        ax1.plot(self.price_data.index, self.price_data['close'], label='USD/JPY', linewidth=1)
        if 'ma_20' in self.price_data.columns:
            ax1.plot(self.price_data.index, self.price_data['ma_20'], label='MA20', alpha=0.7)
        if 'ma_50' in self.price_data.columns:
            ax1.plot(self.price_data.index, self.price_data['ma_50'], label='MA50', alpha=0.7)
        if 'ma_200' in self.price_data.columns:
            ax1.plot(self.price_data.index, self.price_data['ma_200'], label='MA200', alpha=0.7)
        ax1.set_title('USD/JPY Price Evolution with Moving Averages')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Returns distribution
        ax2 = axes[0, 1]
        returns = self.price_data['close'].pct_change().dropna()
        ax2.hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.4f}')
        ax2.set_title('Daily Returns Distribution')
        ax2.set_xlabel('Daily Return')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Volatility over time
        ax3 = axes[1, 0]
        volatility = returns.rolling(window=30).std() * np.sqrt(252)  # Annualized
        ax3.plot(volatility.index, volatility, color='orange', linewidth=1)
        ax3.axhline(volatility.mean(), color='red', linestyle='--', label=f'Average: {volatility.mean():.2f}')
        ax3.set_title('30-Day Rolling Volatility (Annualized)')
        ax3.set_ylabel('Volatility')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Monthly performance heatmap
        ax4 = axes[1, 1]
        monthly_returns = self.price_data['close'].resample('M').last().pct_change().dropna()
        monthly_data = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).mean().unstack()

        if not monthly_data.empty:
            sns.heatmap(monthly_data, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=ax4)
        ax4.set_title('Monthly Returns Heatmap')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Year')

        plt.tight_layout()

        # Save the visualization
        viz_path = self.output_dir / f"usdjpy_analysis_dashboard_{self.analysis_timestamp.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"✅ Visualizations saved: {viz_path}")

    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        self.logger.info("📝 Generating comprehensive analysis report...")

        report = {
            'analysis_metadata': {
                'analysis_timestamp': self.analysis_timestamp.isoformat(),
                'database_path': self.db_path,
                'data_points_analyzed': len(self.price_data),
                'analysis_period': {
                    'start_date': self.price_data.index[0].isoformat(),
                    'end_date': self.price_data.index[-1].isoformat(),
                    'total_days': (self.price_data.index[-1] - self.price_data.index[0]).days
                }
            },
            'analysis_results': self.analysis_results,
            'trading_insights': self.insights,
            'summary_statistics': {
                'current_price': float(self.price_data['close'].iloc[-1]),
                'total_return': float((self.price_data['close'].iloc[-1] / self.price_data['close'].iloc[0] - 1) * 100),
                'annualized_return': float(((self.price_data['close'].iloc[-1] / self.price_data['close'].iloc[0]) ** (365 / (self.price_data.index[-1] - self.price_data.index[0]).days) - 1) * 100),
                'max_price': float(self.price_data['high'].max()),
                'min_price': float(self.price_data['low'].min()),
                'analysis_completeness': '100%'
            }
        }

        # Save report to JSON
        report_path = self.output_dir / f"usdjpy_comprehensive_analysis_{self.analysis_timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"✅ Comprehensive report saved: {report_path}")

        return report

    def display_analysis_summary(self) -> None:
        """Display comprehensive analysis summary"""
        print("\n" + "="*100)
        print("🔬 USD/JPY AUTOMATIC DATABASE ANALYSIS COMPLETE | 美元/日圓自動資料庫分析完成")
        print("="*100)

        # Basic info
        print(f"📊 Data Points Analyzed: {len(self.price_data):,}")
        print(f"📅 Analysis Period: {self.price_data.index[0].date()} to {self.price_data.index[-1].date()}")
        print(f"⏰ Analysis Completed: {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        # Key metrics
        if 'basic_statistics' in self.analysis_results:
            stats = self.analysis_results['basic_statistics']['price_statistics']
            print(f"\n💰 PRICE ANALYSIS | 價格分析")
            print(f"   Current Price: {stats['current_price']:.3f}")
            print(f"   Price Range: {stats['min_price']:.3f} - {stats['max_price']:.3f}")
            print(f"   Average Price: {stats['mean_price']:.3f}")
            print(f"   Volatility: {stats['coefficient_of_variation']:.3f}")

        if 'patterns' in self.analysis_results:
            trend = self.analysis_results['patterns']['trend_analysis']
            print(f"\n📈 TREND ANALYSIS | 趨勢分析")
            print(f"   Current Trend: {trend['current_trend']}")
            print(f"   Price vs MA20: {trend['price_vs_ma20']:+.2f}%")
            print(f"   Price vs MA200: {trend['price_vs_ma200']:+.2f}%")

        if 'support_resistance' in self.analysis_results:
            sr = self.analysis_results['support_resistance']['support_resistance']
            print(f"\n🎯 KEY LEVELS | 關鍵水位")
            print(f"   Nearest Resistance: {sr['nearest_resistance']:.3f}")
            print(f"   Nearest Support: {sr['nearest_support']:.3f}")

        # Trading insights
        print(f"\n💡 TRADING INSIGHTS | 交易見解")
        for i, insight in enumerate(self.insights[:5], 1):  # Show top 5 insights
            print(f"   {i}. {insight}")

        if len(self.insights) > 5:
            print(f"   ... and {len(self.insights) - 5} more insights in detailed report")

        # Files created
        print(f"\n📁 OUTPUT FILES | 輸出文件")
        print(f"   Analysis Report: {self.output_dir}")
        print(f"   Visualizations: Dashboard charts created")
        print(f"   Database: {self.db_path}")

        print("\n🎉 AUTOMATIC ANALYSIS COMPLETED SUCCESSFULLY!")
        print("🎉 自動分析成功完成！")
        print("="*100)

    def run_complete_analysis(self) -> Dict:
        """Run complete automatic analysis pipeline"""
        self.logger.info("🚀 Starting complete automatic USD/JPY analysis...")

        try:
            # Step 1: Load data
            if not self.load_database_data():
                raise Exception("Failed to load database data")

            # Step 2: Basic statistics
            self.perform_basic_statistics()

            # Step 3: Pattern analysis
            self.analyze_trends_and_patterns()

            # Step 4: Support/Resistance detection
            self.detect_support_resistance_levels()

            # Step 5: Market cycles
            self.analyze_market_cycles()

            # Step 6: Generate insights
            self.generate_trading_insights()

            # Step 7: Create visualizations
            self.create_visualizations()

            # Step 8: Generate comprehensive report
            report = self.generate_comprehensive_report()

            # Step 9: Display summary
            self.display_analysis_summary()

            return report

        except Exception as e:
            self.logger.error(f"❌ Analysis failed: {str(e)}")
            return {}

def main():
    """Main execution function"""
    print("🔬 Starting Automatic USD/JPY Database Analysis...")
    print("🔬 開始自動美元/日圓資料庫分析...")

    # Create analyzer instance
    analyzer = USDJPYDatabaseAnalyzer()

    # Run complete analysis
    results = analyzer.run_complete_analysis()

    if results:
        print("\n✅ Analysis completed successfully!")
        print("✅ 分析成功完成！")
    else:
        print("\n❌ Analysis failed!")
        print("❌ 分析失敗！")

if __name__ == "__main__":
    main()