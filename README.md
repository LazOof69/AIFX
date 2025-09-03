# AIFX

## Quick Start

1. **Read CLAUDE.md first** - Contains essential rules for Claude Code
2. Follow the pre-task compliance checklist before starting any work
3. Use proper module structure under `src/main/python/`
4. Commit after every completed task

## Project Overview

**AIFX** is a professional quantitative trading system that implements medium-term forex strategies enhanced with AI models. The system focuses on EUR/USD and USD/JPY trading pairs using 1-hour timeframes.

### Key Features

- **Data & Feature Engineering**: Historical OHLCV data with technical indicators (MA, MACD, RSI, Bollinger Bands, ATR)
- **AI Models**: Machine learning models (XGBoost, Random Forest, LSTM) for price direction prediction  
- **Strategy Logic**: Combined technical and AI signals with confidence filtering
- **Risk Management**: Fixed percentage risk, stop-loss/take-profit using ATR multiples
- **Backtesting**: Backtrader framework with comprehensive performance metrics

## AI/ML Project Structure

**Complete MLOps-ready structure with data, models, experiments**

```
AIFX/
├── CLAUDE.md              # Essential rules for Claude Code
├── src/                   # Source code (NEVER put files in root)
│   ├── main/              # Main application code
│   │   ├── python/        # Python implementation
│   │   │   ├── core/      # Core trading algorithms
│   │   │   ├── utils/     # Data processing utilities
│   │   │   ├── models/    # AI model implementations
│   │   │   ├── services/  # Trading services and pipelines
│   │   │   ├── training/  # Model training scripts
│   │   │   ├── inference/ # Trading inference code
│   │   │   └── evaluation/# Strategy evaluation
│   │   └── resources/     # Configuration and assets
│   │       ├── config/    # Trading configuration files
│   │       └── data/      # Sample/seed data
│   └── test/              # Test code
├── data/                  # Dataset management
│   ├── raw/               # Raw forex data
│   ├── processed/         # Cleaned trading data
│   └── external/          # External data sources
├── notebooks/             # Analysis notebooks
│   ├── exploratory/       # Data exploration
│   ├── experiments/       # Strategy experiments
│   └── reports/           # Trading reports
├── models/                # Trained models
│   ├── trained/           # Production models
│   └── checkpoints/       # Training checkpoints
├── experiments/           # Strategy experiments
│   ├── configs/           # Experiment configs
│   └── results/           # Backtest results
└── output/                # Generated trading outputs
```

## Development Guidelines

- **Always search first** before creating new files
- **Extend existing** functionality rather than duplicating  
- **Use Task agents** for operations >30 seconds
- **Single source of truth** for all functionality
- **Language-agnostic structure** - works with Python, JS, Java, etc.
- **Scalable** - start simple, grow as needed
- **Flexible** - choose complexity level based on project needs

## Trading Strategy Requirements

### Fixed Conditions
- **Trading Instruments**: EUR/USD or USD/JPY
- **Timeframe**: 1-hour (H1)

### Implementation Areas
1. **Data & Feature Engineering** - OHLCV data with technical indicators
2. **AI Model** - ML/DL models for direction prediction  
3. **Strategy Logic** - Combined technical + AI signals
4. **Risk Management** - Fixed % risk, stop-loss/take-profit
5. **Backtesting** - 2-3 years historical data analysis
6. **Performance Evaluation** - Win rate, profit factor, drawdown, Sharpe ratio