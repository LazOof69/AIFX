# AIFX | 人工智能外匯交易系統

## Quick Start | 快速開始

1. **Read CLAUDE.md first** - Contains essential rules for Claude Code | **先閱讀 CLAUDE.md** - 包含 Claude Code 的重要規則
2. Follow the pre-task compliance checklist before starting any work | 在開始任何工作前遵循預任務合規檢查清單
3. Use proper module structure under `src/main/python/` | 在 `src/main/python/` 下使用適當的模組結構
4. Commit after every completed task | 每個完成的任務後進行提交

## Project Overview | 專案概述

**AIFX** is a professional quantitative trading system that implements medium-term forex strategies enhanced with AI models. The system focuses on EUR/USD and USD/JPY trading pairs using 1-hour timeframes.  
**AIFX** 是一個專業的量化交易系統，實現結合AI模型的中期外匯策略。系統專注於歐元/美元和美元/日圓貨幣對，使用1小時時間框架。

### Key Features | 主要功能

- **Data & Feature Engineering | 數據與特徵工程**: Historical OHLCV data with technical indicators (MA, MACD, RSI, Bollinger Bands, ATR) | 歷史OHLCV數據配合技術指標（移動平均線、MACD、RSI、布林帶、ATR）
- **AI Models | AI模型**: Machine learning models (XGBoost, Random Forest, LSTM) for price direction prediction | 機器學習模型（XGBoost、隨機森林、LSTM）用於價格方向預測
- **Strategy Logic | 策略邏輯**: Combined technical and AI signals with confidence filtering | 結合技術和AI信號並進行信心過濾
- **Risk Management | 風險管理**: Fixed percentage risk, stop-loss/take-profit using ATR multiples | 固定百分比風險、使用ATR倍數的止損/止盈
- **Backtesting | 回測**: Backtrader framework with comprehensive performance metrics | Backtrader框架配合全面的績效指標

## AI/ML Project Structure | AI/ML 專案結構

**Complete MLOps-ready structure with data, models, experiments**  
**完整的 MLOps 就緒結構，包含數據、模型和實驗**

```
AIFX/
├── CLAUDE.md              # Essential rules for Claude Code | Claude Code 重要規則
├── src/                   # Source code (NEVER put files in root) | 源代碼（絕不在根目錄放文件）
│   ├── main/              # Main application code | 主要應用程式代碼
│   │   ├── python/        # Python implementation | Python 實現
│   │   │   ├── core/      # Core trading algorithms | 核心交易演算法
│   │   │   ├── utils/     # Data processing utilities | 數據處理工具
│   │   │   ├── models/    # AI model implementations | AI模型實現
│   │   │   ├── services/  # Trading services and pipelines | 交易服務與管道
│   │   │   ├── training/  # Model training scripts | 模型訓練腳本
│   │   │   ├── inference/ # Trading inference code | 交易推理代碼
│   │   │   └── evaluation/# Strategy evaluation | 策略評估
│   │   └── resources/     # Configuration and assets | 配置與資產
│   │       ├── config/    # Trading configuration files | 交易配置文件
│   │       └── data/      # Sample/seed data | 樣本/種子數據
│   └── test/              # Test code | 測試代碼
├── data/                  # Dataset management | 數據集管理
│   ├── raw/               # Raw forex data | 原始外匯數據
│   ├── processed/         # Cleaned trading data | 清理後的交易數據
│   └── external/          # External data sources | 外部數據源
├── notebooks/             # Analysis notebooks | 分析筆記本
│   ├── exploratory/       # Data exploration | 數據探索
│   ├── experiments/       # Strategy experiments | 策略實驗
│   └── reports/           # Trading reports | 交易報告
├── models/                # Trained models | 訓練好的模型
│   ├── trained/           # Production models | 生產模型
│   └── checkpoints/       # Training checkpoints | 訓練檢查點
├── experiments/           # Strategy experiments | 策略實驗
│   ├── configs/           # Experiment configs | 實驗配置
│   └── results/           # Backtest results | 回測結果
└── output/                # Generated trading outputs | 生成的交易輸出
```

## Development Guidelines | 開發指引

- **Always search first** before creating new files | **創建新文件前先搜索** 現有實現
- **Extend existing** functionality rather than duplicating | **擴展現有** 功能而非重複開發
- **Use Task agents** for operations >30 seconds | **使用任務代理** 處理超過30秒的操作
- **Single source of truth** for all functionality | 所有功能保持 **單一真實來源**
- **Language-agnostic structure** - works with Python, JS, Java, etc. | **語言無關結構** - 支援 Python、JS、Java 等
- **Scalable** - start simple, grow as needed | **可擴展** - 從簡單開始，按需增長
- **Flexible** - choose complexity level based on project needs | **靈活性** - 根據專案需求選擇複雜度級別

## Trading Strategy Requirements | 交易策略需求

### Fixed Conditions | 固定條件
- **Trading Instruments | 交易工具**: EUR/USD or USD/JPY | 歐元/美元 或 美元/日圓
- **Timeframe | 時間框架**: 1-hour (H1) | 1小時 (H1)

### Implementation Areas | 實施領域
1. **Data & Feature Engineering | 數據與特徵工程** - OHLCV data with technical indicators | OHLCV數據配合技術指標
2. **AI Model | AI模型** - ML/DL models for direction prediction | ML/DL模型用於方向預測
3. **Strategy Logic | 策略邏輯** - Combined technical + AI signals | 結合技術+AI信號
4. **Risk Management | 風險管理** - Fixed % risk, stop-loss/take-profit | 固定%風險、止損/止盈
5. **Backtesting | 回測** - 2-3 years historical data analysis | 2-3年歷史數據分析
6. **Performance Evaluation | 績效評估** - Win rate, profit factor, drawdown, Sharpe ratio | 勝率、盈利因子、回撤、夏普比率