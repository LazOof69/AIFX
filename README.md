# AIFX | 人工智能外匯交易系統

## Quick Start | 快速開始

1. **Read CLAUDE.md first** - Contains essential rules for Claude Code | **先閱讀 CLAUDE.md** - 包含 Claude Code 的重要規則
2. Follow the pre-task compliance checklist before starting any work | 在開始任何工作前遵循預任務合規檢查清單
3. Use proper module structure under `src/main/python/` | 在 `src/main/python/` 下使用適當的模組結構
4. Commit after every completed task | 每個完成的任務後進行提交

## Project Overview | 專案概述

**AIFX** is a professional quantitative trading system that implements medium-term forex strategies enhanced with AI models. The system focuses on EUR/USD and USD/JPY trading pairs using 1-hour timeframes.  
**AIFX** 是一個專業的量化交易系統，實現結合AI模型的中期外匯策略。系統專注於歐元/美元和美元/日圓貨幣對，使用1小時時間框架。

## 🎯 Development Status | 開發狀態

```
Phase 1: Infrastructure    ████████████████████ 100% ✅ COMPLETED
Phase 2: AI Models         ████████████████████ 100% ✅ COMPLETED  
Phase 3: Strategy          ░░░░░░░░░░░░░░░░░░░░   0% 🔄 NEXT
Phase 4: Production        ░░░░░░░░░░░░░░░░░░░░   0% ⏳ PLANNED
```

**Latest Achievement: Phase 2 AI Model Development Completed**  
**最新成就：第二階段AI模型開發已完成**

### Key Features | 主要功能

- **Data & Feature Engineering | 數據與特徵工程**: ✅ Historical OHLCV data with technical indicators (MA, MACD, RSI, Bollinger Bands, ATR) | 歷史OHLCV數據配合技術指標（移動平均線、MACD、RSI、布林帶、ATR）
- **AI Models | AI模型**: ✅ **IMPLEMENTED** - XGBoost, Random Forest, LSTM with training pipeline | **已實現** - XGBoost、隨機森林、LSTM配合訓練管道
- **Strategy Logic | 策略邏輯**: 🔄 Combined technical and AI signals with confidence filtering | 結合技術和AI信號並進行信心過濾
- **Risk Management | 風險管理**: 🔄 Fixed percentage risk, stop-loss/take-profit using ATR multiples | 固定百分比風險、使用ATR倍數的止損/止盈
- **Backtesting | 回測**: 🔄 Backtrader framework with comprehensive performance metrics | Backtrader框架配合全面的績效指標

## 🤖 AI Model Components (Phase 2 - Completed) | AI模型組件（第二階段 - 已完成）

### **Implemented Models | 已實現模型**
1. **XGBoost Classifier** - Gradient boosting with hyperparameter optimization | 梯度提升配合超參數優化
   - Path: `src/main/python/models/xgboost_model.py`
   - Features: GridSearch, feature importance, cross-validation | 網格搜索、特徵重要性、交叉驗證

2. **Random Forest Ensemble** - Bootstrap aggregating with tree diversity analysis | 自助聚合配合樹多樣性分析
   - Path: `src/main/python/models/random_forest_model.py`
   - Features: OOB scoring, ensemble statistics, learning curves | OOB評分、集成統計、學習曲線

3. **LSTM Neural Network** - Deep learning for time series prediction | 時間序列預測的深度學習
   - Path: `src/main/python/models/lstm_model.py`
   - Features: TensorFlow/Keras, sequence modeling, early stopping | TensorFlow/Keras、序列建模、早期停止

### **Supporting Infrastructure | 支援基礎設施**
- **Base Model Framework** (`base_model.py`) - Abstract classes and model registry | 抽象類和模型註冊表
- **Training Pipeline** (`training/model_pipeline.py`) - Multi-model training and comparison | 多模型訓練和比較
- **Performance Metrics** (`evaluation/performance_metrics.py`) - Trading-specific evaluation | 交易特定評估
- **Model Management** (`services/model_manager.py`) - Versioning and deployment | 版本控制和部署

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
- **Documentation compliance** - Update UPDATE.log and check README.md after every change | **文件合規性** - 每次更改後更新UPDATE.log並檢查README.md
- **Single source of truth** for all functionality | 所有功能保持 **單一真實來源**
- **Language-agnostic structure** - works with Python, JS, Java, etc. | **語言無關結構** - 支援 Python、JS、Java 等
- **Scalable** - start simple, grow as needed | **可擴展** - 從簡單開始，按需增長
- **Flexible** - choose complexity level based on project needs | **靈活性** - 根據專案需求選擇複雜度級別

### 📝 Documentation Standards | 文件標準
- **UPDATE.log**: Mandatory milestone tracking for all significant changes | 所有重大更改的強制里程碑追蹤
- **README.md**: Must reflect current project status and capabilities | 必須反映當前專案狀態和功能  
- **CHANGELOG.md**: Semantic versioning with comprehensive release notes | 語義化版本控制配合全面發布說明
- **Bilingual**: All documentation in English and Traditional Chinese | 所有文件均為英文和繁體中文雙語

## 🚀 Installation & Setup | 安裝與設置

### **Prerequisites | 前置條件**
- Python 3.8+ (tested with 3.12) | Python 3.8+（已在3.12測試）
- Git for version control | Git用於版本控制

### **Installation Steps | 安裝步驟**

1. **Clone repository | 克隆儲存庫**
   ```bash
   git clone https://github.com/LazOof69/AIFX.git
   cd AIFX
   ```

2. **Create virtual environment | 創建虛擬環境**
   ```bash
   python -m venv aifx-venv
   source aifx-venv/bin/activate  # Linux/Mac
   # or
   aifx-venv\Scripts\activate     # Windows
   ```

3. **Install dependencies | 安裝依賴項**
   ```bash
   pip install -r requirements.txt
   ```

### **Key Dependencies Added in Phase 2 | 第二階段新增的關鍵依賴項**
- `tensorflow>=2.10.0` - Deep learning framework for LSTM | LSTM的深度學習框架
- `keras>=2.10.0` - High-level neural networks API | 高級神經網絡API
- `xgboost>=1.6.0` - Gradient boosting framework | 梯度提升框架
- `joblib>=1.1.0` - Model serialization | 模型序列化

## 💡 Usage Examples | 使用範例

### **Basic Model Training | 基礎模型訓練**

```python
from src.main.python.training.model_pipeline import ModelTrainingPipeline
from src.main.python.models import XGBoostModel, RandomForestModel, LSTMModel

# Initialize training pipeline | 初始化訓練管道
pipeline = ModelTrainingPipeline()

# Prepare your forex data | 準備外匯數據
# data = load_forex_data()  # Your data loading logic

# Train multiple models | 訓練多個模型
results = pipeline.train_multiple_models(
    data_splits=data_splits,
    model_types=['xgboost', 'random_forest', 'lstm'],
    optimize_hyperparameters=True
)

# Compare model performance | 比較模型性能
best_model = results['best_model']
print(f"Best performing model: {best_model['model_type']}")
```

### **Individual Model Usage | 單個模型使用**

```python
# XGBoost Example | XGBoost示例
from src.main.python.models.xgboost_model import XGBoostModel

model = XGBoostModel()
training_history = model.train(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Get feature importance | 獲取特徵重要性
importance = model.get_feature_importance()
model.plot_feature_importance(top_n=20)
```

## Trading Strategy Requirements | 交易策略需求

### Fixed Conditions | 固定條件
- **Trading Instruments | 交易工具**: EUR/USD or USD/JPY | 歐元/美元 或 美元/日圓
- **Timeframe | 時間框架**: 1-hour (H1) | 1小時 (H1)

### Implementation Areas | 實施領域
1. **Data & Feature Engineering | 數據與特徵工程** - ✅ COMPLETED | 已完成
2. **AI Model | AI模型** - ✅ COMPLETED | 已完成  
3. **Strategy Logic | 策略邏輯** - 🔄 Phase 3 Target | 第三階段目標
4. **Risk Management | 風險管理** - 🔄 Phase 3 Target | 第三階段目標
5. **Backtesting | 回測** - 🔄 Phase 3 Target | 第三階段目標
6. **Performance Evaluation | 績效評估** - ✅ Framework Ready | 框架就緒