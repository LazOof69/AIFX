# Data Directory | 數據目錄

This directory contains all datasets used by the AIFX trading system.  
此目錄包含AIFX交易系統使用的所有數據集。

## Structure | 結構

- **raw/ | 原始數據**: Original, unprocessed forex data from various sources | 來自各種來源的原始、未處理外匯數據
- **processed/ | 處理後數據**: Cleaned and transformed data ready for analysis | 已清理和轉換的數據，可用於分析
- **external/ | 外部數據**: External data sources and reference datasets | 外部數據源和參考數據集

## Data Sources | 數據來源

- EUR/USD 1-hour OHLCV data | 歐元/美元 1小時 OHLCV數據
- USD/JPY 1-hour OHLCV data | 美元/日圓 1小時 OHLCV數據
- Economic indicators and news data | 經濟指標和新聞數據
- Market volatility and volume data | 市場波動率和交易量數據

## Usage Guidelines | 使用指引

1. Always store raw data in `raw/` directory | 始終將原始數據存儲在 `raw/` 目錄中
2. Process and clean data before storing in `processed/` | 在存儲到 `processed/` 之前處理和清理數據
3. Use external data for feature enrichment | 使用外部數據進行特徵豐富化
4. Maintain data versioning and documentation | 維護數據版本控制和文件記錄