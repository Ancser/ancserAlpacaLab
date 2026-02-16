# Ancser Alpaca Lab

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Polars](https://img.shields.io/badge/Polars-Fast-orange)](https://pola.rs/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io/)
[![Alpaca](https://img.shields.io/badge/Alpaca-Trading-yellow)](https://alpaca.markets/)

**High-Performance Quantitative Trading System (Institutional Grade)**

[中文版本](#中文版本) | [English Version](#english-version)

---

# English Version

## Introduction

**ancserAlpacaLab** represents a complete architectural overhaul of the legacy system. It separates "Research (Brain)" from "Execution (Body)" and leverages **Polars** for lightning-fast data processing (10-50x faster than Pandas). The frontend is built with **Streamlit**, providing a modern, interactive dashboard for monitoring strategies and factors.

---

## Key Features

### 1. High-Performance Data Engine

**Polars Core**: Rust-based DataFrame library for memory efficiency and speed.

**Strict Schema**: `schema.py` enforces data types (`Float32`, `Categorical`) to prevent data corruption.

**Unified Adapters**: Seamless data ingestion from **Yahoo Finance** (2015+ historical data) and **Alpaca** (2021+ real-time execution).

**Data Caching**: Automatic Parquet caching provides 24x speedup on repeated queries (0.73s → 0.03s).

### 2. Alpha Factor Library

**Cross-Sectional Focus**: Factors rank stocks against their peers rather than just time-series analysis.

**7 Factor Categories**:

1. **Momentum**
   - Rate of Change (ROC) across multiple timeframes (5/10/20 days)
   - Captures trending stocks with strong recent performance

2. **Reversion**
   - RSI Divergence from 50 (neutral point)
   - Identifies oversold/overbought conditions for mean reversion

3. **Drift-Reversion**
   - Smart reversion that neutralizes in trending markets
   - Uses drift detection to avoid counter-trend losses
   - **Improvement**: Fixed logic bug, reducing max drawdown by 2.77%

4. **Skew**
   - Measures return distribution asymmetry
   - Negative skew = lottery-like payoff preference

5. **Microstructure**
   - Amihud Illiquidity Ratio (price impact per dollar volume)
   - Spread Proxy (High-Low range as transaction cost indicator)

6. **Alpha 101**
   - Mathematical combination of price, volume, and correlation
   - Institutional-grade alpha discovered through research

7. **Volatility**
   - Realized volatility over multiple windows
   - Low-vol anomaly: stable stocks often outperform

**MWU Engine (Multiplicative Weight Update)**:

Dynamic weighting mechanism that adjusts factor exposure based on recent IC (Information Coefficient).

Prevents over-concentration with configurable bounds (min 5%, max 60% per factor).

Tracks IC history for performance analysis and debugging.

### 3. Risk Management

**Volatility Targeting (Constant Risk)**: Automatically adjusts portfolio leverage based on realized volatility (e.g., Target 20% Vol).

**Leverage Control**: Max leverage configurable from 1x to 2x to prevent over-exposure.

**Daily Batch Control**: Designed for robust daily rebalancing, protecting against intra-day noise.

**Idempotent OMS**: Target-based execution ensures safety even if scripts are re-run.

### 4. Interactive Dashboard

**Streamlit App**: Real-time visualization of Equity Curve, P&L, Positions, and Performance Metrics.

**Backtest Engine**: Configure and run high-speed backtests directly from the UI with intuitive controls.

**Factor Presets**: Quick-load top-performing factor combinations from historical analysis.

**Live Strategy Config**: Adjust risk parameters (Target Vol, Max Leverage) on the fly without restarting code.

**Force Execute Button**: Run live strategy execution directly from dashboard with one click.

**Professional Charts**: TradingView-like equity curves with proper range limits, date selectors, and zoom controls.

**Benchmark Comparison**: Compare strategy performance against SPY, QQQ, and GLD with normalized curves.

---

## Quick Start

### 1. Install & Setup

Double-click **`setup.bat`** (or run `pip install -r requirements.txt`).

This will install all required Python packages including Polars, Streamlit, and Alpaca SDK.

### 2. Configuration

Create a `.env` file in the root directory with your Alpaca API credentials:

```env
APCA_API_KEY_ID=your_api_key
APCA_API_SECRET_KEY=your_secret_key
```

**Get Free Alpaca Account**: Register at [https://alpaca.markets/](https://alpaca.markets/) - Free paper trading account with real-time data.

**Note**: Paper trading is recommended for testing. You can enable live trading later by switching the base URL.

### 3. Daily Operation (Safe)

Double-click **`daily_run.bat`**:

- Checks market hours (doesn't trade if closed)
- Launches the Dashboard
- Runs Execution Logic to rebalance portfolio

This is the **recommended** way for automated daily trading.

### 4. Force Execution (Manual)

**Option 1: Command Line**

Double-click **`force_run.bat`**:

- **Ignores market hours** (run anytime)
- **Cancels all open orders** first
- **Forces a rebalance** to target weights immediately
- Shows detailed console output with pause

**Option 2: Dashboard Button**

Open Dashboard → Backtest page → Click **"Force Execute Now"**:

- Same functionality as `force_run.bat`
- View execution output directly in the web interface
- No need to open command line

Use force execution to manually update positions or fix stuck orders.

### 5. Backtesting

**Step 1: Launch Dashboard**

Open Dashboard (`daily_run.bat` or `streamlit run frontend/app.py`).

**Step 2: Navigate to Backtest**

Click "Backtest" in the sidebar navigation.

**Step 3: Configure Parameters**

**Factor Selection**:
- **Manual**: Select individual factors from the multiselect dropdown
- **Presets**: Click preset buttons (Top 1-5) to quick-load best combinations
- Active preset is highlighted in blue

**Date Range**:
- Select Start Year and End Year (auto-generates Jan 1 start date)
- Current year automatically uses today as end date
- Default: 2021 (Alpaca data availability)

**Data Source**:
- **Yahoo (2015+)**: Historical data from Yahoo Finance (free, approximated VWAP)
- **Alpaca (2021+)**: Official IEX feed with true VWAP (requires API keys)

**Universe**:
- **Tech 10**: 10 major tech stocks + SPY/QQQ
- **S&P+Nasdaq**: Full constituent list from `constituents.py`
- Custom: Manually edit the ticker list

**Risk Management**:
- **Initial Capital**: Default $4,000 (adjustable $1k-$1M)
- **Leverage**: 1.0x to 2.0x (default 1.0x)
- **Vol Targeting**: Enable constant risk mode (default: 20% target vol)

**Step 4: Run Backtest**

Click **"Run Backtest"** to execute with selected parameters.

Click **"Run All Combos"** to test all 127 factor combinations (saves top 5 to presets).

**Step 5: Analyze Results**

**Performance Metrics**:
- Total Return, Sharpe Ratio, Max Drawdown, Calmar Ratio
- Win Rate, Average Win/Loss, Profit Factor

**Equity Curve**:
- Interactive chart with benchmark comparison (SPY/QQQ/GLD)
- Date range selectors (1M, 3M, 6M, YTD, 1Y, All)
- Zoom and pan controls

**Factor Weights (if MWU enabled)**:
- Time-series chart showing dynamic factor allocation
- IC history for performance diagnosis

**Step 6: Save Configuration**

Click **"Save Config"** to save current settings to `config/live_strategy.json`.

This configuration will be used by the live execution engine.

---

## Project Structure

```
ancserAlpacaLab/
├── ancser_quant/               # Core System Logic
│   ├── alpha/                  # Factor Library & MWU
│   │   ├── factors.py          # Alpha Factor Definitions
│   │   └── mwu.py              # Dynamic Weighting Engine
│   ├── data/                   # Data Adapters
│   │   ├── alpaca_adapter.py   # Alpaca API Adapter
│   │   ├── yahoo_adapter.py    # Yahoo Finance Adapter
│   │   ├── constituents.py     # Stock Universe
│   │   └── schema.py           # Data Schema
│   ├── execution/              # Execution Engine
│   │   ├── main_loop.py        # Main Execution Loop
│   │   ├── oms.py              # Order Management System
│   │   └── strategy.py         # Strategy Logic
│   └── backtest.py             # Polars Backtest Engine
│
├── frontend/                   # Frontend Dashboard
│   ├── app.py                  # Streamlit Dashboard
│   └── error_logger.py         # Error Logging
│
├── scripts/                    # Utility Scripts
│   ├── run_drift_comparison.py # Factor Comparison Test
│   ├── debug_benchmark.py      # Benchmark Debugging
│   └── README.md               # Script Documentation
│
├── config/                     # Configuration
│   └── live_strategy.json      # Live Strategy Config
│
├── data_cache/                 # Data Cache (Parquet)
├── logs/                       # Application Logs
│
├── .env                        # Environment Variables (API Keys)
├── daily_run.bat               # Daily Entry Point
├── force_run.bat               # Force Execution Entry Point
├── requirements.txt            # Python Dependencies
├── setup.bat                   # Setup Script
└── README.md                   # Project Documentation
```

---

## Recent Improvements

**Fixed Drift-Reversion Factor Logic**: Now correctly applies reversion only in non-drift regimes, reducing max drawdown by ~2.77%

**Enhanced MWU Engine**: Added weight bounds (min/max) to prevent extreme factor allocations and IC history tracking

**Improved Dashboard Charts**: TradingView-like equity curves with proper range limits, date selectors, and zoom controls

**Fixed Benchmark Display**: Resolved SPY/QQQ/GLD alignment issues in backtest charts

**Live Strategy Monitor**: Dashboard now auto-loads and displays current trading logic from saved configurations

**Better Project Organization**: Scripts moved to dedicated folder, clearer separation of concerns

**Automatic Error Logging**: All dashboard errors and warnings automatically saved to `logs/dashboard_YYYY-MM-DD.log`

**Factor Presets with Visual Feedback**: Click preset buttons to quick-load top combinations, active preset highlighted

**Year-Based Date Selection**: Simplified date selection with auto-generated start/end dates

**Data Caching System**: Parquet file caching provides 24x speedup on repeated queries

**Compact 5-Column Layout**: Initial Capital, Date Range, Data Source, Universe, and Risk Management in one row

**Save Config & Force Execute Side-by-Side**: Clear separation of configuration saving and strategy execution

---

## Performance Highlights

**Top Factor Combination** (from All Combo analysis):
- **Momentum + Reversion + Skew + Drift-Reversion**: Calmar Ratio 2.12

**Data Processing Speed**:
- Yahoo Adapter with cache: 0.03s (vs 0.73s without cache, 24x faster)
- Polars engine: 10-50x faster than Pandas for large datasets

---

## Troubleshooting

**Issue**: Dashboard shows "No API keys" warning
**Solution**: Create `.env` file with Alpaca credentials or use Yahoo data source

**Issue**: Backtest returns empty data
**Solution**: Check date range (Yahoo: 2015+, Alpaca: 2021+) and ensure tickers are valid

**Issue**: Preset buttons not updating factors
**Solution**: Ensure you're on the Backtest page and click the preset button, then check the multiselect

**Issue**: Force Execute button not working
**Solution**: Check logs in `logs/` directory for detailed error messages

---

## Disclaimer

This software is for educational and research purposes only. Quantitative trading involves significant financial risk. The authors are not responsible for any financial losses incurred from using this software.

**Always test strategies in paper trading mode before using real capital.**

---

---

# 中文版本

## 簡介

**ancserAlpacaLab** 是對舊有系統的全面重構。它將「研究（大腦）」與「執行（手腳）」完全分離，並利用 **Polars** 進行極速數據處理（比 Pandas 快 10-50 倍）。前端採用 **Streamlit** 構建，提供現代化的交互式儀表板，用於監控策略和因子表現。

---

## 核心功能

### 1. 高性能數據引擎

**Polars 核心引擎**：基於 Rust 的數據框架庫，提供記憶體效率與極速運算。

**嚴格數據架構**：`schema.py` 強制執行數據類型（`Float32`、`Categorical`）以防止數據損壞。

**統一適配器**：無縫對接 **Yahoo Finance**（2015+ 歷史數據）與 **Alpaca**（2021+ 實時執行）。

**數據緩存**：自動 Parquet 緩存使重複查詢速度提升 24 倍（0.73 秒 → 0.03 秒）。

### 2. Alpha 因子庫

**橫截面分析**：因子對股票進行同期排名，而非僅時間序列分析。

**7 大因子類別**：

1. **動量**
   - 多時間框架變化率（5/10/20 日）
   - 捕捉近期表現強勁的趨勢股

2. **均值回歸**
   - RSI 偏離 50（中性點）的程度
   - 識別超賣/超買狀態以進行均值回歸

3. **漂移感知回歸**
   - 智能回歸，在趨勢市場中保持中性
   - 使用漂移檢測以避免逆勢虧損
   - **改進**：修復邏輯錯誤，最大回撤降低 2.77%

4. **偏度**
   - 測量收益分佈的不對稱性
   - 負偏度 = 類彩票收益偏好

5. **微觀結構**
   - Amihud 非流動性比率（每美元成交量的價格影響）
   - 價差代理（高低價範圍作為交易成本指標）

6. **Alpha 101（WorldQuant 風格）**
   - 價格、成交量與相關性的數學組合
   - 通過研究發現的機構級 Alpha

7. **波動率**
   - 多窗口期實現波動率
   - 低波動率異象：穩定股票往往表現更佳

**MWU 引擎（乘法權重更新）**：

動態權重機制，根據近期 IC（信息係數）調整因子暴露。

通過可配置邊界（每因子最小 5%、最大 60%）防止過度集中。

追蹤 IC 歷史記錄以進行性能分析與調試。

### 3. 風險管理

**波動率目標（恆定風險）**：根據實現波動率自動調整組合槓桿（例如目標 20% 波動率）。

**槓桿控制**：最大槓桿可配置為 1 倍至 2 倍，防止過度暴露。

**每日批量控制**：專為穩健的每日再平衡設計，抵禦盤中噪音。

**冪等訂單管理系統**：基於目標的執行確保即使重複運行腳本也安全。

### 4. 交互式儀表板

**Streamlit 應用**：實時可視化權益曲線、損益、持倉與績效指標。

**回測引擎**：通過直觀控制界面直接配置與運行高速回測。

**因子預設**：快速加載歷史分析中表現最佳的因子組合。

**實時策略配置**：無需重啟代碼即可動態調整風險參數（目標波動率、最大槓桿）。

**強制執行按鈕**：通過儀表板一鍵運行實時策略執行。

**專業圖表**：類似 TradingView 的權益曲線，具備適當範圍限制、日期選擇器與縮放控制。

**基準比較**：將策略表現與 SPY、QQQ、GLD 進行標準化曲線比較。

---

## 快速啟動

### 1. 安裝與設置

雙擊 **`setup.bat`**（或執行 `pip install -r requirements.txt`）。

這將安裝所有必需的 Python 套件，包括 Polars、Streamlit 與 Alpaca SDK。

### 2. 環境配置

在根目錄創建 `.env` 檔案，填入您的 Alpaca API 憑證：

```env
APCA_API_KEY_ID=your_api_key
APCA_API_SECRET_KEY=your_secret_key
```

**獲取免費 Alpaca 帳戶**：在 [https://alpaca.markets/](https://alpaca.markets/) 註冊 - 免費模擬交易帳戶，提供實時數據。

**注意**：建議使用模擬交易進行測試。稍後可通過切換基礎 URL 啟用實盤交易。

### 3. 日常操作（安全）

雙擊 **`daily_run.bat`**：

- 檢查市場時段（收盤時不交易）
- 啟動儀表板
- 運行執行邏輯以再平衡組合

這是**推薦**的自動化每日交易方式。

### 4. 強制執行（手動）

**選項 1：命令行**

雙擊 **`force_run.bat`**：

- **忽略市場時段**（隨時運行）
- **首先取消所有未成交訂單**
- **強制再平衡**至目標權重
- 顯示詳細控制台輸出並暫停

**選項 2：儀表板按鈕**

打開儀表板 → 回測頁面 → 點擊 **"強制執行"**：

- 與 `force_run.bat` 功能相同
- 直接在網頁界面查看執行輸出
- 無需打開命令行

使用強制執行手動更新倉位或修復卡住的訂單。

### 5. 回測

**步驟 1：啟動儀表板**

打開儀表板（`daily_run.bat` 或 `streamlit run frontend/app.py`）。

**步驟 2：導航至回測**

在側邊欄導航中點擊「Backtest」。

**步驟 3：配置參數**

**因子選擇**：
- **手動選擇**：從多選下拉菜單中選擇個別因子
- **預設組合**：點擊預設按鈕（Top 1-5）快速加載最佳組合
- 啟用的預設以藍色高亮顯示

**日期範圍**：
- 選擇起始年份與結束年份（自動生成 1 月 1 日起始日期）
- 當前年份自動使用今日作為結束日期
- 默認：2021（Alpaca 數據可用性）

**數據來源**：
- **Yahoo（2015+）**：來自 Yahoo Finance 的歷史數據（免費，近似 VWAP）
- **Alpaca（2021+）**：官方 IEX 數據源，真實 VWAP（需要 API 密鑰）

**股票池**：
- **Tech 10**：10 支主要科技股 + SPY/QQQ
- **S&P+Nasdaq**：來自 `constituents.py` 的完整成分股列表
- 自定義：手動編輯股票代碼列表

**風險管理**：
- **初始資金**：默認 $4,000（可調整 $1k-$1M）
- **槓桿**：1.0 倍至 2.0 倍（默認 1.0 倍）
- **波動率目標**：啟用恆定風險模式（默認：20% 目標波動率）

**步驟 4：運行回測**

點擊 **"運行回測"** 以執行所選參數。

點擊 **"運行所有組合"** 以測試所有 127 種因子組合（將前 5 名保存到預設）。

**步驟 5：分析結果**

**績效指標**：
- 總收益、夏普比率、最大回撤、Calmar 比率
- 勝率、平均盈虧、盈利因子

**權益曲線**：
- 交互式圖表，包含基準比較（SPY/QQQ/GLD）
- 日期範圍選擇器（1 月、3 月、6 月、年初至今、1 年、全部）
- 縮放與平移控制

**因子權重（如啟用 MWU）**：
- 時間序列圖表顯示動態因子配置
- IC 歷史記錄用於性能診斷

**步驟 6：保存配置**

點擊 **"保存配置"** 將當前設置保存至 `config/live_strategy.json`。

此配置將被實時執行引擎使用。

---

## 專案結構

```
ancserAlpacaLab/
├── ancser_quant/               # 核心系統邏輯
│   ├── alpha/                  # 因子庫與 MWU
│   │   ├── factors.py          # Alpha 因子定義
│   │   └── mwu.py              # 動態權重更新引擎
│   ├── data/                   # 數據適配器
│   │   ├── alpaca_adapter.py   # Alpaca API 適配器
│   │   ├── yahoo_adapter.py    # Yahoo Finance 適配器
│   │   ├── constituents.py     # 股票池定義
│   │   └── schema.py           # 數據架構定義
│   ├── execution/              # 執行引擎
│   │   ├── main_loop.py        # 主執行循環
│   │   ├── oms.py              # 訂單管理系統
│   │   └── strategy.py         # 策略邏輯
│   └── backtest.py             # Polars 回測引擎
│
├── frontend/                   # 前端儀表板
│   ├── app.py                  # Streamlit 儀表板
│   └── error_logger.py         # 錯誤日誌
│
├── scripts/                    # 工具腳本
│   ├── run_drift_comparison.py # 因子對比測試
│   ├── debug_benchmark.py      # 基準調試
│   └── README.md               # 腳本說明
│
├── config/                     # 配置文件
│   └── live_strategy.json      # 實時策略配置
│
├── data_cache/                 # 數據緩存（Parquet）
├── logs/                       # 應用日誌
│
├── .env                        # 環境變量（API 密鑰）
├── daily_run.bat               # 每日運行入口
├── force_run.bat               # 強制執行入口
├── requirements.txt            # Python 依賴
├── setup.bat                   # 環境設置腳本
└── README.md                   # 項目說明
```

---

## 最新改進

**修復 Drift-Reversion 因子邏輯**：現在正確地僅在非漂移狀態下應用回歸，最大回撤降低約 2.77%

**增強 MWU 引擎**：添加權重邊界（最小/最大）以防止極端因子配置，並追蹤 IC 歷史

**改進儀表板圖表**：類 TradingView 權益曲線，具備適當範圍限制、日期選擇器與縮放控制

**修復基準顯示**：解決回測圖表中 SPY/QQQ/GLD 對齊問題

**實時策略監控**：儀表板現在自動加載並顯示來自保存配置的當前交易邏輯

**更好的項目組織**：腳本移至專用文件夾，更清晰的關注點分離

**自動錯誤日誌**：所有儀表板錯誤與警告自動保存至 `logs/dashboard_YYYY-MM-DD.log`

**帶視覺反饋的因子預設**：點擊預設按鈕快速加載頂級組合，啟用預設高亮顯示

**基於年份的日期選擇**：簡化日期選擇，自動生成起始/結束日期

**數據緩存系統**：Parquet 文件緩存使重複查詢速度提升 24 倍

**緊湊 5 列佈局**：初始資金、日期範圍、數據來源、股票池與風險管理在一行

**並列保存配置與強制執行**：配置保存與策略執行清晰分離

---

## 性能亮點

**頂級因子組合**（來自全組合分析）：
- **動量 + 均值回歸 + 偏度 + 漂移感知回歸**：Calmar 比率 2.12

**數據處理速度**：
- 帶緩存的 Yahoo 適配器：0.03 秒（vs 無緩存 0.73 秒，快 24 倍）
- Polars 引擎：大數據集比 Pandas 快 10-50 倍

---

## 故障排除

**問題**：儀表板顯示「無 API 密鑰」警告
**解決方案**：創建包含 Alpaca 憑證的 `.env` 檔案或使用 Yahoo 數據來源

**問題**：回測返回空數據
**解決方案**：檢查日期範圍（Yahoo：2015+，Alpaca：2021+）並確保股票代碼有效

**問題**：預設按鈕未更新因子
**解決方案**：確保您在回測頁面並點擊預設按鈕，然後檢查多選框

**問題**：強制執行按鈕不工作
**解決方案**：檢查 `logs/` 目錄中的日誌以獲取詳細錯誤消息

---

## 免責聲明

本軟體僅供教育與研究用途。量化交易涉及重大財務風險。作者不對使用本軟體造成的任何財務損失負責。

**在使用真實資金之前，請務必在模擬交易模式下測試策略。**
