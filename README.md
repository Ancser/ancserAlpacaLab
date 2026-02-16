# Ancser Alpaca Lab

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Polars](https://img.shields.io/badge/Polars-Fast-orange)](https://pola.rs/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io/)
[![Alpaca](https://img.shields.io/badge/Alpaca-Trading-yellow)](https://alpaca.markets/)

**Quantitative Trading System**

**量化交易系統**

---

## Introduction / 簡介

**ancserAlpacaLab** 因子/回測/策略/執行
使用Alpaca平臺自動交易，需要每天設置電腦開啓daily_run.bat脚本下單。包含因子回測（未最佳化）、訂單執行。
由Google Antigravity / Claude Code / ancser監督聯合開發。
---

## Key Features / 核心功能

### 1. High-Performance Data Engine

### 高性能數據引擎

**Polars Core**: Rust-based DataFrame library for memory efficiency and speed.

**Polars 核心引擎**：基於 Rust 的數據框架庫，提供記憶體效率與極速運算。

**Strict Schema**: `schema.py` enforces data types (`Float32`, `Categorical`) to prevent data corruption.

**嚴格數據架構**：`schema.py` 強制執行數據類型（`Float32`、`Categorical`）以防止數據損壞。

**Unified Adapters**: Seamless data ingestion from **Yahoo Finance** (2015+ historical data) and **Alpaca** (2021+ real-time execution).

**統一適配器**：無縫對接 **Yahoo Finance**（2015+ 歷史數據）與 **Alpaca**（2021+ 實時執行）。

**Data Caching**: Automatic Parquet caching provides 24x speedup on repeated queries (0.73s → 0.03s).

**數據緩存**：自動 Parquet 緩存使重複查詢速度提升 24 倍（0.73 秒 → 0.03 秒）。

### 2. Alpha Factor Library

### Alpha 因子庫

**Cross-Sectional Focus**: Factors rank stocks against their peers rather than just time-series analysis.

**橫截面分析**：因子對股票進行同期排名，而非僅時間序列分析。

**7 Factor Categories**:

**7 大因子類別**：

1. **Momentum (動量)**
   - Rate of Change (ROC) across multiple timeframes (5/10/20 days)
   - 多時間框架變化率（5/10/20 日）
   - Captures trending stocks with strong recent performance
   - 捕捉近期表現強勁的趨勢股

2. **Reversion (均值回歸)**
   - RSI Divergence from 50 (neutral point)
   - RSI 偏離 50（中性點）的程度
   - Identifies oversold/overbought conditions for mean reversion
   - 識別超賣/超買狀態以進行均值回歸

3. **Drift-Reversion (漂移感知回歸)**
   - Smart reversion that neutralizes in trending markets
   - 智能回歸，在趨勢市場中保持中性
   - Uses drift detection to avoid counter-trend losses
   - 使用漂移檢測以避免逆勢虧損
   - **Improvement**: Fixed logic bug, reducing max drawdown by 2.77%
   - **改進**：修復邏輯錯誤，最大回撤降低 2.77%

4. **Skew (偏度)**
   - Measures return distribution asymmetry
   - 測量收益分佈的不對稱性
   - Negative skew = lottery-like payoff preference
   - 負偏度 = 類彩票收益偏好

5. **Microstructure (微觀結構)**
   - Amihud Illiquidity Ratio (price impact per dollar volume)
   - Amihud 非流動性比率（每美元成交量的價格影響）
   - Spread Proxy (High-Low range as transaction cost indicator)
   - 價差代理（高低價範圍作為交易成本指標）

6. **Alpha 101 (WorldQuant 風格)**
   - Mathematical combination of price, volume, and correlation
   - 價格、成交量與相關性的數學組合
   - Institutional-grade alpha discovered through research
   - 通過研究發現的機構級 Alpha

7. **Volatility (波動率)**
   - Realized volatility over multiple windows
   - 多窗口期實現波動率
   - Low-vol anomaly: stable stocks often outperform
   - 低波動率異象：穩定股票往往表現更佳

**MWU Engine (Multiplicative Weight Update)**:

**MWU 引擎（乘法權重更新）**：

Dynamic weighting mechanism that adjusts factor exposure based on recent IC (Information Coefficient).

動態權重機制，根據近期 IC（信息係數）調整因子暴露。

Prevents over-concentration with configurable bounds (min 5%, max 60% per factor).

通過可配置邊界（每因子最小 5%、最大 60%）防止過度集中。

Tracks IC history for performance analysis and debugging.

追蹤 IC 歷史記錄以進行性能分析與調試。

### 3. Risk Management

### 風險管理

**Volatility Targeting (Constant Risk)**: Automatically adjusts portfolio leverage based on realized volatility (e.g., Target 20% Vol).

**波動率目標（恆定風險）**：根據實現波動率自動調整組合槓桿（例如目標 20% 波動率）。

**Leverage Control**: Max leverage configurable from 1x to 2x to prevent over-exposure.

**槓桿控制**：最大槓桿可配置為 1 倍至 2 倍，防止過度暴露。

**Daily Batch Control**: Designed for robust daily rebalancing, protecting against intra-day noise.

**每日批量控制**：專為穩健的每日再平衡設計，抵禦盤中噪音。

**Idempotent OMS**: Target-based execution ensures safety even if scripts are re-run.

**冪等訂單管理系統**：基於目標的執行確保即使重複運行腳本也安全。

### 4. Interactive Dashboard

### 交互式儀表板

**Streamlit App**: Real-time visualization of Equity Curve, P&L, Positions, and Performance Metrics.

**Streamlit 應用**：實時可視化權益曲線、損益、持倉與績效指標。

**Backtest Engine**: Configure and run high-speed backtests directly from the UI with intuitive controls.

**回測引擎**：通過直觀控制界面直接配置與運行高速回測。

**Factor Presets**: Quick-load top-performing factor combinations from historical analysis.

**因子預設**：快速加載歷史分析中表現最佳的因子組合。

**Live Strategy Config**: Adjust risk parameters (Target Vol, Max Leverage) on the fly without restarting code.

**實時策略配置**：無需重啟代碼即可動態調整風險參數（目標波動率、最大槓桿）。

**Force Execute Button**: Run live strategy execution directly from dashboard with one click.

**強制執行按鈕**：通過儀表板一鍵運行實時策略執行。

**Professional Charts**: TradingView-like equity curves with proper range limits, date selectors, and zoom controls.

**專業圖表**：類似 TradingView 的權益曲線，具備適當範圍限制、日期選擇器與縮放控制。

**Benchmark Comparison**: Compare strategy performance against SPY, QQQ, and GLD with normalized curves.

**基準比較**：將策略表現與 SPY、QQQ、GLD 進行標準化曲線比較。

---

## Quick Start / 快速啟動

### 1. Install & Setup

### 安裝與設置

Double-click **`setup.bat`** (or run `pip install -r requirements.txt`).

雙擊 **`setup.bat`**（或執行 `pip install -r requirements.txt`）。

This will install all required Python packages including Polars, Streamlit, and Alpaca SDK.

這將安裝所有必需的 Python 套件，包括 Polars、Streamlit 與 Alpaca SDK。

### 2. Configuration / Environment Setup

### 設定 / 環境配置

Create a `.env` file in the root directory with your Alpaca API credentials:

在根目錄創建 `.env` 檔案，填入您的 Alpaca API 憑證：

```env
APCA_API_KEY_ID=your_api_key
APCA_API_SECRET_KEY=your_secret_key
```

**Get Free Alpaca Account** / **獲取免費 Alpaca 帳戶**:

Register at [https://alpaca.markets/](https://alpaca.markets/) - Free paper trading account with real-time data.

在 [https://alpaca.markets/](https://alpaca.markets/) 註冊 - 免費模擬交易帳戶，提供實時數據。

**Note**: Paper trading is recommended for testing. You can enable live trading later by switching the base URL.

**注意**：建議使用模擬交易進行測試。稍後可通過切換基礎 URL 啟用實盤交易。

### 3. Daily Operation (Safe)

### 日常操作（安全）

Double-click **`daily_run.bat`**:

雙擊 **`daily_run.bat`**：

- Checks market hours (doesn't trade if closed)
- 檢查市場時段（收盤時不交易）
- Launches the **Dashboard**
- 啟動**儀表板**
- Runs **Execution Logic** to rebalance portfolio
- 運行**執行邏輯**以再平衡組合

This is the **recommended** way for automated daily trading.

這是**推薦**的自動化每日交易方式。

### 4. Force Execution (Manual)

### 強制執行（手動）

**Option 1: Command Line** / **選項 1：命令行**

Double-click **`force_run.bat`**:

雙擊 **`force_run.bat`**：

- **Ignores market hours** (run anytime)
- **忽略市場時段**（隨時運行）
- **Cancels all open orders** first
- **首先取消所有未成交訂單**
- **Forces a rebalance** to target weights immediately
- **強制再平衡**至目標權重
- Shows detailed console output with pause
- 顯示詳細控制台輸出並暫停

**Option 2: Dashboard Button** / **選項 2：儀表板按鈕**

Open Dashboard → Backtest page → Click **"⚡ Force Execute Now"**:

打開儀表板 → 回測頁面 → 點擊 **"⚡ 強制執行"**：

- Same functionality as `force_run.bat`
- 與 `force_run.bat` 功能相同
- View execution output directly in the web interface
- 直接在網頁界面查看執行輸出
- No need to open command line
- 無需打開命令行

Use force execution to manually update positions or fix stuck orders.

使用強制執行手動更新倉位或修復卡住的訂單。

### 5. Backtesting

### 回測

**Step 1: Launch Dashboard** / **步驟 1：啟動儀表板**

Open Dashboard (`daily_run.bat` or `streamlit run frontend/app.py`).

打開儀表板（`daily_run.bat` 或 `streamlit run frontend/app.py`）。

**Step 2: Navigate to Backtest** / **步驟 2：導航至回測**

Click "Backtest" in the sidebar navigation.

在側邊欄導航中點擊「Backtest」。

**Step 3: Configure Parameters** / **步驟 3：配置參數**

**Factor Selection** / **因子選擇**:

- **Manual**: Select individual factors from the multiselect dropdown
- **手動選擇**：從多選下拉菜單中選擇個別因子
- **Presets**: Click preset buttons (Top 1-5) to quick-load best combinations
- **預設組合**：點擊預設按鈕（Top 1-5）快速加載最佳組合
- Active preset is highlighted in blue
- 啟用的預設以藍色高亮顯示

**Date Range** / **日期範圍**:

- Select Start Year and End Year (auto-generates Jan 1 start date)
- 選擇起始年份與結束年份（自動生成 1 月 1 日起始日期）
- Current year automatically uses today as end date
- 當前年份自動使用今日作為結束日期
- Default: 2021 (Alpaca data availability)
- 默認：2021（Alpaca 數據可用性）

**Data Source** / **數據來源**:

- **Yahoo (2015+)**: Historical data from Yahoo Finance (free, approximated VWAP)
- **Yahoo（2015+）**：來自 Yahoo Finance 的歷史數據（免費，近似 VWAP）
- **Alpaca (2021+)**: Official IEX feed with true VWAP (requires API keys)
- **Alpaca（2021+）**：官方 IEX 數據源，真實 VWAP（需要 API 密鑰）

**Universe** / **股票池**:

- **Tech 10**: 10 major tech stocks + SPY/QQQ
- **Tech 10**：10 支主要科技股 + SPY/QQQ
- **S&P+Nasdaq**: Full constituent list from `constituents.py`
- **S&P+Nasdaq**：來自 `constituents.py` 的完整成分股列表
- Custom: Manually edit the ticker list
- 自定義：手動編輯股票代碼列表

**Risk Management** / **風險管理**:

- **Initial Capital**: Default $4,000 (adjustable $1k-$1M)
- **初始資金**：默認 $4,000（可調整 $1k-$1M）
- **Leverage**: 1.0x to 2.0x (default 1.0x)
- **槓桿**：1.0 倍至 2.0 倍（默認 1.0 倍）
- **Vol Targeting**: Enable constant risk mode (default: 20% target vol)
- **波動率目標**：啟用恆定風險模式（默認：20% 目標波動率）

**Step 4: Run Backtest** / **步驟 4：運行回測**

Click **"▶️ Run Backtest"** to execute with selected parameters.

點擊 **"▶️ 運行回測"** 以執行所選參數。

Click **"🔄 Run All Combos"** to test all 127 factor combinations (saves top 5 to presets).

點擊 **"🔄  運行所有組合"** 以測試所有 127 種因子組合（將前 5 名保存到預設）。

**Step 5: Analyze Results** / **步驟 5：分析結果**

**Performance Metrics** / **績效指標**:

- Total Return, Sharpe Ratio, Max Drawdown, Calmar Ratio
- 總收益、夏普比率、最大回撤、Calmar 比率
- Win Rate, Average Win/Loss, Profit Factor
- 勝率、平均盈虧、盈利因子

**Equity Curve** / **權益曲線**:

- Interactive chart with benchmark comparison (SPY/QQQ/GLD)
- 交互式圖表，包含基準比較（SPY/QQQ/GLD）
- Date range selectors (1M, 3M, 6M, YTD, 1Y, All)
- 日期範圍選擇器（1 月、3 月、6 月、年初至今、1 年、全部）
- Zoom and pan controls
- 縮放與平移控制

**Factor Weights (if MWU enabled)** / **因子權重（如啟用 MWU）**:

- Time-series chart showing dynamic factor allocation
- 時間序列圖表顯示動態因子配置
- IC history for performance diagnosis
- IC 歷史記錄用於性能診斷

**Step 6: Save Configuration** / **步驟 6：保存配置**

Click **"💾 Save Config"** to save current settings to `config/live_strategy.json`.

點擊 **"💾 保存配置"** 將當前設置保存至 `config/live_strategy.json`。

This configuration will be used by the live execution engine.

此配置將被實時執行引擎使用。

---

## Project Structure / 專案結構

```
ancserAlpacaLab/
├── ancser_quant/               # Core System Logic / 核心系統邏輯
│   ├── alpha/                  # Factor Library & MWU / 因子庫與 MWU
│   │   ├── factors.py          # Alpha Factor Definitions / Alpha 因子定義
│   │   └── mwu.py              # Dynamic Weighting Engine / 動態權重更新引擎
│   ├── data/                   # Data Adapters / 數據適配器
│   │   ├── alpaca_adapter.py   # Alpaca API Adapter / Alpaca API 適配器
│   │   ├── yahoo_adapter.py    # Yahoo Finance Adapter / Yahoo Finance 適配器
│   │   ├── constituents.py     # Stock Universe / 股票池定義
│   │   └── schema.py           # Data Schema / 數據架構定義
│   ├── execution/              # Execution Engine / 執行引擎
│   │   ├── main_loop.py        # Main Execution Loop / 主執行循環
│   │   ├── oms.py              # Order Management System / 訂單管理系統
│   │   └── strategy.py         # Strategy Logic / 策略邏輯
│   └── backtest.py             # Polars Backtest Engine / Polars 回測引擎
│
├── frontend/                   # Frontend Dashboard / 前端儀表板
│   ├── app.py                  # Streamlit Dashboard / Streamlit 儀表板
│   └── error_logger.py         # Error Logging / 錯誤日誌
│
├── scripts/                    # Utility Scripts / 工具腳本
│   ├── run_drift_comparison.py # Factor Comparison Test / 因子對比測試
│   ├── debug_benchmark.py      # Benchmark Debugging / 基準調試
│   └── README.md               # Script Documentation / 腳本說明
│
├── config/                     # Configuration / 配置文件
│   └── live_strategy.json      # Live Strategy Config / 實時策略配置
│
├── data_cache/                 # Data Cache (Parquet) / 數據緩存（Parquet）
├── logs/                       # Application Logs / 應用日誌
│
├── .env                        # Environment Variables (API Keys) / 環境變量（API 密鑰）
├── daily_run.bat               # Daily Entry Point / 每日運行入口
├── force_run.bat               # Force Execution Entry Point / 強制執行入口
├── requirements.txt            # Python Dependencies / Python 依賴
├── setup.bat                   # Setup Script / 環境設置腳本
└── README.md                   # Project Documentation / 項目說明
```

---

## Recent Improvements / 最新改進

✅ **Fixed Drift-Reversion Factor Logic**: Now correctly applies reversion only in non-drift regimes, reducing max drawdown by ~2.77%

**修復 Drift-Reversion 因子邏輯**：現在正確地僅在非漂移狀態下應用回歸，最大回撤降低約 2.77%

✅ **Enhanced MWU Engine**: Added weight bounds (min/max) to prevent extreme factor allocations and IC history tracking

**增強 MWU 引擎**：添加權重邊界（最小/最大）以防止極端因子配置，並追蹤 IC 歷史

✅ **Improved Dashboard Charts**: TradingView-like equity curves with proper range limits, date selectors, and zoom controls

**改進儀表板圖表**：類 TradingView 權益曲線，具備適當範圍限制、日期選擇器與縮放控制

✅ **Fixed Benchmark Display**: Resolved SPY/QQQ/GLD alignment issues in backtest charts

**修復基準顯示**：解決回測圖表中 SPY/QQQ/GLD 對齊問題

✅ **Live Strategy Monitor**: Dashboard now auto-loads and displays current trading logic from saved configurations

**實時策略監控**：儀表板現在自動加載並顯示來自保存配置的當前交易邏輯

✅ **Better Project Organization**: Scripts moved to dedicated folder, clearer separation of concerns

**更好的項目組織**：腳本移至專用文件夾，更清晰的關注點分離

✅ **Automatic Error Logging**: All dashboard errors and warnings automatically saved to `logs/dashboard_YYYY-MM-DD.log`

**自動錯誤日誌**：所有儀表板錯誤與警告自動保存至 `logs/dashboard_YYYY-MM-DD.log`

✅ **Factor Presets with Visual Feedback**: Click preset buttons to quick-load top combinations, active preset highlighted

**帶視覺反饋的因子預設**：點擊預設按鈕快速加載頂級組合，啟用預設高亮顯示

✅ **Year-Based Date Selection**: Simplified date selection with auto-generated start/end dates

**基於年份的日期選擇**：簡化日期選擇，自動生成起始/結束日期

✅ **Data Caching System**: Parquet file caching provides 24x speedup on repeated queries

**數據緩存系統**：Parquet 文件緩存使重複查詢速度提升 24 倍

✅ **Compact 5-Column Layout**: Initial Capital, Date Range, Data Source, Universe, and Risk Management in one row

**緊湊 5 列佈局**：初始資金、日期範圍、數據來源、股票池與風險管理在一行

✅ **Save Config & Force Execute Side-by-Side**: Clear separation of configuration saving and strategy execution

**並列保存配置與強制執行**：配置保存與策略執行清晰分離

---

## Performance Highlights / 性能亮點

**Top Factor Combination** (from All Combo analysis):

**頂級因子組合**（來自全組合分析）：

- **Momentum + Reversion + Skew + Drift-Reversion**: Calmar Ratio 2.12
- **動量 + 均值回歸 + 偏度 + 漂移感知回歸**：Calmar 比率 2.12

**Data Processing Speed**:

**數據處理速度**：

- Yahoo Adapter with cache: 0.03s (vs 0.73s without cache, 24x faster)
- 帶緩存的 Yahoo 適配器：0.03 秒（vs 無緩存 0.73 秒，快 24 倍）
- Polars engine: 10-50x faster than Pandas for large datasets
- Polars 引擎：大數據集比 Pandas 快 10-50 倍

---

## Troubleshooting / 故障排除

**Issue**: Dashboard shows "No API keys" warning

**問題**：儀表板顯示「無 API 密鑰」警告

**Solution**: Create `.env` file with Alpaca credentials or use Yahoo data source

**解決方案**：創建包含 Alpaca 憑證的 `.env` 檔案或使用 Yahoo 數據來源

**Issue**: Backtest returns empty data

**問題**：回測返回空數據

**Solution**: Check date range (Yahoo: 2015+, Alpaca: 2021+) and ensure tickers are valid

**解決方案**：檢查日期範圍（Yahoo：2015+，Alpaca：2021+）並確保股票代碼有效

**Issue**: Preset buttons not updating factors

**問題**：預設按鈕未更新因子

**Solution**: Ensure you're on the Backtest page and click the preset button, then check the multiselect

**解決方案**：確保您在回測頁面並點擊預設按鈕，然後檢查多選框

**Issue**: Force Execute button not working

**問題**：強制執行按鈕不工作

**Solution**: Check logs in `logs/` directory for detailed error messages

**解決方案**：檢查 `logs/` 目錄中的日誌以獲取詳細錯誤消息

---

## Changelog / 更新日誌

See [CHANGELOG_2026-02-15.md](CHANGELOG_2026-02-15.md) for detailed update history.

詳細更新歷史請參閱 [CHANGELOG_2026-02-15.md](CHANGELOG_2026-02-15.md)。

---

## Disclaimer / 免責聲明

This software is for educational and research purposes only. Quantitative trading involves significant financial risk. The authors are not responsible for any financial losses incurred from using this software.

本軟體僅供教育與研究用途。量化交易涉及重大財務風險。作者不對使用本軟體造成的任何財務損失負責。

**Always test strategies in paper trading mode before using real capital.**

**在使用真實資金之前，請務必在模擬交易模式下測試策略。**
