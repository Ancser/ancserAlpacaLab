# Ancser Alpaca Lab

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Polars](https://img.shields.io/badge/Polars-Fast-orange)](https://pola.rs/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io/)
[![Alpaca](https://img.shields.io/badge/Alpaca-Trading-yellow)](https://alpaca.markets/)

**High-Performance Quantitative Trading System (Institutional Grade)**  
**高性能量化交易系統（機構級架構）**

---

## Introduction / 簡介

**ancserAlpacaLab** represents a complete architectural overhaul of the legacy system. It separates "Research (Brain)" from "Execution (Body)" and leverages **Polars** for lightning-fast data processing (10-50x faster than Pandas). The frontend is built with **Streamlit**, providing a modern, interactive dashboard for monitoring strategies and factors.

**ancserAlpacaLab** 是對舊有系統的全面重構。它將「研究（大腦）」與「執行（手腳）」完全分離，並利用 **Polars** 進行極速數據處理（比 Pandas 快 10-50 倍）。前端採用 **Streamlit** 構建，提供現代化的交互式儀表板，用於監控策略和因子表現。

---

## Key Features / 核心功能

### 1. High-Performance Data Engine (高性能數據引擎)
- **Polars Core**: Rust-based DataFrame library for memory efficiency and speed.
- **Strict Schema**: `schema.py` enforces data types (`Float32`, `Categorical`) to prevent data corruption.
- **Unified Adapters**: Seamless data ingestion from **Yahoo Finance** (Backtesting) and **Alpaca** (Real-time Execution).

### 2. Alpha Factor Library (Alpha 因子庫)
- **Cross-Sectional Focus**: Factors rank stocks against their peers rather than just time-series analysis.
- **Factor Categories**:
    - **Momentum & Reversion**: ROC, RSI Divergence.
    - **Higher Moments**: Skewness (Lottery payoff), Idiosyncratic Volatility.
    - **Microstructure**: Amihud Illiquidity, Spread Proxy.
    - **Alpha 101**: WorldQuant-style mathematical alphas.
- **MWU Engine**: Dynamic Weighting mechanism that adjusts factor exposure based on recent IC (Information Coefficient).

### 3. Risk Management (風險管理)
- **Volatility Targeting (Constant Risk)**: Automatically adjusts portfolio leverage based on realized volatility (e.g., Target 20% Vol).
- **Daily Batch Control**: Designed for robust daily rebalancing, protecting against inter-day trends.
- **Idempotent OMS**: Target-based execution ensures safety even if scripts are re-run.

### 4. Interactive Dashboard (交互式儀表板)
- **Streamlit App**: Real-time visualization of Equity Curve, P&L, and Positions.
- **Backtest Engine**: Configure and run high-speed backtests directly from the UI.
- **Live Strategy Config**: Adjust risk parameters (Target Vol, Max Leverage) on the fly without restarting code.

---

## Quick Start / 快速啟動

### 1. Install Dependencies / 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. Configuration / 設定
Create a `.env` file in the root directory:
建立 `.env` 檔案：
```env
APCA_API_KEY_ID=your_api_key
APCA_API_SECRET_KEY=your_secret_key
```

### 3. Daily Operation / 日常操作 (Recommended)
Just double-click **`daily_run.bat`**!
只需雙擊 **`daily_run.bat`**！

- It checks if the market is Open.
- It launches the **Dashboard** (Web UI) for you to view.
- It runs the **Execution Logic** (Black Window) to rebalance your portfolio, then automatically closes.

### 4. Backtesting / 回測
1. Open Dashboard (`daily_run.bat` or `streamlit run dashboard/app.py`).
2. Go to "Strategy Lab" in the sidebar.
3. Select "Alpaca (API)" data source.
4. Click "Run Backtest".

---

## Project Structure / 專案結構

```
ancserAlpacaLab/
├── ancser_quant/           # Core System Logic
│   ├── alpha/              # Factor Library & MWU
│   ├── data/               # Data Adapters (Alpaca/Yahoo)
│   ├── execution/          # Execution Loop & Logic
│   └── backtest.py         # Polars Backtest Engine
├── config/                 # Configuration
│   ├── ancser_quant.yaml   # Static Config
│   └── live_strategy.json  # Dynamic Strategy State
├── dashboard/              # Frontend (Streamlit)
│   └── app.py              # Dashboard Entry Point
├── daily_run.bat           # Daily Automation Script
└── requirements.txt        # Dependencies
```

---

## Disclaimer / 免責聲明

This software is for educational and research purposes only. Quantitative trading involves significant financial risk. The authors are not responsible for any financial losses incurred from using this software.

本軟體僅供教育與研究用途。量化交易涉及重大財務風險。作者不對使用本軟體造成的任何財務損失負責。