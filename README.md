# Project Titan

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Polars](https://img.shields.io/badge/Polars-Fast-orange)](https://pola.rs/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io/)
[![Alpaca](https://img.shields.io/badge/Alpaca-Trading-yellow)](https://alpaca.markets/)

**High-Performance Quantitative Trading System (Institutional Grade)**  
**高性能量化交易系統（機構級架構）**

---

## Introduction / 簡介

**Project Titan** represents a complete architectural overhaul of the legacy system. It separates "Research (Brain)" from "Execution (Body)" and leverages **Polars** for lightning-fast data processing (10-50x faster than Pandas). The frontend is built with **Streamlit**, providing a modern, interactive dashboard for monitoring strategies and factors.

**Project Titan** 是對舊有系統的全面重構。它將「研究（大腦）」與「執行（手腳）」完全分離，並利用 **Polars** 進行極速數據處理（比 Pandas 快 10-50 倍）。前端採用 **Streamlit** 構建，提供現代化的交互式儀表板，用於監控策略和因子表現。

---

## Key Features / 核心功能

### 1. High-Performance Data Engine (高性能數據引擎)
- **Polars Core**: Rust-based DataFrame library for memory efficiency and speed.
- **Strict Schema**: `schema.py` enforces data types (`Float32`, `Categorical`) to prevent data corruption.
- **Unified Adapters**: Seamless data ingestion from **Yahoo Finance** (Backfill) and **Alpaca** (Real-time).

### 2. Alpha Factor Library (Alpha 因子庫)
- **Cross-Sectional Focus**: Factors rank stocks against their peers rather than just time-series analysis.
- **Factor Categories**:
    - **Momentum & Reversion**: ROC, RSI Divergence.
    - **Higher Moments**: Skewness (Lottery payoff), Idiosyncratic Volatility.
    - **Microstructure**: Amihud Illiquidity, Spread Proxy.
    - **Alpha 101**: WorldQuant-style mathematical alphas.
- **MWU Engine**: Dynamic Weighting mechanism that adjusts factor exposure based on recent IC (Information Coefficient). Automatically rotates between Momentum and Reversion regimes.

### 3. Execution Engine (執行引擎)
- **Event-Driven**: Built on `APScheduler` with reliable Heartbeat (1m) and Rebalance (1h) loops.
- **Order Management System (OMS)**:
    - **Diff Engine**: Calculates exact order deltas to reach target portfolio.
    - **TWAP Slicer**: Splits large orders into smaller chunks to minimize market impact.
    - **Risk Gate**: Enforces position limits (e.g., max 10% per stock).

### 4. Interactive Dashboard (交互式儀表板)
- **Streamlit App**: Real-time visualization of Equity Curve, P&L, and Positions.
- **Factor Lab**: Visualize dynamic MWU weights and Factor IC heatmaps.
- **Backtest**: Configure and run high-speed backtests directly from the UI.

---

## Quick Start / 快速啟動

### 1. Install Dependencies / 安裝依賴
```bash
pip install -r requirements.txt
```

### 2. Configuration / 設定
Create a `.env` file in the root directory with your Alpaca API keys:
建立 `.env` 檔案並填入您的 Alpaca API 金鑰：
```env
APCA_API_KEY_ID=your_api_key
APCA_API_SECRET_KEY=your_secret_key
```

Edit `config/titan_config.yaml` to customize settings (Universe, Risk params, etc.).
編輯 `config/titan_config.yaml` 以自訂設定（股票池、風險參數等）。

### 3. Launch Dashboard / 啟動儀表板
```bash
streamlit run dashboard/app.py
```
> The dashboard will open automatically in your default browser (usually http://localhost:8501).
> 儀表板將自動在您的預設瀏覽器中開啟（通常是 http://localhost:8501）。

### 4. Run Execution / 執行交易引擎 (Server)
```bash
# Windows
python -m titan_core.execution.main_loop
```

---

## Project Structure / 專案結構

```
ancserAlpacaLab/
├── config/                 # Configuration files
│   └── titan_config.yaml   # System settings
├── dashboard/              # Frontend Application
│   └── app.py              # Streamlit Entry Point
├── titan_core/             # Core System Logic
│   ├── alpha/              # Alpha Research Layer (Brain)
│   │   ├── factors.py      # Factor Implementations (Polars)
│   │   └── mwu.py          # Dynamic Weighting Engine
│   ├── data/               # Data Infrastructure
│   │   ├── schema.py       # Data Types Definition
│   │   ├── alpaca_adapter.py
│   │   └── yahoo_adapter.py
│   └── execution/          # Execution Layer (Body)
│       ├── main_loop.py    # Event Loop
│       └── oms.py          # Order Management
└── requirements.txt        # Python Dependencies
```

---

## Disclaimer / 免責聲明

This software is for educational and research purposes only. Quantitative trading involves significant financial risk. The authors are not responsible for any financial losses incurred from using this software.

本軟體僅供教育與研究用途。量化交易涉及重大財務風險。作者不對使用本軟體造成的任何財務損失負責。