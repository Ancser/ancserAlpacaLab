# Dashboard Refactor Implementation Plan

## Overview
7 changes to `frontend/app.py` + supporting files. Estimated ~800 lines changed.

---

## Change 1: Remove "Overview" Header
**File**: `frontend/app.py` line 46
- Delete `st.header("Overview")` — already have `st.title("ancserAlpacaLab")` at top

## Change 2: Remove Tracker Page, Merge into Dashboard
**File**: `frontend/app.py`
- Delete entire Tracker page block (lines 404-525)
- Change sidebar nav from `["Dashboard", "Tracker", "Backtest"]` → `["Dashboard", "Backtest"]`
- Move "Live vs Model" comparison concept into Dashboard (see Change 3)

## Change 3: Backtest vs Live Comparison Chart (KEY FEATURE)
**Files**: `frontend/app.py`, `ancser_quant/execution/oms.py`

### 3a. Save strategy config in rebalance_history
- Modify `oms.py` `generate_and_execute_orders()` to accept and save `strategy_config` in each snapshot
- Each rebalance_history entry gets: `active_factors`, `use_mwu`, `leverage`, `vol_target`, `strategy_mode`
- This lets us reconstruct which strategy was active on which dates

### 3b. Dashboard auto-backtest section
- After "Live Strategy Configuration & Preview", add "Strategy Performance: Backtest vs Live"
- On load (cached with `@st.cache_data`):
  1. Read `rebalance_history.json` to find date range (earliest rebalance date → yesterday)
  2. Group rebalance entries by strategy config (factor set) to identify strategy change points
  3. For each strategy segment: run backtest with that config over that date range
  4. Stitch backtest segments together into one continuous prediction line
  5. Get actual equity from `adapter.get_portfolio_history()` for same date range
  6. Plot both on one Plotly chart: solid green = actual, dashed gray = backtest predicted
- Cache key: hash of rebalance_history + today's date (invalidate daily)
- Show tracking metrics: Actual vs Predicted equity, delta $, delta %

### 3c. Backtest speed optimization (for auto-backtest)
- Use `@st.cache_data(ttl=3600)` to cache backtest results for 1 hour
- For dashboard auto-backtest, use reduced universe if full 500+ symbol backtest is too slow
  - Alternative: cache the `fetch_and_prepare_data()` result separately (heavy I/O)

## Change 4: Equity Curve — Add Realized Gain (Dual Y-Axis)
**File**: `frontend/app.py`

- Add second Y-axis (right side) for cumulative realized gain
- Data source: `adapter.get_activities()` — sum realized P&L from sell fills
  - For each sell: realized = sell_price * qty (proceeds)
  - Track cumulative realized gain by date
- Plotly: `make_subplots(specs=[[{"secondary_y": True}]])` or `fig.add_trace(..., yaxis='y2')`
- Left axis: Equity line (green, fill) — existing
- Right axis: Cumulative realized gain bar/line (gold/yellow)

## Change 5: Holdings → Card Layout
**File**: `frontend/app.py` (lines 157-177)

Replace `st.dataframe()` with card grid:
- Use `st.columns()` to create 3-4 cards per row
- Each card (`st.container(border=True)`):
  ```
  ┌─────────────────────┐
  │  AAPL          +2.3%│
  │  Avg: $189.50       │
  │  Now: $193.86       │
  │  P&L: +$45.21  ▲    │
  └─────────────────────┘
  ```
- Color: Green header/P&L for profit, Red for loss
- Use `st.markdown()` with inline CSS for coloring

## Change 6: Performance Cards → Realized Gains per Date
**File**: `frontend/app.py` (lines 251-364)

Enhance existing `render_day_card()`:
- Keep current: date header, buys/sells list
- Add NEW: **Realized Gain/Loss** column
  - Compare today's rebalance_history snapshot vs yesterday's
  - Stocks that were in yesterday but NOT in today = sold (fully exited)
  - Realized P&L = (sell_price - entry_price) * qty
  - Use activities data for exact sell prices
- Add section in each card:
  ```
  Realized Gains:
    MSTR: sold 6.39 @ $137 → bought @ $130 = +$44.73 ✅
    TXN: sold 3.76 @ $213 → bought @ $215 = -$7.52 ❌
  ```
- Summary at top: "Total Realized Today: +$XX.XX"

## Change 7: Backtest Engine — Multi-thread Acceleration
**File**: `ancser_quant/backtest.py`

The simulation loop is inherently sequential (MWU depends on previous day).
Optimizations:
1. **Parallel data fetching**: Split 500+ symbols into chunks, fetch in parallel with ThreadPoolExecutor
2. **Cache factor data**: `@st.cache_data` on `fetch_and_prepare_data()` in dashboard
3. **Polars optimization**: Factor computation already vectorized, but ensure `.collect()` is deferred as late as possible
4. **NumPy vectorization in simulation loop**: Pre-compute all factor ranks for all dates at once using pivot tables, then just look up during simulation (avoid per-day DataFrame operations)

---

## Execution Order
1. Change 1 (trivial) + Change 2 (remove tracker)
2. Change 5 (holdings cards) + Change 6 (realized gains in date cards)
3. Change 4 (dual-axis equity chart)
4. Change 3 (backtest vs live — biggest feature)
5. Change 7 (backtest acceleration)

## Files Modified
- `frontend/app.py` — main refactor (all 7 changes)
- `ancser_quant/backtest.py` — parallel fetch, vectorization (Change 7)
- `ancser_quant/execution/oms.py` — save strategy config in snapshots (Change 3a)
- `ancser_quant/execution/main_loop.py` — pass config to OMS (Change 3a)
