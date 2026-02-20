import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

load_dotenv() # Load environments from .env file

# Setup Error Logging
from error_logger import setup_logging, log_error, log_info, log_warning
log_file = setup_logging()

# Page Setup
st.set_page_config(page_title="AncserAlpacaLab", layout="wide", page_icon=None)

st.title("ancserAlpacaLab")

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Dashboard", "Backtest"])

# Display Log File Location
st.sidebar.markdown("---")
st.sidebar.caption(f"📝 Error Log: `{log_file}`")
with st.sidebar.expander("View Recent Logs", expanded=False):
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            recent_logs = f.readlines()[-50:]  # Last 50 lines
            st.code("".join(recent_logs), language="log")
    except Exception as e:
        st.error(f"Cannot read log file: {e}")

# --- Helper Functions (Simulation for MVP) ---
@st.cache_data
def load_mock_data():
    dates = pd.date_range(end=datetime.today(), periods=100)
    equity = 100000 * (1 + np.random.randn(100) * 0.01).cumprod()
    return pd.DataFrame({'date': dates, 'equity': equity}).set_index('date')

# --- Page: Dashboard ---
if page == "Dashboard":
    st.header("Overview")
    
    # Fetch Real Data
    from ancser_quant.data.alpaca_adapter import AlpacaAdapter
    try:
        adapter = AlpacaAdapter()
        acct = adapter.get_account()
    except Exception as e:
        st.error(f"Failed to connect to Alpaca: {e}")
        acct = {'equity': 0.0, 'buying_power': 0.0}

    col1, col2, col3, col4 = st.columns(4)
    
    current_equity = acct.get('equity', 0.0)
    buying_power = acct.get('buying_power', 0.0)
    
    # Placeholder for P&L (Alpaca doesn't give daily P&L history in simple account call, requires portfolio history)
    # For now, we show Equity and Buying Power
    
    col1.metric("Total Equity", f"${current_equity:,.2f}")
    col2.metric("Buying Power", f"${buying_power:,.2f}")
    col3.metric("Status", acct.get('status', 'Unknown'))
    col4.metric("Currency", acct.get('currency', 'USD'))
    
    st.subheader("Equity Curve")

    # Timeframe Selector
    tf_col, _ = st.columns([1, 3])
    # Alpaca Valid Periods: 1D, 1W, 1M, 3M, 1A, 5A. (6M and ALL are often not supported explicitly in this endpoint)
    period_map = {
        "1 Month": "1M",
        "3 Months": "3M",
        "1 Year": "1A",
        "5 Years": "5A"
    }
    selected_label = tf_col.selectbox("Timeframe", list(period_map.keys()), index=0)
    selected_tf = period_map[selected_label]

    try:
        hist_df = adapter.get_portfolio_history(period=selected_tf)
        if not hist_df.empty:
            # Create professional Plotly chart (TradingView-like)
            fig_eq = go.Figure()

            # Add equity line
            fig_eq.add_trace(go.Scatter(
                x=hist_df.index,
                y=hist_df['equity'],
                mode='lines',
                name='Portfolio Equity',
                line=dict(color='#00CC96', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 204, 150, 0.1)'
            ))

            # Calculate Y-axis range with padding (like TradingView)
            equity_values = hist_df['equity'].dropna().values
            if len(equity_values) > 0:
                y_min = min(equity_values)
                y_max = max(equity_values)
                y_padding = (y_max - y_min) * 0.1  # 10% padding
                y_range = [max(0, y_min - y_padding), y_max + y_padding]
            else:
                y_range = None

            # Layout with TradingView-like features
            fig_eq.update_layout(
                yaxis=dict(
                    range=y_range,
                    fixedrange=False,  # Allow zoom
                    rangemode='tozero',  # Don't go below zero
                    constraintoward='bottom',  # Constrain toward bottom (zero)
                    title="Equity ($)"
                ),
                xaxis=dict(
                    type='date',
                    rangeslider=dict(visible=False),  # Hide range slider
                    title="Date"
                ),
                xaxis_rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1W", step="day", stepmode="backward"),
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ]),
                    bgcolor='rgba(150, 150, 150, 0.1)',
                    activecolor='rgba(100, 100, 100, 0.3)'
                ),
                hovermode='x unified',
                margin=dict(l=0, r=0, t=40, b=0),
                dragmode='zoom',  # Default to zoom mode
                template='plotly_dark'
            )

            # Config options
            config = {
                'scrollZoom': True,
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'displaylogo': False
            }

            st.plotly_chart(fig_eq, width="stretch", config=config)
        else:
            st.info("No portfolio history available.")
    except Exception as e:
        st.error(f"Failed to load chart: {e}")
    
    st.subheader("Holdings")
    try:
        positions = adapter.get_positions()
        if positions:
            pos_df = pd.DataFrame(positions)
            # Format columns
            st.dataframe(
                pos_df.style.format({
                    'Qty': "{:.2f}",
                    'Market Value': "${:,.2f}",
                    'Avg Entry': "${:,.2f}",
                    'Current Price': "${:,.2f}",
                    'Unrealized P&L': "${:,.2f}",
                    'P&L %': "{:.2f}%"
                }), 
                width="stretch"
            )
        else:
            st.info("No open positions.")
    except Exception as e:
        st.error(f"Failed to load holdings: {e}")

    st.subheader("Recent Orders")
    try:
        orders = adapter.get_orders()
        if orders:
            ord_df = pd.DataFrame(orders)

            # Calculate Display Value: Prefer Notional, else Filled Value (Qty * Price)
            # ord_df['notional'] might be NaN if not notional order
            # ord_df['filled_avg_price'] might be NaN if not filled

            def get_val(row):
                if pd.notna(row.get('notional')) and row.get('notional') > 0:
                    return row['notional']
                elif pd.notna(row.get('filled_avg_price')) and pd.notna(row.get('filled_qty')):
                    return row['filled_avg_price'] * row['filled_qty']
                return 0.0

            ord_df['Value'] = ord_df.apply(get_val, axis=1)

            # Formatting
            display_cols = ['created_at', 'symbol', 'side', 'qty', 'notional', 'Value', 'status', 'filled_avg_price', 'type']
            # Filter available columns
            display_cols = [c for c in display_cols if c in ord_df.columns]

            st.dataframe(
                ord_df[display_cols].style.format({
                    'qty': lambda x: f"{x:.2f}" if pd.notna(x) else "-",
                    'notional': lambda x: f"${x:,.2f}" if pd.notna(x) else "-",
                    'Value': lambda x: f"${x:,.2f}" if pd.notna(x) else "-",
                    'filled_avg_price': lambda x: f"${x:,.2f}" if pd.notna(x) else "-"
                }),
                width="stretch"
            )
        else:
            st.info("No recent orders.")
    except Exception as e:
        st.error(f"Failed to load orders: {e}")

    # --- Live Strategy Monitor (Auto-loaded) ---
    st.markdown("---")
    st.subheader("Live Strategy Configuration & Preview")

    # Load current live strategy config
    import json
    config_path = "config/live_strategy.json"

    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                live_config = json.load(f)

            # Display config summary
            col_cfg1, col_cfg2, col_cfg3, col_cfg4 = st.columns(4)
            col_cfg1.metric("Active Factors", len(live_config.get('active_factors', [])))
            col_cfg2.metric("Max Leverage", f"{live_config.get('leverage', 1.0):.1f}x")
            col_cfg3.metric("MWU Enabled", "Yes" if live_config.get('use_mwu', False) else "No")
            col_cfg4.metric("Vol Targeting", "Yes" if live_config.get('use_vol_target', False) else "No")

            with st.expander("Strategy Details", expanded=False):
                st.json(live_config)

            # Auto-calculate target portfolio
            st.markdown("##### Current Target Portfolio")
            st.caption(f"Last updated: {live_config.get('last_updated', 'N/A')}")

            with st.spinner("Calculating target portfolio..."):
                from ancser_quant.execution.strategy import LiveStrategy

                strat = LiveStrategy()
                res = strat.calculate_targets(live_config)

                if "error" in res:
                    st.error(f"Calculation Failed: {res['error']}")
                else:
                    # 1. Volatility Metrics
                    vol = res.get('vol_metrics', {})
                    if "current_vol" in vol:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Market Vol (20d)", f"{vol['current_vol']:.2%}")
                        c2.metric("Target Vol", f"{vol.get('target_vol', 0):.2%}")
                        c3.metric("Leverage Scalar", f"{vol.get('final_scalar', 1.0):.2f}x",
                                  help=f"Raw: {vol.get('raw_scalar',0):.2f}x | Cap: {vol.get('leverage_cap',0)}x")

                    # 2. Target Allocation
                    alloc = res.get('allocations', {})
                    prices = res.get('latest_prices', {})
                    scores = res.get('factor_scores', {})

                    if alloc:
                        rows = []
                        for sym, w in alloc.items():
                            rows.append({
                                "Symbol": sym,
                                "Target Weight": f"{w:.2%}",
                                "Factor Score": f"{scores.get(sym, 0):.4f}",
                                "Close Price": f"${prices.get(sym, 0):.2f}"
                            })
                        st.dataframe(pd.DataFrame(rows), width="stretch")
                    else:
                        st.warning("No stocks selected. Check data or factors.")

        else:
            st.info("No live strategy configured yet. Go to Backtest page to save a configuration.")

    except Exception as e:
        st.error(f"Failed to load live strategy: {e}")
        import traceback
        st.code(traceback.format_exc())



# --- Page: Backtest ---
elif page == "Backtest":
    st.header("Strategy Backtester (Polars Engine)")

    # Load universes (before form)
    try:
        from ancser_quant.data.constituents import TICKERS as SPY_QQQ_TICKERS
    except ImportError:
        SPY_QQQ_TICKERS = []

    tech_titans = "AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, AMD, SPY, QQQ"
    full_universe = ", ".join(SPY_QQQ_TICKERS) if SPY_QQQ_TICKERS else tech_titans

    has_alpaca_keys = os.getenv("APCA_API_KEY_ID") and os.getenv("APCA_API_SECRET_KEY")

    # === Factor Selection (Outside Form) ===
    st.markdown("**🧬 Factor Selection**")
    factor_col_left, factor_col_right = st.columns([3, 2])

    # Preset combos (Top 5 from allcombo.csv)
    preset_combos = {
        "Top 1: Momentum + Reversion + Skew + Drift-Reversion": ['Momentum', 'Reversion', 'Skew', 'Drift-Reversion'],
        "Top 2: Momentum + Reversion": ['Momentum', 'Reversion'],
        "Top 3: Momentum + Drift-Reversion": ['Momentum', 'Drift-Reversion'],
        "Top 4: Momentum": ['Momentum'],
        "Top 5: Momentum + Reversion + Skew": ['Momentum', 'Reversion', 'Skew'],
    }

    # Initialize selected factors in session state
    if 'selected_factors' not in st.session_state:
        st.session_state['selected_factors'] = ['Momentum', 'Reversion']

    # Left: Manual Selection
    with factor_col_left:
        st.markdown("**Manual Selection**")
        all_factors = ['Momentum', 'Reversion', 'Skew', 'Microstructure', 'Alpha 101', 'Volatility', 'Drift-Reversion', 'Unicorn Edge']

        factors = st.multiselect(
            "Active Factors",
            all_factors,
            default=st.session_state['selected_factors'],
            label_visibility="collapsed"
        )

        # Update session state when user manually changes selection
        if factors != st.session_state['selected_factors']:
            st.session_state['selected_factors'] = factors

        # Show selected count
        st.caption(f"✓ {len(factors)} factors selected")

        use_mwu = st.checkbox("Enable MWU", value=True, help="Multiplicative Weight Update")

    # Right: Presets
    with factor_col_right:
        st.markdown("**Saved Combos**")

        for preset_name, preset_factors in preset_combos.items():
            # Check if this preset is currently active
            is_active = (st.session_state.get('selected_factors', []) == preset_factors)
            button_type = "primary" if is_active else "secondary"

            if st.button(preset_name, key=f"preset_{preset_name}", type=button_type):
                # Update selected factors and rerun
                st.session_state['selected_factors'] = preset_factors
                st.rerun()

        st.caption("Active preset highlighted")

    st.markdown("---")

    # === Configuration Form ===
    with st.form("backtest_config"):

        # === ROW 1: Compact Configuration (6 columns, each ~1/6 width) ===
        col_capital, col_date, col_source, col_universe, col_risk, col_strategy = st.columns(6)

        # Initial Capital
        with col_capital:
            st.markdown("**💰 Capital**")
            initial_capital = st.number_input(
                "Capital",
                min_value=1000.0,
                max_value=1000000.0,
                value=4000.0,
                step=1000.0,
                label_visibility="collapsed"
            )
            st.caption(f"${initial_capital/1000:.0f}k")

        # Date Range (Year-based)
        with col_date:
            st.markdown("**📅 Date Range**")
            current_year = datetime.today().year
            start_year = st.selectbox("Start Year", range(2015, current_year + 1), index=6, label_visibility="collapsed")  # Default 2021
            end_year = st.selectbox("End Year", range(start_year, current_year + 1), index=current_year - start_year, label_visibility="collapsed")

            # Auto-generate dates
            start_date = datetime(start_year, 1, 1)
            if end_year == current_year:
                end_date = datetime.today()  # Today if current year
            else:
                end_date = datetime(end_year, 12, 31)

            st.caption(f"{start_date.strftime('%Y-%m-%d')} → {end_date.strftime('%Y-%m-%d')}")

        # Data Source
        with col_source:
            st.markdown("**📊 Data Source**")
            data_source_options = ["Yahoo (2015+)", "Alpaca (2021+)"]
            default_idx = 1 if has_alpaca_keys else 0

            data_source = st.radio(
                "Source",
                data_source_options,
                index=default_idx,
                label_visibility="collapsed"
            )

            if "Alpaca" in data_source and not has_alpaca_keys:
                st.warning("No API keys!")

        # Universe
        with col_universe:
            st.markdown("**🌐 Universe**")
            u_preset = st.radio(
                "Preset",
                ["Tech 10", "S&P+Nasdaq"],
                index=1,
                label_visibility="collapsed",
                horizontal=True
            )
            universe_input = st.text_area(
                "Symbols",
                value=full_universe if u_preset == "S&P+Nasdaq" else tech_titans,
                height=60,
                label_visibility="collapsed"
            )
            universe = [s.strip().upper() for s in universe_input.split(',') if s.strip()]
            st.caption(f"{len(universe)} symbols")

        # Risk Management
        with col_risk:
            st.markdown("**⚖️ Risk**")
            leverage = st.slider("Leverage", 1.0, 2.0, 1.0, 0.1)  # Max 2x
            use_vol_target = st.checkbox("Vol Targeting", value=True)
            vol_target = st.slider("Target Vol", 0.05, 1.0, 0.20, 0.05, label_visibility="collapsed")

        # Strategy Mode
        with col_strategy:
            st.markdown("**📐 Strategy**")
            strategy_options = ["Top 5 Long Only", "Top 10 Long Only", "Top 10 L/S", "150L / 150S"]
            strategy_label = st.selectbox(
                "Mode",
                strategy_options,
                index=0,
                label_visibility="collapsed"
            )
            strategy_mode_map = {
                "Top 5 Long Only":  'long_only',
                "Top 10 Long Only": 'top10_long',
                "Top 10 L/S":       'top10_ls',
                "150L / 150S":      'long_short',
            }
            strategy_mode = strategy_mode_map[strategy_label]
            captions = {
                'long_only':  "Long top 5",
                'top10_long': "Long top 10",
                'top10_ls':   "L10/S10\n(backtest)",
                'long_short': "L150/S150\n(backtest)",
            }
            st.caption(captions[strategy_mode])

        # === ROW 2: Action Buttons (Run + Combo only) ===
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        submitted = col_btn1.form_submit_button("▶️ Run Backtest", type="primary")
        run_combo = col_btn2.form_submit_button("🔄 Run All Combos")

        # Data source mapping for engine
        if "Alpaca" in data_source:
            data_source_code = "alpaca"
        else:
            data_source_code = "yahoo"

    # Force Execute & Save Config Buttons (Outside Form)
    st.markdown("---")
    col_exec1, col_exec2 = st.columns(2)

    # Save Config Button
    if col_exec1.button("💾 Save Config", type="secondary", help="Save current configuration to config/live_strategy.json"):
        import json
        import os

        config_path = "config/live_strategy.json"
        live_config = {
            "active_factors": factors,
            "leverage": leverage,
            "universe": universe,
            "use_mwu": use_mwu,
            "use_vol_target": use_vol_target,
            "vol_target": vol_target,
            "strategy_mode": strategy_mode,
            "last_updated": datetime.now().isoformat()
        }

        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(live_config, f, indent=4)
            st.success(f"✅ Configuration saved to {config_path}!")
            st.info("Use the 'Force Execute Now' button to apply immediately, or wait for the next scheduled rebalance.")
        except Exception as e:
            st.error(f"Failed to save configuration: {e}")

    # Force Execute Button
    if col_exec2.button("⚡ Force Execute Now", type="primary", help="Run execution immediately with --force flag"):
        with st.spinner("Running forced execution..."):
            import subprocess
            try:
                # Run the force execution script
                result = subprocess.run(
                    ["python", "-m", "ancser_quant.execution.main_loop", "--run-once", "--force"],
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout
                )

                if result.returncode == 0:
                    st.success("✅ Forced execution completed successfully!")
                    with st.expander("Execution Output", expanded=False):
                        st.code(result.stdout)
                    if result.stderr:
                        with st.expander("Warnings/Errors", expanded=False):
                            st.code(result.stderr)
                else:
                    st.error(f"❌ Execution failed with return code {result.returncode}")
                    st.code(result.stderr)

            except subprocess.TimeoutExpired:
                st.error("⏱️ Execution timed out after 2 minutes. Check logs for details.")
            except Exception as e:
                st.error(f"❌ Failed to run execution: {e}")
                import traceback
                st.code(traceback.format_exc())

    # --- Live Strategy Monitor & Preview (Outside Form) ---
    st.markdown("---")
    st.subheader("Live Strategy Monitor & Preview")
    
    col_prev1, col_prev2 = st.columns([1, 3])
    if col_prev1.button("Preview Target Portfolio (Now)"):
        with st.spinner("Calculating Target Portfolio based on current market data..."):
            from ancser_quant.execution.strategy import LiveStrategy
            
            # Use current UI state as config
            temp_config = {
                "active_factors": factors,
                "leverage": leverage,
                "universe": universe,
                "use_mwu": use_mwu,
                "use_vol_target": use_vol_target,
                "vol_target": vol_target
            }
            
            strat = LiveStrategy()
            res = strat.calculate_targets(temp_config)
            
            if "error" in res:
                st.error(f"Calculation Failed: {res['error']}")
            else:
                # 1. Volatility Metrics
                vol = res.get('vol_metrics', {})
                if "current_vol" in vol:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Market Vol (20d)", f"{vol['current_vol']:.2%}")
                    c2.metric("Target Vol", f"{vol.get('target_vol', 0):.2%}")
                    c3.metric("Leverage Scalar", f"{vol.get('final_scalar', 1.0):.2f}x", help=f"Raw: {vol.get('raw_scalar',0):.2f}x | Cap: {vol.get('leverage_cap',0)}x")
                
                # 2. Target Allocation
                alloc = res.get('allocations', {})
                prices = res.get('latest_prices', {})
                scores = res.get('factor_scores', {})
                
                if alloc:
                    st.markdown("##### Target Portfolio (If rebalanced now)")
                    rows = []
                    for sym, w in alloc.items():
                        rows.append({
                            "Symbol": sym,
                            "Target Weight": f"{w:.2%}",
                            "Factor Score": f"{scores.get(sym, 0):.4f}",
                            "Close Price": f"${prices.get(sym, 0):.2f}"
                        })
                    st.dataframe(pd.DataFrame(rows), width="stretch")
                else:
                    st.warning("No stocks selected. Check data or factors.")

    if run_combo:
            st.info("Starting Combinatorial Search... This may take a minute.")
            import itertools
            from ancser_quant.backtest import BacktestEngine
            from ancser_quant.data.yahoo_adapter import YahooAdapter
            
            # 1. Fetch & Prepare Data ONCE
            progress = st.progress(0)
            engine = BacktestEngine(initial_capital=initial_capital, data_source=data_source_code)
            s_str = start_date.strftime('%Y-%m-%d')
            e_str = end_date.strftime('%Y-%m-%d')
            
            with st.spinner("Fetching data and computing factors..."):
                data = engine.fetch_and_prepare_data(universe, s_str, e_str)
                
            if data is None or data.empty:
                st.error("No data available.")
                st.stop()
                
            # 2. Generate Combinations
            # Min 1 factor, max 6
            combos = []
            for r in range(1, len(all_factors) + 1):
                combos.extend(itertools.combinations(all_factors, r))
            
            st.write(f"Testing {len(combos)} combinations...")
            
            results_list = []
            eq_curves = {}
            
            # Benchmarks
            benchmarks = ['SPY', 'QQQ', 'GLD']
            yahoo = YahooAdapter()
            bench_df_lazy = yahoo.fetch_history(benchmarks, s_str, e_str)
            bench_df = bench_df_lazy.collect().to_pandas()
            
            # 3. Multi-threaded Loop
            import concurrent.futures
            
            total_steps = len(combos)
            
            def run_single_combo(combo):
                c_name = " + ".join(combo)
                # Run Simulation
                try:
                    res, _, _ = engine.run_simulation(data, list(combo), leverage, use_mwu, use_vol_target, vol_target, strategy_mode=strategy_mode)
                    if res.empty:
                        return None
                        
                    eq = res['equity']
                    start_eq = eq.iloc[0]
                    end_eq = eq.iloc[-1]
                    
                    # Metrics
                    duration = (pd.to_datetime(e_str) - pd.to_datetime(s_str)).days / 365.25
                    cagr = (end_eq / start_eq) ** (1/duration) - 1 if duration > 0 else 0
                    
                    rmax = eq.cummax()
                    dd = (eq - rmax) / rmax
                    mdd = dd.min()
                    
                    calmar = cagr / abs(mdd) if mdd < 0 else 0
                    
                    daily_ret = eq.pct_change().dropna()
                    sharpe = (daily_ret.mean() / daily_ret.std()) * (252**0.5) if daily_ret.std() > 0 else 0
                    
                    return {
                        'Combination': c_name,
                        'Factors': len(combo),
                        'Final Equity': end_eq,
                        'CAGR': cagr,
                        'Sharpe': sharpe,
                        'MDD': mdd,
                        'Calmar': calmar,
                        'EquityCurve': eq
                    }
                except Exception as e:
                    return None

            # Run in parallel
            # Adjust max_workers as needed (usually 2x cores is fine for I/O mixed, but this is CPU heavy python loop)
            # Python threading has GIL, so this might not be 100% true parallel for pure python logic, 
            # but Polars operations internally release GIL.
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(combos))) as executor:
                futures = {executor.submit(run_single_combo, c): c for c in combos}
                
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    res = future.result()
                    if res:
                        # Separate curve from metrics to avoid huge dataframe display issues
                        curve = res.pop('EquityCurve')
                        eq_curves[res['Combination']] = curve
                        results_list.append(res)
                    
                    progress.progress((i + 1) / total_steps)
            
            # 4. Rank Results
            rank_df = pd.DataFrame(results_list)
            if not rank_df.empty:
                rank_df = rank_df.sort_values('Calmar', ascending=False).reset_index(drop=True)
                rank_df.index += 1 # 1-based rank
                
                st.subheader("Top Combinations (Ranked by Calmar)")
                st.dataframe(rank_df.style.format({
                    'Final Equity': "${:,.2f}",
                    'CAGR': "{:.2%}",
                    'Sharpe': "{:.2f}",
                    'MDD': "{:.2%}",
                    'Calmar': "{:.2f}"
                }))
                
                # 5. Plot Top N + Benchmarks
                st.subheader("Equity Curves: Top 5 Combinations vs Benchmarks")
                fig_combo = go.Figure()
                
                # Add Top 5
                top_5 = rank_df.head(5)['Combination'].tolist()
                colors_list = ['#00CC96', '#EF553B', '#AB63FA', '#FFA15A', '#19D3F3']
                
                for i, name in enumerate(top_5):
                    if name in eq_curves:
                        fig_combo.add_trace(go.Scatter(
                            x=eq_curves[name].index,
                            y=eq_curves[name],
                            mode='lines',
                            name=f"#{i+1}: {name}",
                            line=dict(width=2, color=colors_list[i % len(colors_list)])
                        ))
                
                # Add Benchmarks
                if not bench_df.empty:
                    bench_df['timestamp'] = pd.to_datetime(bench_df['timestamp'])
                    if bench_df['timestamp'].dt.tz is not None:
                         bench_df['timestamp'] = bench_df['timestamp'].dt.tz_localize(None)

                    bench_df['timestamp'] = bench_df['timestamp'].dt.normalize()

                    pivot_bench = bench_df.pivot(index='timestamp', columns='symbol', values='close')

                    # Ensure equity index is also datetime and naive
                    ref_idx = eq_curves[top_5[0]].index
                    if not isinstance(ref_idx, pd.DatetimeIndex):
                        ref_idx = pd.to_datetime(ref_idx)

                    if ref_idx.tz is not None:
                        ref_idx = ref_idx.tz_localize(None)

                    ref_idx = ref_idx.normalize()

                    pivot_bench = pivot_bench.reindex(ref_idx).ffill().bfill()

                    b_colors = {'SPY': 'gray', 'QQQ': 'silver', 'GLD': 'gold'}
                    for b in benchmarks:
                        if b in pivot_bench.columns:
                            # Find first valid value
                            b_series = pivot_bench[b].dropna()
                            if not b_series.empty:
                                b_start = b_series.iloc[0]
                                if b_start > 0:
                                    b_norm = (pivot_bench[b] / b_start) * 100000.0
                                    fig_combo.add_trace(go.Scatter(
                                        x=b_norm.index,
                                        y=b_norm,
                                        mode='lines',
                                        name=b,
                                        line=dict(color=b_colors.get(b, 'gray'), width=1, dash='dot')
                                    ))

                # Calculate Y-axis range for combo chart
                all_combo_values = []
                for name in top_5:
                    if name in eq_curves:
                        all_combo_values.extend(eq_curves[name].dropna().values)

                if not bench_df.empty:
                    for b in benchmarks:
                        if b in pivot_bench.columns:
                            b_series = pivot_bench[b].dropna()
                            if not b_series.empty:
                                b_norm = (b_series / b_series.iloc[0]) * 100000.0
                                all_combo_values.extend(b_norm.values)

                if all_combo_values:
                    y_min_combo = min(all_combo_values)
                    y_max_combo = max(all_combo_values)
                    y_padding_combo = (y_max_combo - y_min_combo) * 0.1
                    y_range_combo = [max(0, y_min_combo - y_padding_combo), y_max_combo + y_padding_combo]
                else:
                    y_range_combo = [0, 200000]

                fig_combo.update_layout(
                    title="Best Combinations vs Benchmarks",
                    xaxis_title="Date",
                    yaxis_title="Equity",
                    hovermode="x unified",
                    yaxis=dict(
                        range=y_range_combo,
                        fixedrange=False,
                        rangemode='tozero',
                        constraintoward='bottom'
                    ),
                    xaxis=dict(
                        type='date',
                        rangeslider=dict(visible=False)
                    ),
                    xaxis_rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(step="all", label="All")
                        ])
                    ),
                    dragmode='zoom'
                )

                config_combo = {
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'displaylogo': False
                }

                st.plotly_chart(fig_combo, width="stretch", config=config_combo)
                
            else:
                st.warning("No combinations produced valid results.")

    if submitted:
        st.info(f"Fetching data and running simulation...")
        
        # Progress bar
        progress = st.progress(0)
        
        # Capture logs
        import io
        import sys
        log_capture = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = log_capture

        try:
            from ancser_quant.backtest import BacktestEngine
            from ancser_quant.data.yahoo_adapter import YahooAdapter

            # Use the data source code from form
            engine = BacktestEngine(initial_capital=initial_capital, data_source=data_source_code)
            
            # Convert dates to string
            s_str = start_date.strftime('%Y-%m-%d')
            e_str = end_date.strftime('%Y-%m-%d')
            
            progress.progress(10)
            
            # 1. Run Strategy Backtest
            results, weight_history, holdings_df = engine.run(universe, s_str, e_str, factors, leverage, use_mwu, use_vol_target, vol_target, strategy_mode=strategy_mode)
            
            progress.progress(50)
            
            # 2. Fetch Benchmarks (SPY, QQQ, GLD)
            benchmarks = ['SPY', 'QQQ', 'GLD']
            print(f"Fetching benchmarks: {benchmarks}")
            
            # Always use Yahoo for benchmarks (Free, Reliable for ETFs)
            # Alpaca IEX feed might miss some ETF data on free tier
            from ancser_quant.data.yahoo_adapter import YahooAdapter
            bench_adapter = YahooAdapter()
                
            bench_df_lazy = bench_adapter.fetch_history(benchmarks, s_str, e_str)
            bench_df = bench_df_lazy.collect().to_pandas()
            print(f"Benchmark DF Shape: {bench_df.shape}")
            if not bench_df.empty:
                print(bench_df.head())
            else:
                print("Benchmark DF is empty!")
            
            progress.progress(80)
            
            sys.stdout = original_stdout # Restore stdout
            
            # Display Logs
            logs = log_capture.getvalue()
            with st.expander("Backtest Logs (Debug)", expanded=False):
                st.code(logs)

            if results.empty:
                st.error("No data returned or strategy didn't trade.")
                st.warning("Possible reasons: 1. Yahoo Finance API blocking. 2. Date range too old.")
            else:
                st.success("Backtest Completed!")
                
                # --- Metrics Calculation ---
                # Strategy
                equity = results['equity']
                start_eq = equity.iloc[0]
                end_eq = equity.iloc[-1]
                total_ret = (end_eq / start_eq) - 1
                
                # ========== BENCHMARK PROCESSING (CLEAN REWRITE) ==========
                # Initialize chart_data with strategy equity
                chart_data = pd.DataFrame({'Strategy': equity.values}, index=equity.index)

                if not bench_df.empty:
                    log_info("Processing benchmarks...")

                    # Step 1: Prepare benchmark data
                    bench_df['timestamp'] = pd.to_datetime(bench_df['timestamp']).dt.tz_localize(None).dt.normalize()

                    # Step 2: Pivot to wide format
                    pivot_bench = bench_df.pivot(index='timestamp', columns='symbol', values='close')
                    log_info(f"Benchmark data range: {pivot_bench.index[0]} to {pivot_bench.index[-1]}")

                    # Step 3: Prepare equity index
                    clean_equity_index = pd.to_datetime(equity.index).tz_localize(None).normalize()
                    log_info(f"Strategy data range: {clean_equity_index[0]} to {clean_equity_index[-1]}")

                    # Step 4: Reindex benchmarks to match strategy dates
                    aligned_bench = pivot_bench.reindex(clean_equity_index)

                    # Step 5: Forward fill then backward fill to handle missing dates
                    aligned_bench = aligned_bench.ffill().bfill()

                    log_info(f"Aligned benchmark shape: {aligned_bench.shape}, NaN count: {aligned_bench.isna().sum().sum()}")

                    # Step 6: Normalize each benchmark to start at same equity as strategy
                    for b in benchmarks:
                        if b in aligned_bench.columns:
                            b_series = aligned_bench[b]

                            # Skip if all NaN
                            if b_series.isna().all():
                                log_warning(f"Benchmark {b} is all NaN - skipping")
                                continue

                            # Get first valid value
                            first_idx = b_series.first_valid_index()
                            if first_idx is not None:
                                b_start = b_series.loc[first_idx]

                                if b_start > 0:
                                    # Normalize: (price / start_price) * start_equity
                                    normalized_series = (b_series / b_start) * start_eq
                                    chart_data[b] = normalized_series.values
                                    log_info(f"✓ {b}: Start={b_start:.2f}, Chart Start={normalized_series.iloc[0]:.2f}")
                                else:
                                    log_warning(f"Benchmark {b} has invalid start value: {b_start}")
                            else:
                                log_warning(f"Benchmark {b} has no valid data")
                        else:
                            log_warning(f"Benchmark {b} not found in data")

                    log_info(f"Final chart_data columns: {chart_data.columns.tolist()}")
                    log_info(f"Final chart_data shape: {chart_data.shape}")
                else:
                    log_warning("Benchmark data is empty!")

                # Performance Metrics
                duration_years = (pd.to_datetime(e_str) - pd.to_datetime(s_str)).days / 365.25
                cagr = (end_eq / start_eq) ** (1/duration_years) - 1 if duration_years > 0 else 0
                
                # Max Drawdown
                rolling_max = equity.cummax()
                drawdown = (equity - rolling_max) / rolling_max
                max_dd = drawdown.min()
                
                # Sharpe Ratio
                daily_ret = equity.pct_change().dropna()
                if daily_ret.std() > 0:
                    sharpe = (daily_ret.mean() / daily_ret.std()) * (252 ** 0.5)
                else:
                    sharpe = 0.0
                    
                # Calmar Ratio
                calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0
                
                
                # Draw Metrics
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Final Equity", f"${end_eq:,.2f}", f"{total_ret:.2%}")
                m2.metric("CAGR", f"{cagr:.2%}")
                m3.metric("Sharpe", f"{sharpe:.2f}")
                m4.metric("Calmar", f"{calmar:.2f}")
                m5.metric("Max Drawdown", f"{max_dd:.2%}")
                
                st.subheader("Equity Curve vs Benchmarks")
                
                # Interactive Plotly Chart
                fig_eq = go.Figure()
                
                # Add Strategy
                fig_eq.add_trace(go.Scatter(
                    x=chart_data.index, 
                    y=chart_data['Strategy'], 
                    mode='lines', 
                    name='Strategy',
                    line=dict(color='#00CC96', width=2)
                ))
                
                # Add Benchmarks
                colors = {'SPY': '#636EFA', 'QQQ': '#EF553B', 'GLD': '#FECB52'}
                for col in chart_data.columns:
                    if col == 'Strategy': continue
                    fig_eq.add_trace(go.Scatter(
                        x=chart_data.index, 
                        y=chart_data[col], 
                        mode='lines', 
                        name=col,
                        line=dict(color=colors.get(col, '#AB63FA'), width=1)
                    ))
                    
                # Calculate Y-axis range with padding (like TradingView)
                all_values = []
                for col in chart_data.columns:
                    all_values.extend(chart_data[col].dropna().values)

                if all_values:
                    y_min = min(all_values)
                    y_max = max(all_values)
                    y_padding = (y_max - y_min) * 0.1  # 10% padding
                    y_range = [max(0, y_min - y_padding), y_max + y_padding]
                else:
                    y_range = [0, start_eq * 2]

                # Layout Constraints (TradingView-like)
                fig_eq.update_layout(
                    yaxis=dict(
                        range=y_range,  # Fixed range based on data
                        fixedrange=False,  # Allow zoom
                        rangemode='tozero',  # Don't go below zero
                        constraintoward='bottom'  # Constrain toward bottom (zero)
                    ),
                    xaxis=dict(
                        range=[chart_data.index[0], chart_data.index[-1]],  # Lock to data range
                        rangeslider=dict(visible=False),  # Hide range slider for cleaner look
                        type='date'
                    ),
                    xaxis_rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1M", step="month", stepmode="backward"),
                            dict(count=3, label="3M", step="month", stepmode="backward"),
                            dict(count=6, label="6M", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(step="all", label="All")
                        ]),
                        bgcolor='rgba(150, 150, 150, 0.1)',
                        activecolor='rgba(100, 100, 100, 0.3)'
                    ),
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=0, r=0, t=60, b=0),
                    dragmode='zoom'  # Default to zoom mode
                )

                # Config to restrict panning outside data range
                config = {
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                    'displaylogo': False
                }

                st.plotly_chart(fig_eq, width="stretch", config=config)
                
                # --- Portfolio Holdings ---
                st.subheader("Portfolio Holdings")
                if not holdings_df.empty:
                    # Format index as date string for cleaner display
                    h_display = holdings_df.copy()
                    h_display.index = pd.to_datetime(h_display.index).strftime('%Y-%m-%d')

                    # Latest portfolio (most recent date)
                    latest = h_display.iloc[-1]
                    latest_date = h_display.index[-1]
                    long_stocks = [s.strip() for s in str(latest.get('long', '')).split(',') if s.strip()]
                    st.markdown(f"**Latest selection ({latest_date}):** " + "  ".join([f"`{s}`" for s in long_stocks]))
                    if strategy_mode == 'long_short':
                        short_stocks = [s.strip() for s in str(latest.get('short', '')).split(',') if s.strip()]
                        st.markdown(f"**Short leg:** " + "  ".join([f"`{s}`" for s in short_stocks[:10]]) + (f" ...+{len(short_stocks)-10}" if len(short_stocks) > 10 else ""))

                    # Portfolio rotation (days where long composition changed)
                    if strategy_mode == 'long_only':
                        changed_mask = h_display['long'] != h_display['long'].shift(1)
                    else:
                        changed_mask = (h_display['long'] != h_display['long'].shift(1)) | (h_display['short'] != h_display['short'].shift(1))
                    rotation_df = h_display[changed_mask]

                    with st.expander(f"Portfolio Rotation Log ({len(rotation_df)} changes)", expanded=True):
                        cols_to_show = ['long'] if strategy_mode == 'long_only' else ['long', 'short']
                        st.dataframe(rotation_df[cols_to_show].tail(50), width="stretch")

                    with st.expander("Full Daily Holdings (last 252 trading days)", expanded=False):
                        cols_to_show = ['long'] if strategy_mode == 'long_only' else ['long', 'short']
                        st.dataframe(h_display[cols_to_show].tail(252), width="stretch")

                # --- Yearly Returns ---
                st.subheader("Yearly Returns")

                try:
                    # Ensure chart_data index is datetime
                    if not isinstance(chart_data.index, pd.DatetimeIndex):
                        chart_data.index = pd.to_datetime(chart_data.index)

                    # Calculate Yearly Returns
                    yearly_groups = chart_data.groupby(chart_data.index.year)
                    yearly_rets = {}

                    for year, data in yearly_groups:
                        if len(data) > 0:
                            start_vals = data.iloc[0]
                            end_vals = data.iloc[-1]
                            ret = (end_vals / start_vals) - 1
                            yearly_rets[year] = ret

                    if yearly_rets:
                        yearly_df = pd.DataFrame(yearly_rets).T  # Index is Year, Columns are Assets
                        yearly_df.index.name = 'Year'
                    else:
                        yearly_df = pd.DataFrame()
                        st.warning("No yearly data available")
                except Exception as e:
                    log_error(e, "Yearly Returns Calculation")
                    st.error(f"Failed to calculate yearly returns: {e}")
                    yearly_df = pd.DataFrame()
                
                st.subheader("Yearly Seasonality (Equity Curve Comparison)")
                
                fig_season = go.Figure()
                
                strat_series = chart_data['Strategy']
                years = strat_series.index.year.unique()
                
                # Color map for years (gradient or distinct)
                # changing years to string for legend
                
                for year in years:
                    subset = strat_series[strat_series.index.year == year]
                    if subset.empty: continue
                    
                    # Normalize to percentage return for that year
                    # Start at 0%
                    normalized = (subset / subset.iloc[0]) - 1
                    
                    # Create a unified timeline (year 2000 is a leap year, good for all dates)
                    # We map all dates to year 2000 keeping month/day
                    dates_2000 = subset.index.map(lambda d: d.replace(year=2000))
                    
                    fig_season.add_trace(go.Scatter(
                        x=dates_2000,
                        y=normalized,
                        mode='lines',
                        name=str(year),
                        hovertemplate='%{x|%b %d}: %{y:.2%}'
                    ))
                
                fig_season.update_layout(
                    title='Strategy Annual Performance Overlay',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return (%)',
                    yaxis_tickformat='.0%',
                    xaxis_tickformat='%b %d', # Month Day format
                    hovermode='x unified',
                    shapes=[dict(type="line", xref="paper", yref="y", x0=0, y0=0, x1=1, y1=0, line=dict(color="gray", width=1, dash="dash"))]
                )
                
                st.plotly_chart(fig_season, width="stretch")
                
                # Yearly Returns Table
                st.caption("Yearly Performance Metrics")
                st.dataframe(yearly_df.style.format("{:.2%}"), width="stretch")

                
                if not weight_history.empty and use_mwu:
                    st.subheader("Dynamic Factor Allocations (MWU)")
                    # Normalized Stacked Area Chart
                    fig = go.Figure()
                    # Filter out non-factor columns
                    ignore_cols = ['vol_scalar', 'realized_vol', 'date']
                    plot_cols = [c for c in weight_history.columns if c not in ignore_cols]
                    
                    for col in plot_cols:
                        fig.add_trace(go.Scatter(
                            x=weight_history.index, y=weight_history[col],
                            mode='lines', name=col, stackgroup='one', groupnorm='percent'
                        ))
                    fig.update_layout(title="Factor Weight Evolution (Normalized)", yaxis=dict(range=[0, 100]), hovermode='x unified')
                    st.plotly_chart(fig, width="stretch")
                
        except Exception as e:
            sys.stdout = original_stdout
            st.error(f"Backtest Failed: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            progress.progress(100)




