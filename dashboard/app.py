import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv

load_dotenv() # Load environments from .env file

# Page Setup
st.set_page_config(page_title="AncserAlpacaLab", layout="wide", page_icon=None)

st.title("ancserAlpacaLab")

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Dashboard", "Factor Lab", "Backtest", "Event Log"])

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
    from titan_core.data.alpaca_adapter import AlpacaAdapter
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
    
    st.subheader("Equity Curve (1 Month)")
    try:
        hist_df = adapter.get_portfolio_history()
        if not hist_df.empty:
            st.area_chart(hist_df['equity'], color='#00CC96') # Standard Alpaca Green-ish
        else:
            st.info("No portfolio history available.")
    except Exception as e:
        st.error(f"Failed to load chart: {e}")
    
    st.subheader("Holdings (Real)")
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
                use_container_width=True
            )
        else:
            st.info("No open positions.")
    except Exception as e:
        st.error(f"Failed to load holdings: {e}")


# --- Page: Factor Lab ---
elif page == "Factor Lab":
    st.header("Factor Lab (Alpha Research)")
    
    tab1, tab2 = st.tabs(["MWU Weights", "Factor IC"])
    
    with tab1:
        st.subheader("Dynamic Factor Weights (MWU)")
        # Mock Weights History
        dates = pd.date_range(end=datetime.today(), periods=30)
        weights = pd.DataFrame(np.random.dirichlet(np.ones(4), size=30), 
                               columns=['Momentum', 'Reversion', 'Skew', 'Microstructure'],
                               index=dates)
        
        st.area_chart(weights)
        st.caption("Shows how the system automatically reallocates capital to performing factors.")

    with tab2:
        st.subheader("Information Coefficient (IC) Heatmap")
        st.write("Correlation between factor values and forward returns.")
        # Mock IC
        ic_data = pd.DataFrame(np.random.randn(10, 4) * 0.1, 
                               columns=['Momentum', 'Reversion', 'Skew', 'Microstructure'],
                               index=[f"Day -{i}" for i in range(10)])
        st.dataframe(ic_data.style.background_gradient(cmap='RdYlGn', vmin=-0.2, vmax=0.2))

# --- Page: Backtest ---

# --- Page: Backtest ---
elif page == "Backtest":
    st.header("Strategy Backtester (Polars Engine)")
    
    with st.form("backtest_config"):
        col1, col2 = st.columns(2)
        # Default start date 2018-01-01
        start_date = col1.date_input("Start Date", datetime(2018, 1, 1))
        end_date = col2.date_input("End Date", datetime.today())
        
        # Define Universe (Tech Titans)
        universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'SPY', 'QQQ']
        st.write(f"Universe: {len(universe)} symbols (Tech Titans)")
        
        all_factors = ['Momentum', 'Reversion', 'Skew', 'Microstructure', 'Alpha 101', 'Volatility']
        factors = st.multiselect("Active Factors", all_factors, default=['Momentum', 'Reversion'])
        
        c3, c4 = st.columns(2)
        leverage = c3.slider("Max Leverage", 1.0, 3.0, 1.0)
        use_mwu = c4.checkbox("Enable MWU (Dynamic Weighting)", value=True)
        
        col_btn1, col_btn2 = st.columns([1,2])
        submitted = col_btn1.form_submit_button("Run Backtest")
        run_combo = col_btn2.form_submit_button("Run Factor Combinatorial Search (Loop All)")
        
        if run_combo:
            st.info("Starting Combinatorial Search... This may take a minute.")
            import itertools
            from titan_core.backtest import BacktestEngine
            from titan_core.data.yahoo_adapter import YahooAdapter
            
            # 1. Fetch & Prepare Data ONCE
            progress = st.progress(0)
            engine = BacktestEngine(initial_capital=100000.0)
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
            
            # 3. Loop
            total_steps = len(combos)
            
            for idx, combo in enumerate(combos):
                combo_name = " + ".join(combo)
                # Run Simulation
                res, _ = engine.run_simulation(data, list(combo), leverage, use_mwu)
                
                if not res.empty:
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
                    
                    results_list.append({
                        'Combination': combo_name,
                        'Factors': len(combo),
                        'Final Equity': end_eq,
                        'CAGR': cagr,
                        'Sharpe': sharpe,
                        'MDD': mdd,
                        'Calmar': calmar
                    })
                    
                    # Store curve for later plotting
                    eq_curves[combo_name] = eq
                
                progress.progress((idx + 1) / total_steps)
            
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
                    pivot_bench = bench_df.pivot(index='timestamp', columns='symbol', values='close')
                    pivot_bench = pivot_bench.reindex(eq_curves[top_5[0]].index, method='ffill')
                    
                    b_colors = {'SPY': 'gray', 'QQQ': 'silver', 'GLD': 'gold'}
                    for b in benchmarks:
                        if b in pivot_bench:
                             # Normalize to Strategy Initial Capital
                            b_start = pivot_bench[b].iloc[0]
                            if b_start > 0:
                                b_norm = (pivot_bench[b] / b_start) * 100000.0
                                fig_combo.add_trace(go.Scatter(
                                    x=b_norm.index,
                                    y=b_norm,
                                    mode='lines',
                                    name=b,
                                    line=dict(color=b_colors.get(b, 'gray'), width=1, dash='dot')
                                ))

                fig_combo.update_layout(
                    title="Best Combinations vs Benchmarks",
                    xaxis_title="Date",
                    yaxis_title="Equity",
                    hovermode="x unified",
                     yaxis=dict(rangemode='nonnegative')
                )
                st.plotly_chart(fig_combo, use_container_width=True)
                
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
                from titan_core.backtest import BacktestEngine
                from titan_core.data.yahoo_adapter import YahooAdapter
                
                engine = BacktestEngine(initial_capital=100000.0)
                
                # Convert dates to string
                s_str = start_date.strftime('%Y-%m-%d')
                e_str = end_date.strftime('%Y-%m-%d')
                
                progress.progress(10)
                
                # 1. Run Strategy Backtest
                results, weight_history = engine.run(universe, s_str, e_str, factors, leverage, use_mwu)
                
                progress.progress(50)
                
                # 2. Fetch Benchmarks (SPY, QQQ, GLD)
                benchmarks = ['SPY', 'QQQ', 'GLD']
                print(f"Fetching benchmarks: {benchmarks}")
                yahoo = YahooAdapter()
                bench_df_lazy = yahoo.fetch_history(benchmarks, s_str, e_str)
                bench_df = bench_df_lazy.collect().to_pandas()
                
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
                    
                    # Benchmarks Processing
                    chart_data = pd.DataFrame({'Strategy': equity})
                    
                    if not bench_df.empty:
                        bench_df['timestamp'] = pd.to_datetime(bench_df['timestamp'])
                        pivot_bench = bench_df.pivot(index='timestamp', columns='symbol', values='close')
                        
                        # Align benchmarks to strategy dates
                        pivot_bench = pivot_bench.reindex(equity.index, method='ffill')
                        
                        # Normalize to Strategy Initial Capital
                        for b in benchmarks:
                            if b in pivot_bench:
                                # Start at same capital
                                b_start = pivot_bench[b].iloc[0]
                                if not pd.isna(b_start) and b_start > 0:
                                    chart_data[b] = (pivot_bench[b] / b_start) * start_eq

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
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Final Equity", f"${end_eq:,.2f}", f"{total_ret:.2%}")
                    m2.metric("CAGR", f"{cagr:.2%}")
                    m3.metric("Sharpe", f"{sharpe:.2f}")
                    m4.metric("Calmar", f"{calmar:.2f}", f"MDD: {max_dd:.2%}")
                    
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
                        
                    # Layout Constraints
                    fig_eq.update_layout(
                        yaxis=dict(
                            rangemode='nonnegative', # Prevent negative Y
                            autorange=True,
                            fixedrange=False
                        ),
                        xaxis=dict(
                            range=[chart_data.index[0], chart_data.index[-1]], # Set initial view
                            constrain='domain' # Restrict panning? (Plotly doesn't strictly lock pan without config)
                        ),
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    
                    # config={'scrollZoom': True} to allow zoom, but maybe user wants to lock it?
                    # "limit left/right out of bounds" -> typically means distinct range.
                    
                    st.plotly_chart(fig_eq, use_container_width=True)
                    
                    # --- Yearly Returns ---
                    st.subheader("Yearly Returns")
                    
                    # Calculate Yearly Returns
                    # Group by Year and compute return (End / Start - 1)
                    yearly_groups = chart_data.groupby(chart_data.index.year)
                    yearly_rets = {}
                    
                    for year, data in yearly_groups:
                        start_vals = data.iloc[0]
                        end_vals = data.iloc[-1]
                        ret = (end_vals / start_vals) - 1
                        yearly_rets[year] = ret
                        
                    yearly_df = pd.DataFrame(yearly_rets).T # Index is Year, Columns are Assets
                    yearly_df.index.name = 'Year'
                    
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
                    
                    st.plotly_chart(fig_season, use_container_width=True)
                    
                    # Yearly Returns Table
                    st.caption("Yearly Performance Metrics")
                    st.dataframe(yearly_df.style.format("{:.2%}"), use_container_width=True)

                    
                    if not weight_history.empty and use_mwu:
                        st.subheader("Dynamic Factor Allocations (MWU)")
                        # Normalized Stacked Area Chart
                        fig = go.Figure()
                        for col in weight_history.columns:
                            fig.add_trace(go.Scatter(
                                x=weight_history.index, y=weight_history[col],
                                mode='lines', name=col, stackgroup='one', groupnorm='percent'
                            ))
                        fig.update_layout(title="Factor Weight Evolution (Normalized)", yaxis=dict(range=[0, 100]), hovermode='x unified')
                        st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                sys.stdout = original_stdout
                st.error(f"Backtest Failed: {e}")
                import traceback
                st.code(traceback.format_exc())
            finally:
                progress.progress(100)



# --- Page: Event Log ---
elif page == "Event Log":
    st.header("System Events")
    logs = pd.DataFrame({
        'Timestamp': [datetime.now() - timedelta(minutes=i*15) for i in range(10)],
        'Level': ['INFO', 'INFO', 'WARNING', 'INFO'] * 2 + ['INFO', 'INFO'],
        'Message': [
            'Heartbeat check passed',
            'Rebalance complete',
            'Slippage warning on TSLA order',
            'Factor calculation finished',
            'Market data updated',
            'System started',
            'Config loaded',
            'Database connected',
            'User login',
            'Daily maintenance'
        ]
    })
    st.dataframe(logs, use_container_width=True)
