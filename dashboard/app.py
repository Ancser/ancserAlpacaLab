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
        
        submitted = st.form_submit_button("Run Backtest")
        
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
                    st.line_chart(chart_data)
                    
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
