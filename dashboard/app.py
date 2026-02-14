import streamlit as st
import polars as pl
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Page Setup
st.set_page_config(page_title="Project Titan", layout="wide", page_icon="⚡")

st.title("⚡ Project Titan: High-Performance Quant System")

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
    
    col1, col2, col3, col4 = st.columns(4)
    data = load_mock_data()
    current_equity = data['equity'].iloc[-1]
    last_equity = data['equity'].iloc[-2]
    pnl = current_equity - last_equity
    pnl_pct = (current_equity / last_equity) - 1
    
    col1.metric("Total Equity", f"${current_equity:,.2f}", f"{pnl_pct:.2%}")
    col2.metric("Daily P&L", f"${pnl:,.2f}", f"{pnl_pct:.2%}")
    col3.metric("Buying Power", "$50,000", "0%")
    col4.metric("Market Regime", "Bullish", "Risk On")
    
    st.subheader("Equity Curve")
    st.line_chart(data)
    
    st.subheader("Current Positions")
    # Mock Positions
    positions = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'NVDA', 'TSLA'],
        'Qty': [50, 20, 10, 5],
        'Value': [8500, 7200, 6500, 1200],
        'P&L %': [5.2, 1.4, 12.8, -2.1]
    })
    st.dataframe(positions, use_container_width=True)

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
elif page == "Backtest":
    st.header("Strategy Backtester (Polars Engine)")
    
    with st.form("backtest_config"):
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date", datetime.today() - timedelta(days=365))
        end_date = col2.date_input("End Date", datetime.today())
        
        factors = st.multiselect("Active Factors", ['Momentum', 'Reversion', 'Skew', 'Alpha101'], default=['Momentum', 'Reversion'])
        leverage = st.slider("Max Leverage", 1.0, 3.0, 1.5)
        
        submitted = st.form_submit_button("Run Backtest")
        
        if submitted:
            st.info(f"Running backtest from {start_date} to {end_date} with {factors}...")
            # Here we would call titan_core.backtest.run()
            time.sleep(1)
            st.success("Backtest Completed!")
            st.line_chart(load_mock_data())

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
