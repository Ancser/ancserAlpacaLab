from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
import polars as pl
from dotenv import load_dotenv
import os

load_dotenv() # Force load
from datetime import datetime
from typing import List, Dict
from .schema import MARKET_DATA_SCHEMA

class AlpacaAdapter:
    """
    Fetches data from Alpaca (Paid/Free) and adapts to Titan Schema.
    Also handles Account info.
    """
    def __init__(self):
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY")
        if not self.api_key:
            raise ValueError("Alpaca API keys missing.")
            
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True) # Default to paper

    def get_account(self) -> Dict:
        """Fetch account information."""
        try:
            acct = self.trading_client.get_account()
            return {
                'equity': float(acct.equity),
                'buying_power': float(acct.buying_power),
                'cash': float(acct.cash),
                'daytrade_count': int(acct.daytrade_count),
                'status': acct.status,
                'currency': acct.currency
            }
        except Exception as e:
            print(f"[AlpacaAdapter] Get Account Error: {e}")
            return {'equity': 0.0, 'buying_power': 0.0, 'status': 'Error'}


    def fetch_history(self, symbols: List[str], start_date: str, end_date: str = None) -> pl.LazyFrame:
        print(f"[AlpacaAdapter] Fetching {len(symbols)} symbols...")
        
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=datetime.strptime(start_date, '%Y-%m-%d'),
            end=datetime.strptime(end_date, '%Y-%m-%d'),
            adjustment=Adjustment.ALL,
            feed=DataFeed.IEX # or SIP if paid
        )
        
        try:
            bars = self.data_client.get_stock_bars(req).df
        except Exception as e:
            print(f"[AlpacaAdapter] Error: {e}")
            return pl.LazyFrame({})
            
        if bars.empty:
            return pl.LazyFrame({})
            
        # Reset index to get timestamp and symbol as columns
        bars = bars.reset_index()
        
        # Convert to Polars
        df = pl.from_pandas(bars)
        
        # Rename and Cast
        # Alpaca: timestamp, symbol, open, high, low, close, volume, trade_count, vwap
        
        df = df.with_columns([
            pl.col("timestamp").alias("timestamp"), # already correct name
            pl.col("symbol").cast(pl.Categorical),
            pl.col("open").cast(pl.Float32),
            pl.col("high").cast(pl.Float32),
            pl.col("low").cast(pl.Float32),
            pl.col("close").cast(pl.Float32),
            pl.col("volume").cast(pl.Float32),
            pl.col("vwap").cast(pl.Float32),
            pl.col("trade_count").cast(pl.UInt32)
        ])
        
        # Select standard columns
        df = df.select(list(MARKET_DATA_SCHEMA.keys()))
        
        return df.lazy()

    def get_positions(self) -> List[Dict]:
        """Fetch current open positions."""
        try:
            positions = self.trading_client.get_all_positions()
            return [{
                'Symbol': p.symbol,
                'Qty': float(p.qty),
                'Market Value': float(p.market_value),
                'Avg Entry': float(p.avg_entry_price),
                'Current Price': float(p.current_price),
                'Unrealized P&L': float(p.unrealized_pl),
                'P&L %': float(p.unrealized_plpc) * 100
            } for p in positions]
        except Exception as e:
            print(f"[AlpacaAdapter] Get Positions Error: {e}")
            return []

    def get_portfolio_history(self) -> pl.DataFrame:
        """Fetch portfolio history (Equity Curve)."""
        import pandas as pd
        try:
            # Fetch last 30 days or suitable period
            # period: 1M, 1A, etc. timeframe: 1D
            history = self.trading_client.get_account_portfolio_history(period="1M", timeframe="1D")
            
            # Check if history is valid
            if not history or not hasattr(history, 'timestamp'):
                return pl.DataFrame()

            # Alpaca returns lists
            data = {
                'timestamp': history.timestamp,
                'equity': history.equity,
                'profit_loss': history.profit_loss,
                'profit_loss_pct': history.profit_loss_pct
            }
            
            df = pd.DataFrame(data)
            
            # Convert timestamp (Unix epoch seconds)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Convert to Polars for consistency or keep Pandas for Streamlit
            # Streamlit likes Pandas.
            return df.set_index('timestamp')
        except Exception as e:
            print(f"[AlpacaAdapter] Get History Error: {e}")
            return pd.DataFrame()
