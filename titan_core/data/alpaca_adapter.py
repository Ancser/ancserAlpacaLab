import os
import polars as pl
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment
from datetime import datetime
from typing import List
from .schema import MARKET_DATA_SCHEMA

class AlpacaAdapter:
    """
    Fetches data from Alpaca (Paid/Free) and adapts to Titan Schema.
    """
    def __init__(self):
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY")
        if not self.api_key:
            raise ValueError("Alpaca API keys missing.")
            
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)

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
