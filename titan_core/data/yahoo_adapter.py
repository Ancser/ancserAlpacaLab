import yfinance as yf
import polars as pl
from .schema import MARKET_DATA_SCHEMA
from datetime import datetime
from typing import List

class YahooAdapter:
    """
    Fetches historical data from Yahoo Finance and converts to Polars LazyFrame.
    Primarily used for Backfill.
    """
    def fetch_history(self, symbols: List[str], start_date: str, end_date: str = None) -> pl.LazyFrame:
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        print(f"[YahooAdapter] Fetching {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Download using yfinance (Pandas based)
        # We download individually to avoid MultiIndex complexity with Polars for now,
        # or download all and process.
        
        # Optimized: Group download
        try:
            df_pandas = yf.download(
                symbols, 
                start=start_date, 
                end=end_date, 
                group_by='ticker', 
                auto_adjust=True,
                threads=True,
                progress=False
            )
        except Exception as e:
            print(f"[YahooAdapter] Download failed: {e}")
            return pl.LazyFrame({}) # Empty

        # Process into long format
        # Pandas MultiIndex (Ticker, Attributes) -> Ticker column
        
        frames = []
        if len(symbols) == 1:
             # Single ticker structure is flat
             sym = symbols[0]
             df_pandas['symbol'] = sym
             frames.append(pl.from_pandas(df_pandas.reset_index()))
        else: 
            # Multi-ticker
            for sym in symbols:
                try:
                    subset = df_pandas[sym].copy()
                    subset['symbol'] = sym
                    # Reset index to get Date column
                    subset = subset.reset_index()
                    frames.append(pl.from_pandas(subset))
                except KeyError:
                    pass # Symbol not found in data
        
        if not frames:
            return pl.LazyFrame({})
            
        # Concat all Polars DataFrames
        df_pl = pl.concat(frames)

        # Rename columns to match Schema (case insensitive check)
        # Yahoo: Date, Open, High, Low, Close, Volume
        
        df_pl = df_pl.rename({
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })

        # Feature Engineering for missing columns
        # VWAP approx = (High + Low + Close) / 3
        # Trade Count = 0 (Yahoo doesn't provide)
        
        df_pl = df_pl.with_columns([
            ( (pl.col("high") + pl.col("low") + pl.col("close")) / 3 ).alias("vwap"),
            pl.lit(0).cast(pl.UInt32).alias("trade_count")
        ])
        
        # Enforce Types
        df_pl = df_pl.with_columns([
            pl.col("timestamp").cast(pl.Datetime),
            pl.col("symbol").cast(pl.Categorical),
            pl.col("open").cast(pl.Float32),
            pl.col("high").cast(pl.Float32),
            pl.col("low").cast(pl.Float32),
            pl.col("close").cast(pl.Float32),
            pl.col("volume").cast(pl.Float32),
            pl.col("vwap").cast(pl.Float32),
            pl.col("trade_count").cast(pl.UInt32)
        ])
        
        # Filter 0 volume
        df_pl = df_pl.filter(pl.col("volume") > 0)
        
        # Sort
        df_pl = df_pl.sort(["symbol", "timestamp"])

        return df_pl.lazy()
