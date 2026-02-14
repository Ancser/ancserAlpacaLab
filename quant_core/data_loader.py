import os
import pickle
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DataLoader(ABC):
    """
    Abstract Base Class for Data Loading.
    Standardizes output to (close_df, volume_df).
    """

    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main entry point. Handles caching and calls implementation-specific download.
        
        Returns:
            close_df: (Date, Symbol) index
            volume_df: (Date, Symbol) index
        """
        # Cache key based on source, dates, and symbols hash
        symbols_hash = '_'.join(sorted(symbols))
        if len(symbols_hash) > 50:
             symbols_hash = f"{len(symbols)}_symbols_{hash(tuple(sorted(symbols)))}"
             
        cache_key = f"{self.__class__.__name__}_{start_date}_{end_date}_{symbols_hash}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if use_cache and cache_file.exists():
            logger.info(f"Loading data from cache: {cache_file.name}")
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                return data['close'], data['volume']
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_file}: {e}")

        logger.info(f"Downloading data for {len(symbols)} symbols from {start_date} to {end_date}")
        close, volume = self._download(symbols, start_date, end_date)
        
        # Standardize format
        close = self._standardize(close)
        volume = self._standardize(volume)

        # Basic cleaning
        close = close.ffill()
        volume = volume.fillna(0)

        if use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump({'close': close, 'volume': volume}, f)
            logger.info(f"Cached data to: {cache_file.name}")

        return close, volume

    @abstractmethod
    def _download(self, symbols: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Implementation specific download logic."""
        pass

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure index is DatetimeIndex and columns are symbols."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df

class YahooDataLoader(DataLoader):
    def _download(self, symbols: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        import yfinance as yf
        
        try:
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                group_by='column',
                threads=True
            )
            
            if len(symbols) == 1:
                 # YFinance structure differs for single symbol
                 sym = symbols[0]
                 close = data[['Close']].rename(columns={'Close': sym})
                 volume = data[['Volume']].rename(columns={'Volume': sym})
            else:
                if isinstance(data.columns, pd.MultiIndex):
                    close = data['Close'].copy()
                    volume = data['Volume'].copy()
                else:
                    # Fallback if structure is flat (unexpected for multiple symbols but possible)
                    close = data
                    volume = pd.DataFrame(0, index=close.index, columns=close.columns)

            return close, volume
            
        except Exception as e:
            logger.error(f"Yahoo download failed: {e}")
            raise

class AlpacaDataLoader(DataLoader):
    def __init__(self, api_key: str = None, secret_key: str = None, cache_dir: str = "data_cache"):
        super().__init__(cache_dir)
        self.api_key = api_key or os.getenv("APCA_API_KEY_ID")
        self.secret_key = secret_key or os.getenv("APCA_API_SECRET_KEY")
        
        if not self.api_key or not self.secret_key:
             # Just warning, might not be needed if using other loaders
             logger.warning("Alpaca credentials not provided or found in env.")

    def _download(self, symbols: List[str], start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed, Adjustment

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca credentials required for AlpacaDataLoader")

        client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        chunk_size = 50
        all_bars = []
        
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=chunk,
                    timeframe=TimeFrame.Day,
                    start=datetime.strptime(start_date, '%Y-%m-%d'),
                    end=datetime.strptime(end_date, '%Y-%m-%d'),
                    adjustment=Adjustment.ALL,
                    feed=DataFeed.IEX
                )
                bars = client.get_stock_bars(req).df
                if not bars.empty:
                    all_bars.append(bars)
            except Exception as e:
                logger.warning(f"Failed chunk {chunk[0]}...: {e}")
        
        if not all_bars:
             # Return empty DFs if no data found to avoid crash
             return pd.DataFrame(), pd.DataFrame()

        df = pd.concat(all_bars).reset_index()
        # Alpaca returns timestamp, we want date
        df['date'] = df['timestamp'].dt.date
        
        # Pivot
        close = df.pivot(index='date', columns='symbol', values='close')
        volume = df.pivot(index='date', columns='symbol', values='volume')
        
        return close, volume
