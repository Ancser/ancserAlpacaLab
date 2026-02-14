import pandas as pd
import numpy as np

def momentum(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """
    12-1 Month Momentum.
    Returns: (Close / Close_shifted(lookback)) - (Close / Close_shifted(skip))
    """
    return close.pct_change(lookback) - close.pct_change(skip)

def pullback(close: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Short-term reversal (negative return).
    """
    return -close.pct_change(lookback)

def rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index (0-100).
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def volatility(close: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Annualized Volatility.
    """
    return close.pct_change().rolling(period).std() * np.sqrt(252)

def volume_surge(volume: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    Volume / Moving Average Volume.
    """
    return volume / volume.rolling(period).mean()

def trend_strength(close: pd.DataFrame, short: int = 20, long: int = 50) -> pd.DataFrame:
    """
    (Short MA / Long MA) - 1.
    """
    sma_short = close.rolling(short).mean()
    sma_long = close.rolling(long).mean()
    return (sma_short / sma_long) - 1

def zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional Z-Score standardization.
    """
    return df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)

def rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional Percentile Rank (0.0 to 1.0).
    """
    return df.rank(axis=1, pct=True)
