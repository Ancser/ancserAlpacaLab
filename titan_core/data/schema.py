import polars as pl

# 統一數據結構 (Schema)
# 所有數據必須包含以下標準欄位，並強制類型轉換
MARKET_DATA_SCHEMA = {
    "timestamp": pl.Datetime,  # UTC 時間
    "symbol": pl.Categorical,  # 股票代碼 (Categorical 省內存)
    "open": pl.Float32,
    "high": pl.Float32,
    "low": pl.Float32,
    "close": pl.Float32,
    "volume": pl.Float32,
    "vwap": pl.Float32,        # 關鍵：機構算法交易核心
    "trade_count": pl.UInt32   # 關鍵：計算微觀結構因子
}

def enforce_schema(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensures the DataFrame conforms to the strict Titan schema.
    Casts types and handles missing columns (filling with nulls/zeros).
    """
    # 1. Cast existing columns
    for col, dtype in MARKET_DATA_SCHEMA.items():
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype))
        else:
            # Add missing columns with nulls (or sensible defaults)
            if dtype == pl.Categorical:
                pass # Symbol should exist
            elif dtype in [pl.Float32, pl.Float64]:
                 df = df.with_columns(pl.lit(0.0).cast(dtype).alias(col))
            elif dtype in [pl.UInt32, pl.Int32, pl.Int64]:
                 df = df.with_columns(pl.lit(0).cast(dtype).alias(col))
            elif dtype == pl.Datetime:
                 pass # Timestamp should exist
                 
    # 2. Select and Reorder
    return df.select(list(MARKET_DATA_SCHEMA.keys()))
