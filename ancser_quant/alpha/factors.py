import polars as pl
import numpy as np

# ==========================================
# 1. 基礎動量與反轉 (Momentum & Reversion)
# ==========================================

def ts_momentum(window: int = 252) -> pl.Expr:
    """
    Time-Series Momentum (ROC)
    Returns: (Close / Close_lag) - 1
    """
    return (pl.col("close") / pl.col("close").shift(window)) - 1

def cs_momentum(window: int = 20) -> pl.Expr:
    """
    Cross-Sectional Momentum (Rank)
    Returns: Rank of 1-month return across all stocks
    RANK(RETURN_1M)
    """
    # Note: Rank needs to be done over("timestamp") context in the main flow
    return pl.col("return_1m").rank()

def rsi(period: int = 14) -> pl.Expr:
    """
    Relative Strength Index (Polars implementation)
    """
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0)
    loss = -delta.clip(upper_bound=0)
    
    avg_gain = gain.rolling_mean(window_size=period)
    avg_loss = loss.rolling_mean(window_size=period)
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ==========================================
# 2. 高階矩因子 (Higher Moments - Jane Street Style)
# ==========================================

def realized_skew(window: int = 60) -> pl.Expr:
    """
    Realized Skewness of daily returns.
    Captures 'lottery-like' payoff potential.
    """
    return pl.col("returns").rolling_skew(window_size=window)

def idiosyncratic_vol(window: int = 20) -> pl.Expr:
    """
    Idiosyncratic Volatility (simplified).
    Residual volatility relative to market is hard in pure expr without beta.
    Here we use standard deviation of returns as a proxy for total vol.
    """
    return pl.col("returns").rolling_std(window_size=window)

def max_daily_return(window: int = 20) -> pl.Expr:
    """
    MAX: Maximum daily return in the past month.
    """
    return pl.col("returns").rolling_max(window_size=window)

# ==========================================
# 3. 微觀結構因子 (Microstructure)
# ==========================================

def amihud_illiquidity(window: int = 20) -> pl.Expr:
    """
    Amihud Illiquidity: Avg(|Ret| / (Price * Volume))
    High value = Illiquid.
    """
    term = pl.col("returns").abs() / (pl.col("close") * pl.col("volume"))
    return term.rolling_mean(window_size=window)

def spread_proxy(window: int = 1) -> pl.Expr:
    """
    High-Low Spread Proxy (Corwin-Schultz simplified)
    (High - Low) / Close
    """
    return (pl.col("high") - pl.col("low")) / pl.col("close")

# ==========================================
# 4. Alpha 101 (Mathematical Alphas)
# ==========================================

def alpha_006(window: int = 10) -> pl.Expr:
    """
    Alpha#6: -1 * correlation(open, volume, 10)
    """
    return -1 * pl.rolling_corr(pl.col("open"), pl.col("volume"), window_size=window)

def alpha_012() -> pl.Expr:
    """
    Alpha#12: sign(delta(volume, 1)) * (-1 * delta(close, 1))
    """
    return pl.col("volume").diff().sign() * (-1 * pl.col("close").diff())

# ==========================================
# Helper: Feature Engineering Pipeline
# ==========================================

def compute_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Applies common transformations needed for factors.
    Expected input sorted by [symbol, timestamp].
    """
    return df.with_columns([
        pl.col("close").pct_change().alias("returns"),
        ((pl.col("close") / pl.col("close").shift(20)) - 1).alias("return_1m")
    ])

def compute_all_factors(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Computes all registered factors and appends columns.
    """
    df = compute_features(df)
    
    # Time-series factors (per group)
    df = df.with_columns([
        ts_momentum(252).alias("factor_ts_mom"),
        rsi(14).alias("factor_rsi"),
        realized_skew(60).alias("factor_skew"),
        idiosyncratic_vol(20).alias("factor_ivol"),
        max_daily_return(20).alias("factor_max"),
        amihud_illiquidity(20).alias("factor_amihud"),
        spread_proxy().alias("factor_spread"),
        alpha_006().alias("factor_alpha006"),
        alpha_012().alias("factor_alpha012")
    ])

    # --- Drift Regime Logic (Singha 2025) ---
    # 1. Calculate positive day ratio (window=63)
    # 2. IF ratio > 0.60 THEN Regim=1 ELSE Regime=0
    
    df = df.with_columns([
        (pl.col("returns") > 0).cast(pl.Int32).alias("is_positive_day")
    ])
    
    df = df.with_columns([
        (pl.col("is_positive_day")
         .rolling_sum(window_size=63)
         .over("symbol") / 63).alias("positive_day_ratio")
    ])
    
    df = df.with_columns([
        (pl.col("positive_day_ratio") > 0.60).alias("in_drift_regime") 
    ])
    
    # Create Conditional Factors
    # FIXED: Use Reversion ONLY when NOT in drift regime
    # Theory: Avoid mean-reversion when stock is in strong uptrend (drift regime)
    # When in_drift_regime: Set to RSI median (50) to neutralize the factor
    # When NOT in_drift_regime: Use actual RSI for reversion signal
    df = df.with_columns([
        pl.when(~pl.col("in_drift_regime"))  # 非drift regime时
        .then(pl.col("factor_rsi"))          # 使用实际RSI反转信号
        .otherwise(50.0)                     # drift regime时使用中性值（不参与选股）
        .alias("factor_rsi_filtered")
    ])

    # --- Unicorn Edge (Singha 2025) ---
    # BASE = 0.7 * value(1/price, cross-sectional percentile)
    #      + 0.3 * reversal(-10d return, cross-sectional z-score)
    # EDGE = BASE * REGIME  (non-drift stocks zeroed out)

    # Step 1: time-series components (per symbol)
    df = df.with_columns([
        (1.0 / pl.col("close")).alias("_ue_inv_price"),
        (-(pl.col("close") / pl.col("close").shift(10).over("symbol") - 1.0)).alias("_ue_ret10d_neg"),
    ])

    # Step 2: cross-sectional normalization (per trading day, across all symbols)
    df = df.with_columns([
        # value: percentile rank of 1/price → [0, 1]
        (pl.col("_ue_inv_price").rank().over("timestamp") /
         pl.col("_ue_inv_price").count().over("timestamp")).alias("_ue_value_cs"),
        # reversal: z-score of negated 10-day return
        ((pl.col("_ue_ret10d_neg") - pl.col("_ue_ret10d_neg").mean().over("timestamp")) /
         (pl.col("_ue_ret10d_neg").std().over("timestamp") + 1e-8)).alias("_ue_reversal_cs"),
    ])

    # Step 3: BASE composite
    df = df.with_columns([
        (0.7 * pl.col("_ue_value_cs") + 0.3 * pl.col("_ue_reversal_cs")).alias("_ue_base"),
    ])

    # Step 4: EDGE = BASE * REGIME  (non-drift → 0, excluded from selection)
    df = df.with_columns([
        (pl.col("_ue_base") * pl.col("in_drift_regime").cast(pl.Float64)).alias("factor_unicorn_edge"),
    ])

    # Cross-sectional factors (needs window function over date)
    # Note: Polars LazyFrame execution order matters.
    # Window functions over 'timestamp' are expensive if data is partitioned by symbol.
    # Strategy: Compute TS factors first, then collect/repartition if needed, or use window function.

    # For now, we return TS factors. CS ranking happens in the Strategy/Ensemble step.

    return df
