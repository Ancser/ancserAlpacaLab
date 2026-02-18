import numpy as np
import pandas as pd


# ============================================================
# SIGNAL CONSTRUCTION
# ============================================================

def compute_value_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Price-based value: inverse price, converted to cross-sectional percentile rank.
    prices: DataFrame (dates x tickers), daily closing prices
    returns: DataFrame of ranks (0 to 1)
    """
    inv_price = 1.0 / prices
    ranks = inv_price.rank(axis=1, pct=True)
    return ranks


def compute_reversal_signal(returns: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Short-term reversal: negate trailing window-day return, z-score cross-sectionally.
    returns: DataFrame of daily returns (dates x tickers)
    """
    trailing_ret = returns.rolling(window).sum()
    reversal_raw = -trailing_ret  # contrarian: buy losers

    mean = reversal_raw.mean(axis=1)
    std = reversal_raw.std(axis=1)
    z = reversal_raw.sub(mean, axis=0).div(std, axis=0)
    return z


def compute_base_factor(prices: pd.DataFrame,
                        returns: pd.DataFrame,
                        alpha: float = 0.7,
                        reversal_window: int = 10) -> pd.DataFrame:
    """
    BASE = alpha * value + (1-alpha) * reversal
    """
    value = compute_value_signal(prices)
    reversal = compute_reversal_signal(returns, window=reversal_window)
    base = alpha * value + (1 - alpha) * reversal
    return base


# ============================================================
# REGIME FILTER
# ============================================================

def compute_drift_regime(returns: pd.DataFrame,
                         window: int = 63,
                         threshold: float = 0.60) -> pd.DataFrame:
    """
    REGIME(i,t) = 1 if fraction of positive-return days in trailing window > threshold.
    Only trade when stock is in a sustained uptrend (drift regime).
    returns: DataFrame of daily returns (dates x tickers)
    """
    is_positive = (returns > 0).astype(float)
    up_fraction = is_positive.rolling(window).mean()
    regime = (up_fraction > threshold).astype(float)
    return regime


# ============================================================
# UNICORN EDGE
# ============================================================

def compute_edge(prices: pd.DataFrame,
                 returns: pd.DataFrame,
                 alpha: float = 0.7,
                 reversal_window: int = 10,
                 drift_window: int = 63,
                 drift_threshold: float = 0.60) -> pd.DataFrame:
    """
    EDGE(i,t) = BASE(i,t) * REGIME(i,t)
    Stocks not in drift regime get zero score (excluded from portfolio).
    """
    base = compute_base_factor(prices, returns, alpha, reversal_window)
    regime = compute_drift_regime(returns, drift_window, drift_threshold)
    edge = base * regime
    return edge


# ============================================================
# PORTFOLIO CONSTRUCTION
# ============================================================

def compute_portfolio_weights(edge: pd.Series) -> pd.Series:
    """
    Given a cross-section of EDGE scores for one day,
    construct market-neutral long-short weights.
    Long side sums to +0.5, short side sums to -0.5.
    """
    active = edge[edge != 0].dropna()
    if len(active) == 0:
        return pd.Series(dtype=float)

    z = (active - active.mean()) / active.std()

    long_mask = z > 0
    short_mask = z < 0

    long_z = z[long_mask]
    short_z = z[short_mask]

    weights = pd.Series(0.0, index=active.index)

    if long_z.sum() != 0:
        weights[long_mask] = (long_z / long_z.sum()) * 0.5

    if short_z.sum() != 0:
        weights[short_mask] = (short_z / short_z.abs().sum()) * (-0.5)

    return weights


def compute_long_only_weights(edge: pd.Series) -> pd.Series:
    """
    Long-only variant for live trading: take only positive EDGE stocks,
    normalize weights to sum to 1.0.
    """
    active = edge[edge != 0].dropna()
    if len(active) == 0:
        return pd.Series(dtype=float)

    z = (active - active.mean()) / (active.std() + 1e-9)
    long_mask = z > 0
    long_z = z[long_mask]

    if len(long_z) == 0 or long_z.sum() == 0:
        return pd.Series(dtype=float)

    weights = long_z / long_z.sum()
    return weights


# ============================================================
# RISK SCALING (train-time, frozen for OOS)
# ============================================================

def compute_scale_factor(portfolio_returns: pd.Series,
                         target_vol: float = 0.12,
                         target_max_dd: float = 0.15) -> float:
    """
    Scale factor to satisfy both:
    - annualized vol <= target_vol
    - max drawdown <= target_max_dd
    Computed once on training data, applied frozen to test.
    """
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    vol_scale = target_vol / ann_vol if ann_vol > 0 else 1.0

    cumulative = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_dd = abs(drawdown.min())
    dd_scale = target_max_dd / max_dd if max_dd > 0 else 1.0

    return min(vol_scale, dd_scale)


# ============================================================
# KILL SWITCH
# ============================================================

class KillSwitch:
    def __init__(self,
                 abs_dd_threshold: float = -0.30,
                 rolling_loss_threshold: float = -0.10,
                 rolling_window: int = 63):
        self.abs_dd_threshold = abs_dd_threshold
        self.rolling_loss_threshold = rolling_loss_threshold
        self.rolling_window = rolling_window
        self.peak_value = 1.0
        self.current_value = 1.0
        self.daily_returns = []
        self.is_active = True

    def update(self, daily_return: float) -> bool:
        """
        Update state with new return.
        Returns True if strategy should trade, False if killed.
        """
        if not self.is_active:
            return False

        self.current_value *= (1 + daily_return)
        self.daily_returns.append(daily_return)
        self.peak_value = max(self.peak_value, self.current_value)

        abs_dd = (self.current_value - self.peak_value) / self.peak_value
        if abs_dd < self.abs_dd_threshold:
            self.is_active = False
            return False

        if len(self.daily_returns) >= self.rolling_window:
            window_ret = self.daily_returns[-self.rolling_window:]
            rolling_pnl = (1 + pd.Series(window_ret)).prod() - 1
            if rolling_pnl < self.rolling_loss_threshold:
                self.is_active = False
                return False

        return True


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(prices: pd.DataFrame,
                 train_start: str, train_end: str,
                 test_start: str, test_end: str,
                 transaction_cost_bps: float = 0.6,
                 alpha: float = 0.7,
                 reversal_window: int = 10,
                 drift_window: int = 63,
                 drift_threshold: float = 0.60) -> dict:
    """
    Walk-forward backtest for one window.
    Returns dict with test period returns, Sharpe, max drawdown, etc.
    """
    returns = prices.pct_change()

    edge = compute_edge(prices, returns, alpha, reversal_window, drift_window, drift_threshold)

    # Training phase: compute scale factor
    train_returns_list = []
    prev_weights = pd.Series(dtype=float)

    train_dates = edge.loc[train_start:train_end].index
    for date in train_dates:
        w = compute_portfolio_weights(edge.loc[date])
        if len(prev_weights) > 0:
            turnover = w.subtract(prev_weights, fill_value=0).abs().sum()
            cost = turnover * transaction_cost_bps / 10000
        else:
            cost = 0.0

        next_day_idx = prices.index.get_loc(date) + 1
        if next_day_idx < len(prices):
            next_date = prices.index[next_day_idx]
            day_ret = (returns.loc[next_date] * w).sum() - cost
            train_returns_list.append(day_ret)
        prev_weights = w

    train_port_returns = pd.Series(train_returns_list)
    scale = compute_scale_factor(train_port_returns) if len(train_port_returns) > 20 else 1.0

    # Test phase: apply frozen scale + kill switch
    kill_switch = KillSwitch()
    test_returns_list = []
    test_dates = edge.loc[test_start:test_end].index
    prev_weights = pd.Series(dtype=float)

    for date in test_dates:
        if not kill_switch.is_active:
            test_returns_list.append(0.0)
            continue

        w = compute_portfolio_weights(edge.loc[date]) * scale
        if len(prev_weights) > 0:
            turnover = w.subtract(prev_weights, fill_value=0).abs().sum()
            cost = turnover * transaction_cost_bps / 10000
        else:
            cost = 0.0

        next_day_idx = prices.index.get_loc(date) + 1
        if next_day_idx < len(prices):
            next_date = prices.index[next_day_idx]
            day_ret = (returns.loc[next_date] * w).sum() - cost
            kill_switch.update(day_ret)
            test_returns_list.append(day_ret)
        prev_weights = w

    test_port_returns = pd.Series(
        test_returns_list,
        index=test_dates[:len(test_returns_list)]
    )

    ann_ret = (1 + test_port_returns).prod() ** (252 / max(len(test_port_returns), 1)) - 1
    ann_vol = test_port_returns.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cumulative = (1 + test_port_returns).cumprod()
    rolling_max = cumulative.cummax()
    max_dd = ((cumulative - rolling_max) / rolling_max).min()

    train_sharpe = 0.0
    if len(train_port_returns) > 1 and train_port_returns.std() > 0:
        train_sharpe = (train_port_returns.mean() * 252) / (train_port_returns.std() * np.sqrt(252))

    return {
        "returns": test_port_returns,
        "equity_curve": (1 + test_port_returns).cumprod(),
        "sharpe": sharpe,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "max_drawdown": max_dd,
        "scale_factor": scale,
        "train_sharpe": train_sharpe,
        "kill_switch_active": kill_switch.is_active,
    }
