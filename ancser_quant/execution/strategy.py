import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ancser_quant.data.alpaca_adapter import AlpacaAdapter
from ancser_quant.backtest import BacktestEngine

class LiveStrategy:
    """
    Shared logic for calculating target portfolio state.
    Used by both the Dashboard (Preview) and Main Loop (Execution).

    Supports two strategy types via config['strategy_type']:
      'classic'       — existing multi-factor model (default)
      'unicorn_edge'  — Unicorn Edge (value + reversal, drift regime filter)
    """

    def __init__(self):
        self.alpaca = AlpacaAdapter()

    def calculate_targets(self, config: dict, current_equity: float = None) -> dict:
        """Route to the correct strategy based on config['strategy_type']."""
        strategy_type = config.get('strategy_type', 'classic')
        if strategy_type == 'unicorn_edge':
            return self._calculate_unicorn_edge(config)
        else:
            return self._calculate_classic(config)

    # ------------------------------------------------------------------
    # Classic multi-factor strategy
    # ------------------------------------------------------------------

    def _calculate_classic(self, config: dict) -> dict:
        universe = config.get('universe', [])
        factors = config.get('active_factors', [])

        if not universe or not factors:
            return {"error": "Universe or Factors empty"}

        target_scalar = 1.0
        vol_metrics = {}

        use_vol_target = config.get('use_vol_target', False)
        vol_target = config.get('vol_target', 0.20)
        leverage_cap = config.get('leverage', 1.0)

        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=90)

        try:
            hist_pl = self.alpaca.fetch_history(
                universe,
                start_dt.strftime('%Y-%m-%d'),
                end_dt.strftime('%Y-%m-%d')
            ).collect()

            if hist_pl.is_empty():
                return {"error": "No historical data fetched"}

            hist = hist_pl.to_pandas()
            hist['timestamp'] = pd.to_datetime(hist['timestamp'])

            closes = hist.pivot(index='timestamp', columns='symbol', values='close')

            scores = pd.Series(0.0, index=closes.columns)
            weight_per_factor = 1.0 / len(factors)

            for f in factors:
                factor_rank = pd.Series(0.0, index=closes.columns)

                if f == 'Momentum':
                    mom = closes.pct_change(126).iloc[-1]
                    factor_rank = mom.rank(pct=True, ascending=True)

                elif f == 'Reversion':
                    rev = closes.pct_change(5).iloc[-1]
                    factor_rank = rev.rank(pct=True, ascending=False)

                elif f == 'Volatility':
                    vol = closes.pct_change().tail(20).std()
                    factor_rank = vol.rank(pct=True, ascending=False)

                elif f == 'Skew':
                    skew = closes.pct_change().tail(60).skew()
                    factor_rank = skew.rank(pct=True, ascending=True)

                elif f == 'Microstructure':
                    rev1 = closes.pct_change(1).iloc[-1]
                    factor_rank = rev1.rank(pct=True, ascending=False)

                elif f == 'Drift-Reversion':
                    returns = closes.pct_change()
                    # Positive day ratio over 63 days
                    is_pos = (returns > 0).astype(float)
                    pos_ratio = is_pos.tail(63).mean()
                    in_drift = pos_ratio > 0.60
                    # RSI proxy: 14-day return magnitude
                    rsi_proxy = returns.tail(14).mean()
                    # Neutralize in drift regime
                    adjusted = rsi_proxy.copy()
                    adjusted[in_drift] = 0.0
                    factor_rank = adjusted.rank(pct=True, ascending=False)

                scores += factor_rank.fillna(0.5) * weight_per_factor

            top_n = 5
            top_stocks = scores.nlargest(top_n)
            target_weight = (1.0 / top_n) * target_scalar

            allocations = {sym: target_weight for sym in top_stocks.index}

            return {
                "allocations": allocations,
                "vol_metrics": vol_metrics,
                "latest_prices": closes.iloc[-1].to_dict(),
                "factor_scores": scores.to_dict()
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    # ------------------------------------------------------------------
    # Unicorn Edge strategy (long-only live variant)
    # ------------------------------------------------------------------

    def _calculate_unicorn_edge(self, config: dict) -> dict:
        from ancser_quant.alpha.unicorn_edge import compute_edge, compute_long_only_weights

        universe = config.get('universe', [])
        if not universe:
            return {"error": "Universe empty"}

        alpha = config.get('ue_alpha', 0.7)
        reversal_window = config.get('ue_reversal_window', 10)
        drift_window = config.get('ue_drift_window', 63)
        drift_threshold = config.get('ue_drift_threshold', 0.60)
        scale_factor = config.get('ue_scale_factor', 1.0)

        # Need enough history for drift_window + reversal_window
        lookback_days = max(drift_window, reversal_window) * 2 + 30
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=lookback_days)

        try:
            hist_pl = self.alpaca.fetch_history(
                universe,
                start_dt.strftime('%Y-%m-%d'),
                end_dt.strftime('%Y-%m-%d')
            ).collect()

            if hist_pl.is_empty():
                return {"error": "No historical data fetched"}

            hist = hist_pl.to_pandas()
            hist['timestamp'] = pd.to_datetime(hist['timestamp'])

            prices = hist.pivot(index='timestamp', columns='symbol', values='close')
            returns = prices.pct_change()

            # Compute EDGE for all stocks
            edge_df = compute_edge(
                prices, returns,
                alpha=alpha,
                reversal_window=reversal_window,
                drift_window=drift_window,
                drift_threshold=drift_threshold
            )

            latest_edge = edge_df.iloc[-1]
            weights = compute_long_only_weights(latest_edge)

            if weights.empty:
                return {"error": "No stocks in drift regime — all positions zeroed"}

            # Apply scale factor (capped at 1.0 for long-only to prevent over-leverage)
            effective_scale = min(scale_factor, 1.0)
            allocations = {sym: float(w) * effective_scale for sym, w in weights.items()}

            return {
                "allocations": allocations,
                "vol_metrics": {"final_scalar": effective_scale},
                "latest_prices": prices.iloc[-1].to_dict(),
                "factor_scores": latest_edge.to_dict()
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}
