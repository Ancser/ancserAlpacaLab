import pandas as pd
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from ancser_quant.data.alpaca_adapter import AlpacaAdapter
from ancser_quant.backtest import BacktestEngine

class LiveStrategy:
    """
    Shared logic for calculating target portfolio state.
    Used by both the Dashboard (Preview) and Main Loop (Execution).
    """
    
    def __init__(self, account_name: str = "Main"):
        self.account_name = account_name
        self.alpaca = AlpacaAdapter(account_name)
        
    def calculate_targets(self, config: dict, current_equity: float = None) -> dict:
        """
        Calculate target weights and volatility scalar based on live config.
        """
        universe = config.get('universe', [])
        factors = config.get('active_factors', [])
        
        if not universe or not factors:
            return {"error": "Universe or Factors empty"}

        # 1. Volatility Targeting
        target_scalar = 1.0
        vol_metrics = {}
        
        use_vol_target = config.get('use_vol_target', False)
        vol_target = config.get('vol_target', 0.20)
        leverage_cap = config.get('leverage', 1.0)
        
        # We need history for both Vol and Factors
        # Momentum requires 252 trading days (~365 calendar days), plus buffer, so we fetch 400 days.
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=400)
        
        try:
            hist_pl = self.alpaca.fetch_history(universe, start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')).collect()
            
            if hist_pl.is_empty():
                 return {"error": "No historical data fetched"}
                 
            # 2. Factor Calculation using unified polars factors
            from ancser_quant.alpha.factors import compute_all_factors
            
            factor_df_pl = compute_all_factors(hist_pl.lazy()).collect()
            factor_df = factor_df_pl.to_pandas()
            factor_df['timestamp'] = pd.to_datetime(factor_df['timestamp'])
             
            # Pivot prices for Volatility Targeting and latest_prices
            hist = hist_pl.to_pandas()
            hist['timestamp'] = pd.to_datetime(hist['timestamp'])
            closes = hist.pivot(index='timestamp', columns='symbol', values='close')
            
            # Get latest available data for each symbol
            latest_date = factor_df['timestamp'].max()
            latest_data = factor_df[factor_df['timestamp'] == latest_date].set_index('symbol')
            
            # Initialize scores
            scores = pd.Series(0.0, index=closes.columns)
            
            # Factor directions mapping
            col_map = {
                'Momentum': 'factor_ts_mom',
                'Reversion': 'factor_rsi',
                'Skew': 'factor_skew',
                'Microstructure': 'factor_amihud',
                'Alpha 101': 'factor_alpha006',
                'Volatility': 'factor_ivol',
                'Drift-Reversion': 'factor_rsi_filtered',
                'Unicorn Edge': 'factor_unicorn_edge',
            }
            descending_factors = {'Reversion', 'Volatility', 'Microstructure', 'Drift-Reversion'}
            
            # Load MWU setting
            use_mwu = config.get('use_mwu', False)
            weight_per_factor = 1.0 / len(factors)
            factor_weights = {f: weight_per_factor for f in factors}
            
            # MWU Process
            if use_mwu:
                from ancser_quant.alpha.mwu import MWUEngine
                mwu = MWUEngine(factors)
                
                # We need historical factor ranks and forward returns to train MWU
                # To align with Backtest engine exactly:
                factor_df_mwu = factor_df_pl.with_columns([
                    (pl.col("close").shift(-1).over("symbol") / pl.col("close") - 1).alias("fwd_ret")
                ]).to_pandas()
                
                factor_df_mwu['timestamp'] = pd.to_datetime(factor_df_mwu['timestamp'])
                
                # Precompute structures like backtest
                dates = sorted(factor_df_mwu['timestamp'].unique())
                fwd_ret_pivot = factor_df_mwu.pivot_table(index='timestamp', columns='symbol', values='fwd_ret')
                
                factor_pivots = {}
                for f, col in col_map.items():
                    if f in factors and col in factor_df_mwu.columns:
                        factor_pivots[f] = factor_df_mwu.pivot_table(index='timestamp', columns='symbol', values=col)
                        
                for i, date in enumerate(dates[:-1]):
                    if i > 0:
                        day_ics = {}
                        if date in fwd_ret_pivot.index:
                            fwd_ret_row = fwd_ret_pivot.loc[date].dropna()
                            for f in factors:
                                if f in factor_pivots and date in factor_pivots[f].index:
                                    fac_row = factor_pivots[f].loc[date].dropna()
                                    common = fac_row.index.intersection(fwd_ret_row.index)
                                    if len(common) > 5:
                                        corr = fac_row[common].corr(fwd_ret_row[common], method='spearman')
                                        day_ics[f] = 0.0 if np.isnan(corr) else corr
                        mwu.update(date, day_ics)
                
                factor_weights = mwu.weights.copy()
            
            for f in factors:
                if f not in col_map:
                    continue
                col_name = col_map[f]
                
                if col_name in latest_data.columns:
                    # Some symbols might be missing in latest_data because of NaNs, map to closes.columns
                    factor_vals = latest_data[col_name].reindex(closes.columns)
                    ascending = f not in descending_factors
                    factor_rank = factor_vals.rank(pct=True, ascending=ascending)
                    scores += factor_rank.fillna(0.5) * factor_weights[f]

            # 3. Portfolio Construction
            # Select Top 5 Stocks
            top_n = 5
            top_stocks = scores.nlargest(top_n)
            
            # Calculate Target Weights
            # Equal Weight * Vol Scalar
            target_weight = (1.0 / top_n) * target_scalar
            
            allocations = {}
            for sym, score in top_stocks.items():
                allocations[sym] = target_weight
                
            return {
                "allocations": allocations, # Symbol -> Weight (e.g. 0.20)
                "vol_metrics": vol_metrics,
                "latest_prices": closes.iloc[-1].to_dict(),
                "factor_scores": scores.to_dict(),
                "factor_weights": factor_weights
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

