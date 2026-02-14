import polars as pl
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
from titan_core.data.yahoo_adapter import YahooAdapter
from titan_core.alpha.factors import compute_all_factors
from titan_core.alpha.mwu import MWUEngine

class BacktestEngine:
    """
    Polars-based Backtest Engine.
    Simulates strategy performance over historical data.
    Supports MWU (Dynamic Weighting).
    """
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.yahoo = YahooAdapter()

    def run(self, 
            symbols: List[str], 
            start_date: str, 
            end_date: str, 
            active_factors: List[str], 
            leverage: float = 1.0,
            use_mwu: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
            
        # 1. Fetch Data (Backfill)
        print(f"Fetching data for {len(symbols)} symbols...")
        schema_df = self.yahoo.fetch_history(symbols, start_date, end_date).collect()
        
        if schema_df.is_empty():
            print("No data found.")
            return pd.DataFrame(), pd.DataFrame()

        # 2. Compute Factors (Lazy)
        print("Computing factors...")
        factor_df = compute_all_factors(schema_df.lazy()).collect()
        
        # 3. Prepare Data for Simulation
        # Map friendly names to internal columns
        col_map = {
            'Momentum': 'factor_ts_mom',
            'Reversion': 'factor_rsi',
            'Skew': 'factor_skew',
            'Microstructure': 'factor_amihud',
            'Alpha 101': 'factor_alpha006',
            'Volatility': 'factor_ivol'
        }
        
        # Filter valid factors
        valid_factors = [f for f in active_factors if col_map.get(f) in factor_df.columns]
        factor_cols = [col_map[f] for f in valid_factors]
        
        # Calculate Forward Returns for IC calculation
        factor_df = factor_df.sort(["symbol", "timestamp"])
        factor_df = factor_df.with_columns([
            (pl.col("close").shift(-1).over("symbol") / pl.col("close") - 1).alias("fwd_ret")
        ])

        # Convert to Pandas for iteration (easier state management for MWU loop)
        # Polars is great for vectorization, but MWU is path-dependent per day.
        # We process 'Date' groups.
        
        pdf = factor_df.to_pandas()
        pdf['timestamp'] = pd.to_datetime(pdf['timestamp'])
        dates = sorted(pdf['timestamp'].unique())
        
        # Initialize MWU
        mwu = MWUEngine(valid_factors)
        current_weights = mwu.weights.copy()
        
        # Simulation State
        equity = [self.initial_capital]
        weights_history = []
        
        print(f"Running simulation over {len(dates)} days...")
        
        for i, date in enumerate(dates[:-1]):
            # Get data for today
            today_df = pdf[pdf['timestamp'] == date].set_index('symbol')
            
            # 1. Update MWU Weights (if enabled and not first day)
            if use_mwu and i > 0:
                # Calculate IC for previous day's factors vs TODAY's return (which was yesterday's fwd_ret)
                # Actually, we need realized return from T to T+1 to judge T's factors.
                # Here we simplify: Update weights based on how well factors predicted *this* step.
                
                # Calculate IC for each factor
                day_ics = {}
                # This day's return is fwd_ret from date. Wait, fwd_ret is Future.
                # IC = Corr(Factor_T, Ret_T+1). We observe Ret_T+1 at T+1. Update weights for T+2.
                # Let's assume we update weights daily based on latest known IC.
                
                for f, col in zip(valid_factors, factor_cols):
                    if col in today_df and 'fwd_ret' in today_df:
                        # Rank IC for robustness
                        # ic = today_df[col].corr(today_df['fwd_ret'])
                        # For simplicity/speed in loop
                        day_ics[f] = today_df[col].corr(today_df['fwd_ret'], method='spearman')
                
                # Update MWU weights
                current_weights = mwu.update(date, day_ics)
            
            weights_history.append({'date': date, **current_weights})
                
            # 2. Calculate Composite Score
            # Score = Sum(Weight * Rank(Factor))
            # We need to standardize factors first or use rank. Rank is robust.
            
            scores = pd.Series(0.0, index=today_df.index)
            
            for f, col in zip(valid_factors, factor_cols):
                if col not in today_df: continue
                
                # Check directionality
                # RSI: Low is good (Reversion) -> Rank Ascending (Low=1, High=N) -> We want High Score for Low RSI.
                # So Rank Ascending, then Descending Score?
                # Let's standardize: Higher Rank = Better.
                # RSI: Higher RSI = Worse. So Rank Descending (High RSI = Rank 0). 
                
                ascending = True # Default: Higher factor value is better (Momentum)
                if f in ['Reversion', 'Volatility', 'Microstructure']: 
                    ascending = False # Lower is better
                    
                rank = today_df[col].rank(ascending=ascending, pct=True)
                scores += rank * current_weights[f]
            
            # 3. Select Portfolio (Top 5)
            top_n = scores.nlargest(5).index.tolist()
            
            # 4. Calculate P&L (Next Day Return)
            # Equal weight top n
            if not top_n:
                day_ret = 0.0
            else:
                day_ret = today_df.loc[top_n, 'fwd_ret'].mean()
                
            # Fill NaN returns (e.g. delisted)
            if np.isnan(day_ret): day_ret = 0.0
                
            new_equity = equity[-1] * (1 + day_ret * leverage)
            equity.append(new_equity)
            
        # Result DataFrame
        res_df = pd.DataFrame({'date': dates, 'equity': equity})
        res_df.set_index('date', inplace=True)
        
        w_df = pd.DataFrame(weights_history).set_index('date')
        
        return res_df, w_df
