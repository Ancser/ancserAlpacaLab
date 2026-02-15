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

    def fetch_and_prepare_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data and compute all factors once."""
        print(f"Fetching data for {len(symbols)} symbols...")
        schema_df = self.yahoo.fetch_history(symbols, start_date, end_date).collect()
        
        if schema_df.is_empty():
            print("No data found.")
            return pd.DataFrame() # Empty

        # 2. Compute Factors (Lazy)
        print("Computing factors...")
        factor_df = compute_all_factors(schema_df.lazy()).collect()
        
        # 3. Calculate Forward Returns
        factor_df = factor_df.sort(["symbol", "timestamp"])
        factor_df = factor_df.with_columns([
            (pl.col("close").shift(-1).over("symbol") / pl.col("close") - 1).alias("fwd_ret")
        ])
        
        # Convert to Pandas
        pdf = factor_df.to_pandas()
        pdf['timestamp'] = pd.to_datetime(pdf['timestamp'])
        return pdf

    def run_simulation(self, 
                       data: pd.DataFrame, 
                       active_factors: List[str], 
                       leverage: float = 1.0,
                       use_mwu: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run simulation on precomputed data."""
        if data.empty:
            return pd.DataFrame(), pd.DataFrame()
            
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
        valid_factors = [f for f in active_factors if col_map.get(f) in data.columns]
        factor_cols = [col_map[f] for f in valid_factors]
        
        if not valid_factors:
            return pd.DataFrame(), pd.DataFrame()
        
        dates = sorted(data['timestamp'].unique())
        
        # Initialize MWU
        mwu = MWUEngine(valid_factors)
        current_weights = mwu.weights.copy()
        
        # Simulation State
        equity = [self.initial_capital]
        weights_history = []
        
        # print(f"Running simulation over {len(dates)} days...")
        
        for i, date in enumerate(dates[:-1]):
            # Get data for today
            today_df = data[data['timestamp'] == date].set_index('symbol')
            
            # 1. Update MWU Weights (if enabled and not first day)
            if use_mwu and i > 0:
                day_ics = {}
                for f, col in zip(valid_factors, factor_cols):
                    if col in today_df and 'fwd_ret' in today_df:
                        # Rank IC for robustness
                        # Update weights based on how well factors predicted *this* step.
                        # Using Spearman correlation
                        corr = today_df[col].corr(today_df['fwd_ret'], method='spearman')
                        if np.isnan(corr): corr = 0.0
                        day_ics[f] = corr
                
                # Update MWU weights
                current_weights = mwu.update(date, day_ics)
            
            weights_history.append({'date': date, **current_weights})
                
            # 2. Calculate Composite Score
            scores = pd.Series(0.0, index=today_df.index)
            
            for f, col in zip(valid_factors, factor_cols):
                if col not in today_df: continue
                
                # Directionality
                ascending = True 
                if f in ['Reversion', 'Volatility', 'Microstructure']: 
                    ascending = False 
                    
                rank = today_df[col].rank(ascending=ascending, pct=True)
                scores += rank * current_weights[f]
            
            # 3. Select Portfolio (Top 5)
            top_n = scores.nlargest(5).index.tolist()
            
            # 4. Calculate P&L (Next Day Return)
            if not top_n:
                day_ret = 0.0
            else:
                day_ret = today_df.loc[top_n, 'fwd_ret'].mean()
                
            if np.isnan(day_ret): day_ret = 0.0
                
            new_equity = equity[-1] * (1 + day_ret * leverage)
            equity.append(new_equity)
            
        # Result DataFrame
        res_df = pd.DataFrame({'date': dates, 'equity': equity})
        res_df.set_index('date', inplace=True)
        
        w_df = pd.DataFrame(weights_history).set_index('date')
        
        return res_df, w_df

    def run(self, 
            symbols: List[str], 
            start_date: str, 
            end_date: str, 
            active_factors: List[str], 
            leverage: float = 1.0,
            use_mwu: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        data = self.fetch_and_prepare_data(symbols, start_date, end_date)
        return self.run_simulation(data, active_factors, leverage, use_mwu)
