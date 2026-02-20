import polars as pl
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
from ancser_quant.data.yahoo_adapter import YahooAdapter
from ancser_quant.alpha.factors import compute_all_factors
from ancser_quant.alpha.mwu import MWUEngine

class BacktestEngine:
    """
    Polars-based Backtest Engine.
    Simulates strategy performance over historical data.
    Supports MWU (Dynamic Weighting).
    """
    def __init__(self, initial_capital: float = 100000.0, data_source: str = 'yahoo'):
        self.initial_capital = initial_capital
        self.data_source = data_source

        if data_source == 'alpaca':
            from ancser_quant.data.alpaca_adapter import AlpacaAdapter
            self.adapter = AlpacaAdapter()
            self.fallback_adapter = None
            print("Using Alpaca Data Source")
        elif data_source == 'mix':
            from ancser_quant.data.alpaca_adapter import AlpacaAdapter
            self.adapter = AlpacaAdapter()
            self.fallback_adapter = YahooAdapter()
            print("Using Mix Data Source (Alpaca + Yahoo fallback)")
        else:
            self.adapter = YahooAdapter()
            self.fallback_adapter = None
            print("Using Yahoo Finance Data Source")

    def fetch_and_prepare_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data and compute all factors once. Uses fallback if in mix mode."""
        print(f"Fetching data for {len(symbols)} symbols using {self.data_source}...")

        # Fetch from primary adapter
        schema_df = self.adapter.fetch_history(symbols, start_date, end_date).collect()

        # If in mix mode and primary data is empty or incomplete, try fallback
        if self.fallback_adapter is not None:
            if schema_df.is_empty():
                print("Primary source returned no data, trying fallback (Yahoo)...")
                try:
                    schema_df = self.fallback_adapter.fetch_history(symbols, start_date, end_date).collect()
                    if not schema_df.is_empty():
                        print(f"✓ Yahoo fallback provided data for {len(symbols)} symbols")
                except Exception as e:
                    print(f"⚠ Yahoo fallback failed: {e}")
            else:
                # Check for missing symbols
                fetched_symbols = schema_df['symbol'].unique().to_list()
                missing_symbols = [s for s in symbols if s not in fetched_symbols]

                if missing_symbols:
                    print(f"Primary source missing {len(missing_symbols)} symbols: {missing_symbols[:5]}...")
                    try:
                        fallback_df = self.fallback_adapter.fetch_history(missing_symbols, start_date, end_date).collect()

                        if not fallback_df.is_empty():
                            # Combine data
                            schema_df = pl.concat([schema_df, fallback_df])
                            print(f"✓ Added {len(missing_symbols)} symbols from Yahoo")
                        else:
                            print(f"⚠ Yahoo fallback returned no data for missing symbols")
                    except Exception as e:
                        print(f"⚠ Yahoo fallback failed: {e}")

        if schema_df.is_empty():
            print("No data found from any source.")
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
                       use_mwu: bool = False,
                       use_vol_target: bool = True,
                       vol_target_pct: float = 0.20,
                       vol_window: int = 20,
                       strategy_mode: str = 'long_only') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run simulation on precomputed data.
        strategy_mode: 'long_only' (top 5) | 'long_short' (top 150 long, bottom 150 short, market-neutral)
        """
        if data.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            
        # Map friendly names to internal columns
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
        
        # Filter valid factors
        valid_factors = [f for f in active_factors if col_map.get(f) in data.columns]
        factor_cols = [col_map[f] for f in valid_factors]
        
        if not valid_factors:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        dates = sorted(data['timestamp'].unique())
        
        # Initialize MWU
        mwu = MWUEngine(valid_factors)
        current_weights = mwu.weights.copy()
        
        # Simulation State
        equity = [self.initial_capital]
        weights_history = []
        holdings_history = []
        
        # Volatility Targeting State
        daily_returns_buffer = [] # Store portfolio daily returns for vol calculation
        current_scalar = leverage # Default to max leverage if no history
        
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
            
            # 2. Calculate Composite Score
            scores = pd.Series(0.0, index=today_df.index)
            
            for f, col in zip(valid_factors, factor_cols):
                if col not in today_df: continue
                
                # Directionality
                ascending = True 
                if f in ['Reversion', 'Volatility', 'Microstructure', 'Drift-Reversion']: 
                    ascending = False 
                    
                rank = today_df[col].rank(ascending=ascending, pct=True)
                scores += rank * current_weights[f]
            
            # 3. Select Portfolio
            if strategy_mode == 'long_short':
                # Market-neutral: top 150 long, bottom 150 short
                n_side = min(150, max(1, len(scores) // 2))
                top_n    = scores.nlargest(n_side).index.tolist()
                bottom_n = scores.nsmallest(n_side).index.tolist()
            elif strategy_mode == 'top10_long':
                # Long-only top 10
                top_n    = scores.nlargest(10).index.tolist()
                bottom_n = []
            elif strategy_mode == 'top10_ls':
                # Long top 10, short bottom 10
                n_side = min(10, max(1, len(scores) // 2))
                top_n    = scores.nlargest(n_side).index.tolist()
                bottom_n = scores.nsmallest(n_side).index.tolist()
            else:
                # Long-only top 5 (default)
                top_n    = scores.nlargest(5).index.tolist()
                bottom_n = []

            # Record holdings for this date
            holdings_history.append({
                'date': date,
                'long': ', '.join(top_n),
                'short': ', '.join(bottom_n) if bottom_n else ''
            })

            # 4. Volatility Targeting Logic
            # We calculate vol based on *past* returns to set exposure for *today* (which affects tomorrow's P&L)
            if use_vol_target and len(daily_returns_buffer) >= vol_window:
                # Calculate Realized Volatility (Annualized)
                recent_rets = np.array(daily_returns_buffer[-vol_window:])
                # std of returns * sqrt(252)
                realized_vol = np.std(recent_rets, ddof=1) * np.sqrt(252)

                if realized_vol > 0.001: # Avoid division by zero
                    # Scalar = Target / Realized
                    calculated_scalar = vol_target_pct / realized_vol
                    # Cap at Max Leverage (Constraint)
                    current_scalar = min(leverage, calculated_scalar)
                else:
                    current_scalar = leverage # Vol is super low, max out
            else:
                current_scalar = leverage # Default or Warmup

            # 5. Calculate P&L (Next Day Return)
            if strategy_mode in ('long_short', 'top10_ls'):
                long_ret  = today_df.loc[top_n,    'fwd_ret'].mean() if top_n    else 0.0
                short_ret = today_df.loc[bottom_n, 'fwd_ret'].mean() if bottom_n else 0.0
                raw_day_ret = (long_ret - short_ret) / 2  # market-neutral, 50/50 gross
            else:
                raw_day_ret = today_df.loc[top_n, 'fwd_ret'].mean() if top_n else 0.0

            if np.isnan(raw_day_ret): raw_day_ret = 0.0
            
            # Apply Scalar (Leverage/De-leverage)
            actual_day_ret = raw_day_ret * current_scalar
            
            # Store return for next vol calc
            daily_returns_buffer.append(raw_day_ret) # Use raw strategy return (basket return) for vol calc 
            # Note: Standard practice uses the underlying asset vol. 
            # Here we use the raw strategy basket vol (unlevered) to determine leverage.
                
            new_equity = equity[-1] * (1 + actual_day_ret)
            equity.append(new_equity)
            
            # Store history (include scalar for debug)
            hist = {'date': date, **current_weights}
            hist['vol_scalar'] = current_scalar
            hist['realized_vol'] = (np.std(daily_returns_buffer[-vol_window:], ddof=1) * np.sqrt(252)) if len(daily_returns_buffer) >= vol_window else 0.0
            weights_history.append(hist)
            
        # Result DataFrame
        res_df = pd.DataFrame({'date': dates, 'equity': equity})
        res_df.set_index('date', inplace=True)

        w_df = pd.DataFrame(weights_history).set_index('date')
        holdings_df = pd.DataFrame(holdings_history).set_index('date') if holdings_history else pd.DataFrame()

        return res_df, w_df, holdings_df

    def run(self,
            symbols: List[str],
            start_date: str,
            end_date: str,
            active_factors: List[str],
            leverage: float = 1.0,
            use_mwu: bool = False,
            use_vol_target: bool = True,
            vol_target_pct: float = 0.20,
            vol_window: int = 20,
            strategy_mode: str = 'long_only') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        data = self.fetch_and_prepare_data(symbols, start_date, end_date)
        return self.run_simulation(data, active_factors, leverage, use_mwu, use_vol_target, vol_target_pct, vol_window, strategy_mode)
