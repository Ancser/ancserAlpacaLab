import polars as pl
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    def _fetch_chunk(self, adapter, chunk: List[str], start_date: str, end_date: str) -> pl.DataFrame:
        """Fetch a chunk of symbols from the given adapter."""
        try:
            return adapter.fetch_history(chunk, start_date, end_date).collect()
        except Exception as e:
            print(f"⚠ Chunk fetch failed ({len(chunk)} symbols): {e}")
            return pl.DataFrame()

    def fetch_and_prepare_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data and compute all factors once. Uses parallel chunked fetching."""
        print(f"Fetching data for {len(symbols)} symbols using {self.data_source}...")

        # Parallel chunked fetching for large universes
        CHUNK_SIZE = 100
        if len(symbols) > CHUNK_SIZE and self.data_source in ('alpaca', 'mix'):
            chunks = [symbols[i:i + CHUNK_SIZE] for i in range(0, len(symbols), CHUNK_SIZE)]
            print(f"Splitting into {len(chunks)} chunks of ~{CHUNK_SIZE} symbols...")
            frames = []
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._fetch_chunk, self.adapter, chunk, start_date, end_date): chunk
                    for chunk in chunks
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None and not result.is_empty():
                        frames.append(result)
            schema_df = pl.concat(frames) if frames else pl.DataFrame()
        else:
            # Single batch fetch (Yahoo already uses threads=True internally)
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
                            schema_df = pl.concat([schema_df, fallback_df])
                            print(f"✓ Added {len(missing_symbols)} symbols from Yahoo")
                        else:
                            print(f"⚠ Yahoo fallback returned no data for missing symbols")
                    except Exception as e:
                        print(f"⚠ Yahoo fallback failed: {e}")

        if schema_df.is_empty():
            print("No data found from any source.")
            return pd.DataFrame()

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

        # === Pre-compute pivot tables for all factors (vectorized, avoid per-day ops) ===
        # Directionality map
        descending_factors = {'Reversion', 'Volatility', 'Microstructure', 'Drift-Reversion'}

        # Pre-compute rank pivots: {factor_name: DataFrame(date x symbol)}
        rank_pivots = {}
        _temp_cols = []
        for f, col in zip(valid_factors, factor_cols):
            if col not in data.columns:
                continue
            ascending = f not in descending_factors
            rank_col = f'_rank_{col}'
            data[rank_col] = data.groupby('timestamp')[col].rank(ascending=ascending, pct=True)
            pivot = data.pivot_table(index='timestamp', columns='symbol', values=rank_col)
            rank_pivots[f] = pivot
            _temp_cols.append(rank_col)

        # Clean up temp columns
        data.drop(columns=_temp_cols, inplace=True, errors='ignore')

        # Pre-compute fwd_ret pivot
        fwd_ret_pivot = data.pivot_table(index='timestamp', columns='symbol', values='fwd_ret')

        # Pre-compute factor value pivots for IC calc (MWU)
        factor_pivots = {}
        if use_mwu:
            for f, col in zip(valid_factors, factor_cols):
                if col in data.columns:
                    factor_pivots[f] = data.pivot_table(index='timestamp', columns='symbol', values=col)

        # Initialize MWU
        mwu = MWUEngine(valid_factors)
        current_weights = mwu.weights.copy()

        # Simulation State
        equity = [self.initial_capital]
        weights_history = []
        holdings_history = []

        # Volatility Targeting State
        daily_returns_buffer = []
        current_scalar = leverage

        for i, date in enumerate(dates[:-1]):
            # 1. Update MWU Weights (if enabled and not first day)
            if use_mwu and i > 0:
                day_ics = {}
                if date in fwd_ret_pivot.index:
                    fwd_ret_row = fwd_ret_pivot.loc[date].dropna()
                    for f in valid_factors:
                        if f in factor_pivots and date in factor_pivots[f].index:
                            fac_row = factor_pivots[f].loc[date].dropna()
                            common = fac_row.index.intersection(fwd_ret_row.index)
                            if len(common) > 5:
                                corr = fac_row[common].corr(fwd_ret_row[common], method='spearman')
                                day_ics[f] = 0.0 if np.isnan(corr) else corr
                current_weights = mwu.update(date, day_ics)

            # 2. Calculate Composite Score using pre-computed rank pivots
            if date not in fwd_ret_pivot.index:
                daily_returns_buffer.append(0.0)
                equity.append(equity[-1])
                continue

            # Get symbols available on this date
            score_series = None
            for f in valid_factors:
                if f not in rank_pivots or date not in rank_pivots[f].index:
                    continue
                rank_row = rank_pivots[f].loc[date].dropna()
                weighted = rank_row * current_weights[f]
                if score_series is None:
                    score_series = weighted
                else:
                    score_series = score_series.add(weighted, fill_value=0.0)

            if score_series is None or score_series.empty:
                daily_returns_buffer.append(0.0)
                equity.append(equity[-1])
                continue

            scores = score_series

            # 3. Select Portfolio
            if strategy_mode == 'long_short':
                n_side = min(150, max(1, len(scores) // 2))
                top_n    = scores.nlargest(n_side).index.tolist()
                bottom_n = scores.nsmallest(n_side).index.tolist()
            elif strategy_mode == 'top10_long':
                top_n    = scores.nlargest(10).index.tolist()
                bottom_n = []
            elif strategy_mode == 'top10_ls':
                n_side = min(10, max(1, len(scores) // 2))
                top_n    = scores.nlargest(n_side).index.tolist()
                bottom_n = scores.nsmallest(n_side).index.tolist()
            else:
                top_n    = scores.nlargest(5).index.tolist()
                bottom_n = []

            holdings_history.append({
                'date': date,
                'long': ', '.join(top_n),
                'short': ', '.join(bottom_n) if bottom_n else ''
            })

            # 4. Volatility Targeting Logic
            if use_vol_target and len(daily_returns_buffer) >= vol_window:
                recent_rets = np.array(daily_returns_buffer[-vol_window:])
                realized_vol = np.std(recent_rets, ddof=1) * np.sqrt(252)
                if realized_vol > 0.001:
                    calculated_scalar = vol_target_pct / realized_vol
                    current_scalar = min(leverage, calculated_scalar)
                else:
                    current_scalar = leverage
            else:
                current_scalar = leverage

            # 5. Calculate P&L using pre-computed fwd_ret pivot
            fwd_row = fwd_ret_pivot.loc[date]
            if strategy_mode in ('long_short', 'top10_ls'):
                long_rets = fwd_row.reindex(top_n).dropna()
                short_rets = fwd_row.reindex(bottom_n).dropna()
                long_ret = long_rets.mean() if len(long_rets) > 0 else 0.0
                short_ret = short_rets.mean() if len(short_rets) > 0 else 0.0
                raw_day_ret = (long_ret - short_ret) / 2
            else:
                port_rets = fwd_row.reindex(top_n).dropna()
                raw_day_ret = port_rets.mean() if len(port_rets) > 0 else 0.0

            if np.isnan(raw_day_ret): raw_day_ret = 0.0

            actual_day_ret = raw_day_ret * current_scalar
            daily_returns_buffer.append(raw_day_ret)

            new_equity = equity[-1] * (1 + actual_day_ret)
            equity.append(new_equity)

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
