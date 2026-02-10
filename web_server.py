"""
Factor Analysis Web Dashboard
Flask server wrapping existing quant modules
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import math
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from copy import deepcopy

import pandas as pd
import numpy as np
import yaml
from flask import Flask, request, jsonify, send_from_directory

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_manager import DataManager
from factor_library import (
    FactorEngine, FactorAnalyzer,
    momentum_12_1, pullback_5d, rsi_factor,
    volatility_factor, volume_surge, trend_strength,
    zscore_cross_sectional
)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# Flask App
# ==========================================

app = Flask(__name__, static_folder='static', static_url_path='/static')

SAVED_DIR = Path(__file__).parent / 'saved_data'
SAVED_DIR.mkdir(exist_ok=True)
FACTORS_FILE = SAVED_DIR / 'factors.json'
STRATEGIES_FILE = SAVED_DIR / 'strategies.json'

# Factor metadata for the UI
FACTOR_INFO = {
    'momentum_12_1': {
        'name': 'Momentum 12-1',
        'description': '12-month return minus most recent 1-month return',
        'params': {
            'lookback': {'default': 252, 'min': 20, 'max': 504, 'step': 1, 'label': 'Lookback (days)'},
            'skip': {'default': 21, 'min': 0, 'max': 63, 'step': 1, 'label': 'Skip Recent (days)'}
        },
        'default_weight': 0.70
    },
    'pullback_5d': {
        'name': 'Pullback 5D',
        'description': 'Short-term reversal factor (5-day pullback)',
        'params': {
            'lookback': {'default': 5, 'min': 1, 'max': 30, 'step': 1, 'label': 'Lookback (days)'}
        },
        'default_weight': 0.30
    },
    'rsi': {
        'name': 'RSI',
        'description': 'Relative Strength Index (0-100, >70 overbought, <30 oversold)',
        'params': {
            'period': {'default': 14, 'min': 5, 'max': 50, 'step': 1, 'label': 'Period (days)'}
        },
        'default_weight': 0.0
    },
    'volatility': {
        'name': 'Volatility',
        'description': 'Annualized volatility — lower is more stable',
        'params': {
            'period': {'default': 20, 'min': 5, 'max': 60, 'step': 1, 'label': 'Period (days)'}
        },
        'default_weight': 0.0
    },
    'volume_surge': {
        'name': 'Volume Surge',
        'description': 'Current volume / average volume ratio',
        'params': {
            'period': {'default': 20, 'min': 5, 'max': 60, 'step': 1, 'label': 'Period (days)'}
        },
        'default_weight': 0.0
    },
    'trend_strength': {
        'name': 'Trend Strength',
        'description': 'Short SMA / Long SMA - 1 (dual moving average)',
        'params': {
            'short': {'default': 20, 'min': 5, 'max': 100, 'step': 1, 'label': 'Short SMA'},
            'long': {'default': 50, 'min': 20, 'max': 252, 'step': 1, 'label': 'Long SMA'}
        },
        'default_weight': 0.0
    }
}

# Factor function mapping
FACTOR_FUNCTIONS = {
    'momentum_12_1': momentum_12_1,
    'pullback_5d': pullback_5d,
    'rsi': rsi_factor,
    'volatility': volatility_factor,
    'volume_surge': volume_surge,
    'trend_strength': trend_strength,
}

# Rebalance frequency mapping
REBALANCE_MAP = {
    'daily': 'D',
    'weekly_fri': 'W-FRI',
    'weekly_mon': 'W-MON',
    '2week': '2W-FRI',
    'monthly': 'ME',
    'quarterly': 'QS',
    'half_year': '2QS',
    'yearly': 'YS',
}


# ==========================================
# Helpers
# ==========================================

def _load_json(filepath):
    if filepath.exists():
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def _save_json(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _equity_to_json(series):
    """Convert a pandas Series to {date_string: value} dict."""
    result = {}
    for dt, val in series.items():
        if pd.isna(val):
            continue
        date_str = dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else str(dt)
        result[date_str] = round(float(val), 6)
    return result


def _compute_stats(equity_series, returns_series):
    """Compute performance stats from equity curve and returns."""
    if len(equity_series) < 2:
        return {}

    days = (equity_series.index[-1] - equity_series.index[0]).days
    years = max(days / 365.25, 0.01)
    final = float(equity_series.iloc[-1])
    cagr = float(final ** (1 / years) - 1) if final > 0 else 0

    peak = equity_series.cummax()
    dd = equity_series / peak - 1
    max_dd = float(dd.min())

    std = returns_series.std()
    sharpe = float((returns_series.mean() / std) * math.sqrt(252)) if std > 0 else 0
    win_rate = float((returns_series > 0).mean())

    return {
        'cagr': round(cagr, 4),
        'max_dd': round(max_dd, 4),
        'sharpe': round(sharpe, 4),
        'win_rate': round(win_rate, 4),
        'final_value': round(final, 4),
        'years': round(years, 2)
    }


def _run_backtest_with_config(factor_config, rebalance_rule, years, top_n,
                              tickers=None, universe_mode='sp500_nasdaq100',
                              start_date=None, end_date=None):
    """
    Run a backtest using the provided factor configuration.
    Returns equity curves + stats.
    """
    # Load base config
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Override settings
    config['portfolio']['top_n'] = top_n
    config['portfolio']['rebalance'] = rebalance_rule
    config['backtest']['years'] = years

    # Date range
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')

    # Get data — override DataManager's config AFTER init
    dm = DataManager(str(config_path))

    if tickers:
        universe = tickers
    else:
        # Override universe mode in the DataManager's own config
        dm.config['universe']['mode'] = universe_mode
        universe = dm.get_universe_list()

    logger.info(f"Universe: {len(universe)} symbols, mode: {universe_mode}, period: {start_date} to {end_date}")

    close_df, volume_df = dm.get_market_data(
        universe=universe,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )

    if close_df.empty:
        raise ValueError("No market data available")

    benchmarks_list = config['data']['benchmarks']
    defensive_list = config['data'].get('defensive_assets', [])

    spy = close_df[benchmarks_list[0]] if benchmarks_list[0] in close_df.columns else None
    qqq = close_df[benchmarks_list[1]] if len(benchmarks_list) > 1 and benchmarks_list[1] in close_df.columns else None

    # Separate stock columns
    exclude_cols = set(benchmarks_list + defensive_list)
    stock_cols = [c for c in close_df.columns if c not in exclude_cols]
    close_stocks = close_df[stock_cols] if stock_cols else close_df
    volume_stocks = volume_df[[c for c in stock_cols if c in volume_df.columns]] if stock_cols else volume_df

    if close_stocks.empty:
        raise ValueError("No stock data after filtering")

    # Compute factors based on user config
    factors = {}
    total_weight = 0

    for factor_name, fcfg in factor_config.items():
        if not fcfg.get('enabled', False):
            continue
        weight = fcfg.get('weight', 0)
        if weight <= 0:
            continue
        total_weight += weight

        func = FACTOR_FUNCTIONS.get(factor_name)
        if func is None:
            continue

        params = {k: v for k, v in fcfg.items() if k not in ('enabled', 'weight')}
        
        try:
            if factor_name == 'volume_surge':
                factor_df = func(volume_stocks, **params)
            else:
                factor_df = func(close_stocks, **params)
            factors[factor_name] = (factor_df, weight)
        except Exception as e:
            logger.warning(f"Factor {factor_name} computation failed: {e}")

    if not factors:
        raise ValueError("No valid factors computed")

    # Compute composite score
    sample_df = list(factors.values())[0][0]
    score = pd.DataFrame(0.0, index=sample_df.index, columns=sample_df.columns)

    for fname, (fdf, w) in factors.items():
        norm = zscore_cross_sectional(fdf)
        norm = norm.reindex(index=score.index, columns=score.columns).fillna(0)
        score += (w / total_weight) * norm

    # Filters
    filter_cfg = config['filters']
    dvol = (close_stocks * volume_stocks.reindex(columns=close_stocks.columns, fill_value=0)).rolling(
        filter_cfg['adv_window'], min_periods=10).mean()
    eligible = (close_stocks >= filter_cfg['min_price']) & (dvol >= filter_cfg['min_adv_dollar'])

    # Regime filter
    regime_cfg = config['regime']
    dates = close_stocks.index
    ret_stk = close_stocks.pct_change().fillna(0)

    if regime_cfg['enabled'] and spy is not None:
        indicator = spy
        sma = indicator.rolling(regime_cfg['sma_length'], min_periods=regime_cfg['sma_length'] // 2).mean()
        mom = indicator.pct_change(regime_cfg['momentum_length'])
        risk_on = (indicator > sma) & (mom > 0)
    else:
        risk_on = pd.Series(True, index=dates)

    # Rebalance dates
    idx = pd.DatetimeIndex(dates).sort_values()
    s = pd.Series(1.0, index=idx)
    if rebalance_rule.upper() == 'D':
        rebal_dates = set(idx)
    else:
        rebal = s.resample(rebalance_rule).last().dropna().index
        rebal_dates = set(rebal)

    # Cash return
    cash_asset_name = defensive_list[0] if defensive_list and defensive_list[0] in close_df.columns else None
    ret_cash = close_df[cash_asset_name].pct_change().fillna(0) if cash_asset_name else pd.Series(0, index=dates)

    # Core backtest loop
    portfolio_cfg = config['portfolio']
    costs_cfg = config['costs']
    w_prev = pd.Series(0.0, index=close_stocks.columns)
    port_ret = pd.Series(0.0, index=dates)

    for i in range(1, len(dates)):
        dt = dates[i]
        sig_dt = dates[i - 1]
        w_tgt = w_prev.copy()

        if dt in rebal_dates:
            if not risk_on.get(sig_dt, True) and regime_cfg.get('risk_off_mode') == 'defensive':
                w_tgt[:] = 0.0
            else:
                mask = eligible.loc[sig_dt] if sig_dt in eligible.index else pd.Series(True, index=close_stocks.columns)
                scores = score.loc[sig_dt].where(mask).dropna() if sig_dt in score.index else pd.Series(dtype=float)

                if len(scores) >= portfolio_cfg.get('min_names_to_trade', 1):
                    ranks = scores.rank(ascending=False, method='first')
                    candidates = list(scores.sort_values(ascending=False).index)
                    picks = candidates[:top_n]

                    if picks:
                        base_w = 1.0 / len(picks)
                        w_tgt[:] = 0.0
                        for ticker in picks:
                            w_tgt.loc[ticker] = min(base_w, portfolio_cfg.get('max_weight', 0.4))
                        total = w_tgt.sum()
                        if total > 0:
                            w_tgt = w_tgt / total
                else:
                    w_tgt[:] = 0.0

        turn = 0.5 * float(np.abs(w_tgt - w_prev).sum())
        cost = turn * (costs_cfg['total_bps'] / 10000.0)

        stock_ret = float((w_tgt * ret_stk.loc[dt]).sum())
        if float(w_tgt.sum()) < 1e-12:
            stock_ret = float(ret_cash.get(dt, 0))

        port_ret.loc[dt] = stock_ret - cost
        w_prev = w_tgt

    # Build equity curves
    eq_strategy = (1 + port_ret).cumprod()
    ret_spy = spy.pct_change().fillna(0) if spy is not None else pd.Series(0, index=dates)
    ret_qqq = qqq.pct_change().fillna(0) if qqq is not None else pd.Series(0, index=dates)
    eq_spy = (1 + ret_spy).cumprod()
    eq_qqq = (1 + ret_qqq).cumprod()

    return {
        'strategy': _equity_to_json(eq_strategy),
        'spy': _equity_to_json(eq_spy),
        'qqq': _equity_to_json(eq_qqq),
        'stats': {
            'strategy': _compute_stats(eq_strategy, port_ret),
            'spy': _compute_stats(eq_spy, ret_spy),
            'qqq': _compute_stats(eq_qqq, ret_qqq),
        }
    }


# ==========================================
# Routes
# ==========================================

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/factors', methods=['GET'])
def get_factors():
    return jsonify(FACTOR_INFO)


@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    try:
        data = request.get_json()
        factor_config = data.get('factors', {})
        rebalance_key = data.get('rebalance', 'weekly_fri')
        rebalance_rule = REBALANCE_MAP.get(rebalance_key, 'W-FRI')
        years = data.get('years', 5)
        top_n = data.get('top_n', 10)
        tickers = data.get('tickers', None)
        universe_mode = data.get('universe_mode', 'sp500_nasdaq100')

        if tickers and isinstance(tickers, str):
            tickers = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        if not tickers:
            tickers = None

        logger.info(f"Backtest request: factors={list(factor_config.keys())}, "
                    f"rebalance={rebalance_rule}, years={years}, top_n={top_n}, "
                    f"universe={universe_mode}, tickers={tickers}")

        result = _run_backtest_with_config(
            factor_config, rebalance_rule, years, top_n,
            tickers=tickers, universe_mode=universe_mode
        )
        return jsonify({'status': 'ok', 'data': result})

    except Exception as e:
        logger.exception("Backtest failed")
        return jsonify({'status': 'error', 'message': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/rolling', methods=['POST'])
def run_rolling():
    """Run rolling 1-year window backtests and return overlaid curves."""
    try:
        data = request.get_json()
        factor_config = data.get('factors', {})
        rebalance_key = data.get('rebalance', 'weekly_fri')
        rebalance_rule = REBALANCE_MAP.get(rebalance_key, 'W-FRI')
        total_years = data.get('years', 5)
        window_years = data.get('window_years', 1)
        top_n = data.get('top_n', 10)
        tickers = data.get('tickers', None)
        universe_mode = data.get('universe_mode', 'sp500_nasdaq100')

        if tickers and isinstance(tickers, str):
            tickers = [t.strip().upper() for t in tickers.split(',') if t.strip()]
        if not tickers:
            tickers = None

        windows = []
        now = datetime.now()

        logger.info(f"Rolling request: total_years={total_years}, window={window_years}yr, "
                    f"generating {total_years - window_years + 1} windows")

        for start_year_offset in range(total_years - window_years, -1, -1):
            w_end_dt = now - timedelta(days=start_year_offset * 365)
            w_start_dt = w_end_dt - timedelta(days=window_years * 365)
            w_start = w_start_dt.strftime('%Y-%m-%d')
            w_end = w_end_dt.strftime('%Y-%m-%d')

            try:
                logger.info(f"  Rolling window: {w_start} to {w_end}")
                result = _run_backtest_with_config(
                    factor_config, rebalance_rule, window_years, top_n,
                    tickers=tickers, universe_mode=universe_mode,
                    start_date=w_start, end_date=w_end
                )
                # Normalize to start at 1.0
                strat_data = result['strategy']
                if strat_data:
                    first_val = list(strat_data.values())[0]
                    if first_val > 0:
                        normalized = {k: round(v / first_val, 6) for k, v in strat_data.items()}
                    else:
                        normalized = strat_data
                else:
                    normalized = strat_data

                windows.append({
                    'label': f"{w_start[:4]}-{w_end[:4]}",
                    'start': w_start,
                    'end': w_end,
                    'equity': normalized,
                    'stats': result['stats']['strategy']
                })
            except Exception as e:
                logger.warning(f"Rolling window {w_start}-{w_end} failed: {e}")
                continue

        return jsonify({'status': 'ok', 'windows': windows})

    except Exception as e:
        logger.exception("Rolling backtest failed")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/save-factor', methods=['POST'])
def save_factor():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        config = data.get('config', {})

        if not name:
            return jsonify({'status': 'error', 'message': 'Name is required'}), 400

        saved = _load_json(FACTORS_FILE)
        saved[name] = {
            'config': config,
            'saved_at': datetime.now().isoformat()
        }
        _save_json(FACTORS_FILE, saved)

        return jsonify({'status': 'ok', 'message': f'Factor config "{name}" saved'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/save-strategy', methods=['POST'])
def save_strategy():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        strategy = data.get('strategy', {})

        if not name:
            return jsonify({'status': 'error', 'message': 'Name is required'}), 400

        saved = _load_json(STRATEGIES_FILE)
        saved[name] = {
            'strategy': strategy,
            'saved_at': datetime.now().isoformat()
        }
        _save_json(STRATEGIES_FILE, saved)

        return jsonify({'status': 'ok', 'message': f'Strategy "{name}" saved'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/saved', methods=['GET'])
def get_saved():
    factors = _load_json(FACTORS_FILE)
    strategies = _load_json(STRATEGIES_FILE)
    return jsonify({'factors': factors, 'strategies': strategies})


@app.route('/api/delete-saved', methods=['POST'])
def delete_saved():
    try:
        data = request.get_json()
        name = data.get('name', '').strip()
        save_type = data.get('type', 'factor')

        if save_type == 'factor':
            saved = _load_json(FACTORS_FILE)
            saved.pop(name, None)
            _save_json(FACTORS_FILE, saved)
        else:
            saved = _load_json(STRATEGIES_FILE)
            saved.pop(name, None)
            _save_json(STRATEGIES_FILE, saved)

        return jsonify({'status': 'ok', 'message': f'Deleted "{name}"'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ==========================================
# Main
# ==========================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Factor Analysis Web Dashboard")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=True)
