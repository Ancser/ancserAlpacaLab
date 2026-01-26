"""
回测引擎 - 独立的回测系统
可以单独运行,也可以被其他模块调用
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yaml
import math
from typing import Dict, Optional
from datetime import datetime, timedelta
import logging

from data_manager import DataManager
from factor_library import FactorEngine

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    回测引擎
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.dm = DataManager(config_path)
        self.factor_engine = FactorEngine(config_path)
        
        logger.info("BacktestEngine initialized")
    
    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        运行回测
        
        返回: 包含所有结果的字典
        """
        # 1. 准备数据
        logger.info("\n" + "="*60)
        logger.info("开始回测")
        logger.info("="*60)
        
        # 计算日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            years = self.config['backtest']['years']
            start_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
        
        logger.info(f"回测期间: {start_date} ~ {end_date}")
        
        # 2. 获取股票池
        universe = self.dm.get_universe_list()
        logger.info(f"股票池: {len(universe)} 个标的")
        
        # 3. 下载数据
        close_df, volume_df = self.dm.get_market_data(
            universe=universe,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        # 4. 提取基准和股票数据
        benchmarks = self.config['data']['benchmarks']
        defensive = self.config['data']['defensive_assets']
        
        spy = close_df[benchmarks[0]] if benchmarks[0] in close_df.columns else None
        qqq = close_df[benchmarks[1]] if len(benchmarks) > 1 and benchmarks[1] in close_df.columns else None
        cash_asset = close_df[defensive[0]] if defensive[0] in close_df.columns else None
        
        stock_cols = [c for c in close_df.columns if c not in benchmarks + defensive]
        close_stocks = close_df[stock_cols]
        volume_stocks = volume_df[stock_cols]
        
        # 5. 运行核心回测
        stats, benchmarks_stats, equity_curves, port_returns = self._backtest_core(
            close_stk=close_stocks,
            vol_stk=volume_stocks,
            spy_close=spy,
            qqq_close=qqq,
            cash_close=cash_asset
        )
        
        # 6. 打印结果
        self._print_results(stats, benchmarks_stats)
        
        # 7. 绘图
        self._plot_results(equity_curves, stats)
        
        return {
            'stats': stats,
            'benchmarks': benchmarks_stats,
            'equity_curves': equity_curves,
            'returns': port_returns
        }
    
    def _backtest_core(
        self,
        close_stk: pd.DataFrame,
        vol_stk: pd.DataFrame,
        spy_close: pd.Series,
        qqq_close: pd.Series,
        cash_close: Optional[pd.Series]
    ):
        """
        核心回测逻辑
        """
        dates = close_stk.index
        ret_stk = close_stk.pct_change().fillna(0)
        ret_spy = spy_close.pct_change().fillna(0) if spy_close is not None else pd.Series(0, index=dates)
        ret_qqq = qqq_close.pct_change().fillna(0) if qqq_close is not None else pd.Series(0, index=dates)
        ret_cash = cash_close.pct_change().fillna(0) if cash_close is not None else pd.Series(0, index=dates)
        
        # 过滤器配置
        filter_cfg = self.config['filters']
        portfolio_cfg = self.config['portfolio']
        regime_cfg = self.config['regime']
        costs_cfg = self.config['costs']
        
        # 时变过滤器
        dvol = (close_stk * vol_stk).rolling(filter_cfg['adv_window'], min_periods=10).mean()
        eligible = (close_stk >= filter_cfg['min_price']) & (dvol >= filter_cfg['min_adv_dollar'])
        
        # 计算因子
        logger.info("计算因子...")
        factors = self.factor_engine.compute_all_factors(close_stk, vol_stk)
        
        # 复合得分
        score = self.factor_engine.compute_composite_score(factors)
        
        # 趋势过滤
        if regime_cfg['enabled']:
            indicator = spy_close if spy_close is not None else close_stk.mean(axis=1)
            sma = indicator.rolling(regime_cfg['sma_length'], min_periods=regime_cfg['sma_length']//2).mean()
            mom = indicator.pct_change(regime_cfg['momentum_length'])
            risk_on = (indicator > sma) & (mom > 0)
        else:
            risk_on = pd.Series(True, index=dates)
        
        # 调仓日期
        rebal_dates = self._get_rebalance_dates(dates, portfolio_cfg['rebalance'])
        
        # 状态变量
        w_prev = pd.Series(0.0, index=close_stk.columns)
        held_days = {ticker: 0 for ticker in close_stk.columns}
        
        # 记录
        port_ret = pd.Series(0.0, index=dates)
        turnover = pd.Series(0.0, index=dates)
        n_positions = pd.Series(0, index=dates)
        defensive_count = 0
        
        logger.info("执行回测...")
        
        for i in range(1, len(dates)):
            dt = dates[i]
            sig_dt = dates[i-1]
            
            # 默认保持上期权重
            w_tgt = w_prev.copy()
            
            # 是否调仓
            do_rebal = dt in rebal_dates
            
            if do_rebal:
                # 更新持有天数
                for ticker in held_days:
                    if w_prev.get(ticker, 0) > 0:
                        held_days[ticker] += 1
                    else:
                        held_days[ticker] = 0
                
                # 风险判断
                if not risk_on.loc[sig_dt] and regime_cfg.get('risk_off_mode') == 'defensive':
                    # 防御模式
                    w_tgt[:] = 0.0
                    defensive_count += 1
                
                else:
                    # 主动模式
                    mask = eligible.loc[sig_dt]
                    scores = score.loc[sig_dt].where(mask).dropna()
                    
                    if len(scores) < portfolio_cfg['min_names_to_trade']:
                        w_tgt[:] = 0.0
                        defensive_count += 1
                    else:
                        # 排名
                        ranks = scores.rank(ascending=False, method='first')
                        
                        # 保留逻辑: buffer + 最短持有期
                        keep = []
                        for ticker in w_prev.index[w_prev > 0]:
                            rank = ranks.get(ticker, np.inf)
                            days_held = held_days.get(ticker, 0)
                            
                            if days_held < portfolio_cfg['min_hold_days']:
                                keep.append(ticker)
                            elif rank <= portfolio_cfg['top_n'] + portfolio_cfg['rank_buffer']:
                                keep.append(ticker)
                        
                        # 补充新股票到top_n
                        candidates = list(scores.sort_values(ascending=False).index)
                        new_picks = []
                        for ticker in candidates:
                            if ticker in keep:
                                continue
                            new_picks.append(ticker)
                            if len(keep) + len(new_picks) >= portfolio_cfg['top_n']:
                                break
                        
                        final_picks = keep + new_picks
                        final_picks = final_picks[:portfolio_cfg['top_n']]
                        
                        if len(final_picks) > 0:
                            # 等权重
                            base_weight = 1.0 / len(final_picks)
                            w_tgt[:] = 0.0
                            for ticker in final_picks:
                                w_tgt.loc[ticker] = min(base_weight, portfolio_cfg['max_weight'])
                            
                            # 归一化
                            total = w_tgt.sum()
                            if total > 0:
                                w_tgt = w_tgt / total
            
            # 计算换手率
            turn = 0.5 * float(np.abs(w_tgt - w_prev).sum())
            cost = turn * (costs_cfg['total_bps'] / 10000.0)
            
            # 计算收益
            stock_ret = float((w_tgt * ret_stk.loc[dt]).sum())
            
            # 如果没有股票仓位,获取现金收益
            if float(w_tgt.sum()) < 1e-12:
                stock_ret = float(ret_cash.loc[dt])
            
            # 净收益 = 总收益 - 成本
            net_ret = stock_ret - cost
            
            # 记录
            port_ret.loc[dt] = net_ret
            turnover.loc[dt] = turn
            n_positions.loc[dt] = int((w_tgt > 0).sum())
            
            # 更新状态
            w_prev = w_tgt
        
        # 计算权益曲线
        eq_strategy = (1 + port_ret).cumprod()
        eq_spy = (1 + ret_spy).cumprod()
        eq_qqq = (1 + ret_qqq).cumprod()
        
        # 统计指标
        stats = self._calculate_stats(eq_strategy, port_ret, turnover, n_positions, 
                                      defensive_count, len(rebal_dates))
        
        bench_stats = {
            'SPY': self._calculate_benchmark_stats(eq_spy, ret_spy),
            'QQQ': self._calculate_benchmark_stats(eq_qqq, ret_qqq)
        }
        
        equity_curves = pd.DataFrame({
            'Strategy': eq_strategy,
            'SPY': eq_spy,
            'QQQ': eq_qqq
        })
        
        return stats, bench_stats, equity_curves, port_ret
    
    def _get_rebalance_dates(self, index: pd.DatetimeIndex, rule: str) -> set:
        """生成调仓日期"""
        if rule.upper() == 'D':
            return set(index)
        
        # 使用resample
        idx = pd.DatetimeIndex(index).sort_values()
        s = pd.Series(1.0, index=idx)
        rebal = s.resample(rule).last().dropna().index
        
        return set(rebal)
    
    def _calculate_stats(self, equity, returns, turnover, n_pos, def_count, rebal_count):
        """计算策略统计"""
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        
        final_value = float(equity.iloc[-1])
        cagr = float(final_value ** (1/years) - 1)
        
        # 最大回撤
        peak = equity.cummax()
        dd = equity / peak - 1
        max_dd = float(dd.min())
        
        # 夏普比率
        sharpe = float((returns.mean() / returns.std()) * math.sqrt(252)) if returns.std() > 0 else np.nan
        
        # 胜率
        win_rate = float((returns > 0).mean())
        
        return {
            'Final': final_value,
            'CAGR': cagr,
            'MaxDD': max_dd,
            'Sharpe': sharpe,
            'WinRate': win_rate,
            'AvgTurnover': float(turnover.mean()),
            'AvgPositions': float(n_pos.mean()),
            'DefensivePct': float(def_count) / max(1, rebal_count),
            'Years': years
        }
    
    def _calculate_benchmark_stats(self, equity, returns):
        """计算基准统计"""
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        final = float(equity.iloc[-1])
        cagr = float(final ** (1/years) - 1)
        
        peak = equity.cummax()
        dd = equity / peak - 1
        max_dd = float(dd.min())
        
        sharpe = float((returns.mean() / returns.std()) * math.sqrt(252)) if returns.std() > 0 else np.nan
        
        return {
            'Final': final,
            'CAGR': cagr,
            'MaxDD': max_dd,
            'Sharpe': sharpe
        }
    
    def _print_results(self, stats, bench_stats):
        """打印结果"""
        print("\n" + "="*60)
        print("回测结果")
        print("="*60)
        
        for key, value in stats.items():
            if isinstance(value, float):
                if 'pct' in key.lower() or key in ['CAGR', 'MaxDD', 'WinRate', 'DefensivePct']:
                    print(f"{key:>18}: {value:.2%}")
                else:
                    print(f"{key:>18}: {value:.4f}")
            else:
                print(f"{key:>18}: {value}")
        
        print("\n" + "="*60)
        print("基准对比")
        print("="*60)
        
        for bench_name, bench_data in bench_stats.items():
            print(f"\n{bench_name}:")
            print(f"  Final: {bench_data['Final']:.4f}")
            print(f"  CAGR: {bench_data['CAGR']:.2%}")
            print(f"  MaxDD: {bench_data['MaxDD']:.2%}")
            print(f"  Sharpe: {bench_data['Sharpe']:.4f}")
    
    def _plot_results(self, equity_curves, stats):
        """绘制结果"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(14, 8))
            
            # 权益曲线
            plt.subplot(2, 1, 1)
            for col in equity_curves.columns:
                plt.plot(equity_curves.index, equity_curves[col], label=col, linewidth=2)
            
            plt.title(f"权益曲线 | CAGR={stats['CAGR']:.2%} | MaxDD={stats['MaxDD']:.2%} | Sharpe={stats['Sharpe']:.2f}")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylabel('权益倍数')
            
            # 回撤
            plt.subplot(2, 1, 2)
            strategy_eq = equity_curves['Strategy']
            peak = strategy_eq.cummax()
            dd = strategy_eq / peak - 1
            
            plt.fill_between(dd.index, dd, 0, alpha=0.3, color='red')
            plt.plot(dd.index, dd, color='darkred', linewidth=1)
            plt.title('回撤曲线')
            plt.ylabel('回撤 %')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.warning(f"绘图失败: {e}")


# ==========================================
# 独立测试
# ==========================================

def test_backtest():
    """
    测试回测引擎 - Ctrl+F5 快速测试
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    engine = BacktestEngine()
    
    # 运行回测 (最近2年)
    results = engine.run(
        start_date="2023-01-01",
        end_date=None
    )
    
    print("\n✅ 回测完成!")
    
    return results


if __name__ == "__main__":
    test_backtest()