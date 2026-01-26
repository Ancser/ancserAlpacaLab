"""
因子库 - 可扩展的因子计算与测试框架
每个因子都是独立函数，方便单独测试和组合
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ==========================================
# 因子计算函数
# ==========================================

def momentum_12_1(close: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """
    12-1动量因子
    12个月回报减去最近1个月回报
    
    参数:
        lookback: 总回看期 (默认252天 ≈ 12个月)
        skip: 跳过最近N天 (默认21天 ≈ 1个月)
    """
    total_ret = close.pct_change(lookback)
    recent_ret = close.pct_change(skip)
    return total_ret - recent_ret


def pullback_5d(close: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    短期反转因子 (5日回撤)
    值越高表示回撤越大，可能反弹
    
    参数:
        lookback: 回看期 (默认5天)
    """
    return -close.pct_change(lookback)


def rsi_factor(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    RSI (相对强弱指标)
    范围 0-100, >70超买, <30超卖
    """
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def volatility_factor(close: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    波动率因子
    低波动通常更稳定
    """
    returns = close.pct_change()
    return returns.rolling(period).std() * np.sqrt(252)


def volume_surge(volume: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    成交量激增因子
    当前成交量 / 平均成交量
    """
    avg_vol = volume.rolling(period).mean()
    return volume / avg_vol


def trend_strength(close: pd.DataFrame, short: int = 20, long: int = 50) -> pd.DataFrame:
    """
    趋势强度 (双均线)
    短期均线 / 长期均线 - 1
    """
    sma_short = close.rolling(short).mean()
    sma_long = close.rolling(long).mean()
    return (sma_short / sma_long) - 1


def beta_factor(close: pd.DataFrame, benchmark: pd.Series, period: int = 60) -> pd.DataFrame:
    """
    Beta因子 (相对基准的系统性风险)
    
    参数:
        benchmark: 基准指数价格序列 (如SPY)
        period: 计算窗口
    """
    stock_returns = close.pct_change()
    bench_returns = benchmark.pct_change()
    
    betas = pd.DataFrame(index=close.index, columns=close.columns)
    
    for col in close.columns:
        # 滚动计算协方差和方差
        cov = stock_returns[col].rolling(period).cov(bench_returns)
        var = bench_returns.rolling(period).var()
        betas[col] = cov / var
    
    return betas


# ==========================================
# 工具函数
# ==========================================

def zscore_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """
    横截面Z-score标准化
    每一天对所有股票进行标准化
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def rank_cross_sectional(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """
    横截面排名
    每一天对所有股票排名
    """
    return df.rank(axis=1, ascending=ascending, pct=True)


def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    缩尾处理 (去除极端值)
    """
    lower_bound = df.quantile(lower, axis=1)
    upper_bound = df.quantile(upper, axis=1)
    
    return df.clip(lower=lower_bound, upper=upper_bound, axis=0)


# ==========================================
# 因子引擎
# ==========================================

class FactorEngine:
    """
    因子引擎 - 统一管理因子计算
    """
    
    # 因子注册表
    FACTOR_REGISTRY: Dict[str, Callable] = {
        'momentum_12_1': momentum_12_1,
        'pullback_5d': pullback_5d,
        'rsi': rsi_factor,
        'volatility': volatility_factor,
        'volume_surge': volume_surge,
        'trend_strength': trend_strength,
        'beta': beta_factor,
    }
    
    def __init__(self, config_path: str = "config.yaml"):
        """加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.factor_config = self.config['factors']
        logger.info("FactorEngine initialized")
    
    def compute_all_factors(
        self,
        close: pd.DataFrame,
        volume: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        计算所有启用的因子
        
        返回:
            字典 {因子名: DataFrame}
        """
        factors = {}
        
        # 动量因子
        if self.factor_config['momentum']['enabled']:
            cfg = self.factor_config['momentum']
            factors['momentum_12_1'] = momentum_12_1(
                close,
                lookback=cfg.get('lookback', 252),
                skip=cfg.get('skip_recent', 21)
            )
        
        # 反转因子
        if self.factor_config['pullback']['enabled']:
            cfg = self.factor_config['pullback']
            factors['pullback_5d'] = pullback_5d(
                close,
                lookback=cfg.get('lookback', 5)
            )
        
        # 可扩展: 添加更多因子
        # if 'rsi' in self.factor_config and self.factor_config['rsi']['enabled']:
        #     factors['rsi'] = rsi_factor(close)
        
        logger.info(f"Computed {len(factors)} factors")
        return factors
    
    def compute_composite_score(
        self,
        factors: Dict[str, pd.DataFrame],
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        计算复合得分
        
        参数:
            factors: 因子字典
            normalize: 是否先标准化
        
        返回:
            复合得分 DataFrame
        """
        score = pd.DataFrame(0, index=factors[list(factors.keys())[0]].index, 
                            columns=factors[list(factors.keys())[0]].columns)
        
        for factor_name, factor_df in factors.items():
            # 获取权重
            weight = self._get_factor_weight(factor_name)
            
            if weight == 0:
                continue
            
            # 标准化
            if normalize:
                factor_norm = zscore_cross_sectional(factor_df)
            else:
                factor_norm = factor_df
            
            # 加权
            score += weight * factor_norm
        
        return score
    
    def _get_factor_weight(self, factor_name: str) -> float:
        """获取因子权重"""
        # 简化的映射
        mapping = {
            'momentum_12_1': self.factor_config['momentum']['weight'],
            'pullback_5d': self.factor_config['pullback']['weight'],
        }
        return mapping.get(factor_name, 0.0)


# ==========================================
# 因子分析工具
# ==========================================

class FactorAnalyzer:
    """
    因子分析器 - 单因子测试、IC分析
    """
    
    @staticmethod
    def calculate_ic(
        factor: pd.DataFrame,
        forward_returns: pd.DataFrame,
        periods: List[int] = [1, 5, 20]
    ) -> pd.DataFrame:
        """
        计算信息系数 (IC)
        
        参数:
            factor: 因子值
            forward_returns: 未来收益
            periods: 预测期
        
        返回:
            IC时间序列
        """
        results = {}
        
        for period in periods:
            # 计算未来N日收益
            fwd_ret = forward_returns.shift(-period)
            
            # 每日横截面相关系数
            ic_series = []
            for date in factor.index:
                if date not in fwd_ret.index:
                    continue
                
                f = factor.loc[date].dropna()
                r = fwd_ret.loc[date].dropna()
                
                # 取交集
                common = f.index.intersection(r.index)
                if len(common) < 10:
                    continue
                
                ic = f[common].corr(r[common], method='spearman')
                ic_series.append({'date': date, 'ic': ic})
            
            results[f'{period}d'] = pd.DataFrame(ic_series).set_index('date')
        
        return pd.DataFrame(results)
    
    @staticmethod
    def ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
        """
        IC统计摘要
        """
        summary = {
            'IC均值': ic_df.mean(),
            'IC标准差': ic_df.std(),
            'IR (IC均值/标准差)': ic_df.mean() / ic_df.std(),
            'IC>0占比': (ic_df > 0).mean(),
            'IC绝对值均值': ic_df.abs().mean(),
        }
        return pd.DataFrame(summary).T
    
    @staticmethod
    def factor_quantile_returns(
        factor: pd.DataFrame,
        returns: pd.DataFrame,
        quantiles: int = 5,
        periods: int = 20
    ) -> pd.DataFrame:
        """
        因子分组收益测试
        将股票按因子值分N组，计算各组未来收益
        """
        fwd_ret = returns.shift(-periods)
        
        group_rets = []
        
        for date in factor.index[:-periods]:
            f = factor.loc[date].dropna()
            r = fwd_ret.loc[date].dropna()
            
            common = f.index.intersection(r.index)
            if len(common) < quantiles * 2:
                continue
            
            # 分组
            f_common = f[common]
            r_common = r[common]
            
            labels = pd.qcut(f_common, q=quantiles, labels=False, duplicates='drop')
            
            for q in range(quantiles):
                mask = labels == q
                if mask.sum() > 0:
                    group_rets.append({
                        'date': date,
                        'quantile': q + 1,
                        'return': r_common[mask].mean()
                    })
        
        df = pd.DataFrame(group_rets)
        return df.pivot(index='date', columns='quantile', values='return')


# ==========================================
# 滚动窗口回测
# ==========================================

def rolling_window_test(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    window_years: int = 4,
    step_years: int = 1
):
    """
    滚动窗口测试
    每4年为一个窗口，步进1年，检验因子稳定性
    
    这解决你提到的问题: "回滚4年窗口,每个四年都看看同样的因子是不是平均有效"
    """
    from data_manager import DataManager
    
    dm = DataManager()
    engine = FactorEngine()
    analyzer = FactorAnalyzer()
    
    start_year = close.index[0].year
    end_year = close.index[-1].year
    
    results = []
    
    for year in range(start_year, end_year - window_years + 1, step_years):
        window_start = f"{year}-01-01"
        window_end = f"{year + window_years}-12-31"
        
        # 切片数据
        mask = (close.index >= window_start) & (close.index <= window_end)
        close_window = close[mask]
        
        if len(close_window) < 252:
            continue
        
        print(f"\n{'='*60}")
        print(f"窗口: {window_start} ~ {window_end}")
        print(f"{'='*60}")
        
        # 计算因子
        factors = engine.compute_all_factors(close_window, volume)
        
        # IC分析
        returns = close_window.pct_change()
        
        for factor_name, factor_df in factors.items():
            ic_df = analyzer.calculate_ic(factor_df, returns, periods=[5, 20])
            ic_stats = analyzer.ic_summary(ic_df)
            
            print(f"\n[{factor_name}] IC统计:")
            print(ic_stats)
            
            results.append({
                'window': f"{year}-{year+window_years}",
                'factor': factor_name,
                'ic_mean_5d': ic_df['5d'].mean(),
                'ic_mean_20d': ic_df['20d'].mean(),
                'ir_5d': ic_df['5d'].mean() / ic_df['5d'].std()
            })
    
    # 汇总
    results_df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("所有窗口汇总:")
    print(f"{'='*60}")
    print(results_df.groupby('factor').mean())
    
    return results_df


# ==========================================
# 独立测试
# ==========================================

def test_factors():
    """
    测试因子计算 - Ctrl+F5快速运行
    """
    print("\n" + "="*60)
    print("因子库测试")
    print("="*60)
    
    # 1. 加载数据
    from data_manager import DataManager
    dm = DataManager()
    
    universe = dm.get_universe_list()[:30]  # 测试前30个
    close, volume = dm.get_market_data(universe, use_cache=True)
    
    print(f"\n数据: {close.shape[0]}天 x {close.shape[1]}股")
    
    # 2. 计算因子
    engine = FactorEngine()
    factors = engine.compute_all_factors(close, volume)
    
    print(f"\n计算了 {len(factors)} 个因子:")
    for name, df in factors.items():
        print(f"  - {name}: {df.shape}")
    
    # 3. 复合得分
    score = engine.compute_composite_score(factors)
    print(f"\n复合得分: {score.shape}")
    print(score.tail())
    
    # 4. IC分析
    print("\n" + "="*60)
    print("IC分析 (最近1年)")
    print("="*60)
    
    returns = close.pct_change()
    analyzer = FactorAnalyzer()
    
    recent_data = close.last('365D')
    recent_returns = returns.last('365D')
    
    for factor_name, factor_df in factors.items():
        recent_factor = factor_df.last('365D')
        
        # 確保有足夠的數據計算 forward returns
        ic_df = analyzer.calculate_ic(recent_factor, recent_returns, periods=[5, 20])
        
        # 檢查是否有有效結果
        if ic_df.empty or ic_df.isna().all().all():
            print(f"\n[{factor_name}] ⚠️  數據不足，無法計算 IC")
            continue
        
        ic_stats = analyzer.ic_summary(ic_df)
        print(f"\n[{factor_name}]")
        print(ic_stats)
    
    print("\n✅ 因子测试完成!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    test_factors()