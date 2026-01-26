"""
数据管理器 - 统一数据接口
负责数据获取、缓存、清洗
"""

import os
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


class DataManager:
    """
    统一数据管理接口
    - 自动缓存避免重复下载
    - 支持多数据源
    - 数据验证与清洗
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.cache_dir = Path(self.config['data']['cache_dir'])
        self.cache_dir.mkdir(exist_ok=True)
        
        self.source = self.config['data']['source']
        logger.info(f"DataManager initialized with source: {self.source}")
    
    def get_market_data(
        self,
        universe: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取市场数据 (OHLCV)
        
        返回:
            close_df: (date, symbols) 收盘价
            volume_df: (date, symbols) 成交量
        """
        # 自动计算日期
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            lookback = self.config['data']['lookback_days']
            start_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
        
        # 构建完整股票列表
        all_symbols = self._build_symbol_list(universe)
        
        # 检查缓存
        cache_key = f"{self.source}_{start_date}_{end_date}_{'_'.join(sorted(all_symbols[:5]))}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if use_cache and cache_file.exists():
            logger.info(f"Loading from cache: {cache_file.name}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data['close'], data['volume']
        
        # 下载数据
        logger.info(f"Downloading {len(all_symbols)} symbols from {start_date} to {end_date}")
        close_df, volume_df = self._download_data(all_symbols, start_date, end_date)
        
        # 数据清洗
        close_df, volume_df = self._clean_data(close_df, volume_df)
        
        # 保存缓存
        if use_cache:
            with open(cache_file, 'wb') as f:
                pickle.dump({'close': close_df, 'volume': volume_df}, f)
            logger.info(f"Cached to: {cache_file.name}")
        
        return close_df, volume_df
    
    def _build_symbol_list(self, universe: List[str]) -> List[str]:
        """构建完整的符号列表（股票 + 基准 + 防御资产）"""
        symbols = set(universe)
        
        # 添加基准
        symbols.update(self.config['data']['benchmarks'])
        
        # 添加防御资产
        symbols.update(self.config['data']['defensive_assets'])
        
        # 移除排除列表
        exclude = set(self.config['universe'].get('exclude_tickers', []))
        symbols -= exclude
        
        return sorted(symbols)
    
    def _download_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """根据配置选择数据源下载"""
        
        if self.source == "yahoo":
            return self._download_yahoo(symbols, start_date, end_date)
        elif self.source == "alpaca":
            return self._download_alpaca(symbols, start_date, end_date)
        else:
            raise ValueError(f"Unsupported data source: {self.source}")
    
    def _download_yahoo(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Yahoo Finance 下载"""
        import yfinance as yf
        
        try:
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
                group_by='column',
                threads=True
            )
            
            if isinstance(data.columns, pd.MultiIndex):
                close = data['Close'].copy()
                volume = data['Volume'].copy()
            else:
                # 单个股票
                close = data[['Close']].rename(columns={'Close': symbols[0]})
                volume = data[['Volume']].rename(columns={'Volume': symbols[0]})
            
            return close, volume
            
        except Exception as e:
            logger.error(f"Yahoo download failed: {e}")
            raise
    
    def _download_alpaca(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Alpaca 下载 (参考你的 alpaca_execute.py)"""
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import DataFeed, Adjustment
        
        api_key = os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("APCA_API_SECRET_KEY")
        
        if not api_key or not secret_key:
            raise ValueError("Missing Alpaca credentials in .env")
        
        client = StockHistoricalDataClient(api_key, secret_key)
        
        # 分块下载
        chunk_size = 50
        all_bars = []
        
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            try:
                req = StockBarsRequest(
                    symbol_or_symbols=chunk,
                    timeframe=TimeFrame.Day,
                    start=datetime.strptime(start_date, '%Y-%m-%d'),
                    end=datetime.strptime(end_date, '%Y-%m-%d'),
                    adjustment=Adjustment.ALL,
                    feed=DataFeed.IEX
                )
                bars = client.get_stock_bars(req).df
                if not bars.empty:
                    all_bars.append(bars)
            except Exception as e:
                logger.warning(f"Failed chunk {chunk[0]}: {e}")
        
        if not all_bars:
            raise ValueError("No data downloaded from Alpaca")
        
        # 合并与重塑
        df = pd.concat(all_bars).reset_index()
        df['date'] = df['timestamp'].dt.date
        df = df.set_index('date')
        
        close = df.pivot(columns='symbol', values='close')
        volume = df.pivot(columns='symbol', values='volume')
        
        return close, volume
    
    def _clean_data(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """数据清洗"""
        
        # 1. 移除全空列
        valid_cols = close.columns[close.notna().any()]
        close = close[valid_cols]
        volume = volume.loc[:, volume.columns.intersection(valid_cols)]
        
        # 2. 前向填充价格
        close = close.ffill()
        
        # 3. 成交量填0
        volume = volume.fillna(0)
        
        # 4. 移除数据不足的股票
        min_data_points = 252  # 至少1年数据
        sufficient_data = close.count() >= min_data_points
        valid_symbols = sufficient_data[sufficient_data].index
        
        close = close[valid_symbols]
        volume = volume[valid_symbols]
        
        # 5. 异常值检测
        returns = close.pct_change()
        suspicious = (returns.abs() > 0.5).sum()  # 单日涨跌超50%次数
        
        if suspicious.any():
            logger.warning(f"Suspicious returns detected: {suspicious[suspicious > 0].to_dict()}")
        
        logger.info(f"Cleaned data: {close.shape[0]} days, {close.shape[1]} symbols")
        
        return close, volume
    
    def get_universe_list(self) -> List[str]:
        """获取股票池列表"""
        mode = self.config['universe']['mode']
        
        if mode == "sp500":
            return self._get_sp500()
        elif mode == "nasdaq100":
            return self._get_nasdaq100()
        elif mode == "sp500_nasdaq100":
            sp500 = self._get_sp500()
            nasdaq100 = self._get_nasdaq100()
            return sorted(set(sp500 + nasdaq100))
        else:
            raise ValueError(f"Unknown universe mode: {mode}")
    
    def _get_sp500(self) -> List[str]:
        """获取S&P 500成分股"""
        url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
        df = pd.read_csv(url)
        
        # 处理列名差异
        symbol_col = [c for c in df.columns if 'symbol' in c.lower()][0]
        tickers = df[symbol_col].str.replace('.', '-', regex=False).tolist()
        
        return tickers
    
    def _get_nasdaq100(self) -> List[str]:
        """获取Nasdaq 100成分股"""
        csv_urls = [
            "https://yfiua.github.io/index-constituents/constituents-nasdaq100.csv",
        ]
        
        for url in csv_urls:
            try:
                df = pd.read_csv(url)
                symbol_col = [c for c in df.columns if 'symbol' in c.lower() or 'ticker' in c.lower()][0]
                tickers = df[symbol_col].str.replace('.', '-', regex=False).tolist()
                
                if len(tickers) >= 90:
                    return tickers
            except:
                continue
        
        # Fallback to Wikipedia
        import urllib.request
        url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        html = urllib.request.urlopen(req, timeout=20).read()
        tables = pd.read_html(html)
        
        for table in tables:
            cols = [c.lower() for c in table.columns]
            if any('ticker' in c or 'symbol' in c for c in cols):
                symbol_col = [c for c in table.columns if 'ticker' in c.lower() or 'symbol' in c.lower()][0]
                tickers = table[symbol_col].str.replace('.', '-', regex=False).tolist()
                return [t for t in tickers if t and t.upper() == t]
        
        raise ValueError("Failed to fetch Nasdaq 100 constituents")


# ============= 独立测试功能 =============

def test_data_manager():
    """测试数据管理器 - 按 Ctrl+F5 快速测试"""
    
    print("\n" + "="*60)
    print("数据管理器测试")
    print("="*60)
    
    # 初始化
    dm = DataManager("config.yaml")
    
    # 1. 测试获取股票池
    print("\n[1] 获取股票池...")
    universe = dm.get_universe_list()
    print(f"    股票池大小: {len(universe)}")
    print(f"    前10个: {universe[:10]}")
    
    # 2. 测试数据下载
    print("\n[2] 下载市场数据...")
    test_symbols = universe[:20]  # 测试前20个
    
    close, volume = dm.get_market_data(
        universe=test_symbols,
        start_date="2024-01-01",
        end_date="2024-12-31",
        use_cache=True
    )
    
    print(f"    收盘价维度: {close.shape}")
    print(f"    成交量维度: {volume.shape}")
    print(f"\n    最新5天数据:")
    print(close.tail())
    
    # 3. 数据质量检查
    print("\n[3] 数据质量检查...")
    print(f"    缺失值: {close.isnull().sum().sum()}")
    print(f"    零值数量: {(close == 0).sum().sum()}")
    
    # 4. 基准数据
    print("\n[4] 基准与防御资产...")
    benchmarks = dm.config['data']['benchmarks']
    defensive = dm.config['data']['defensive_assets']
    
    for ticker in benchmarks + defensive:
        if ticker in close.columns:
            latest = close[ticker].iloc[-1]
            print(f"    {ticker}: ${latest:.2f}")
    
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # 运行测试
    test_data_manager()