"""
Alpaca执行器 (修复版)
-----------------
修复内容:
1. 添加止损/止盈逻辑
2. 改进错误处理和edge cases
3. 添加头寸大小限制
4. 改进订单管理
5. 添加详细日志

使用方法:
    python alpaca_execute_fixed.py --paper          (模拟交易)
    python alpaca_execute_fixed.py --paper --dry-run (不实际下单)
    python alpaca_execute_fixed.py --paper --force   (强制执行)
"""

import os
import sys
import time
import math
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import numpy as np
import yaml

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest, GetCalendarRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment

# 导入我们的模块
from data_manager import DataManager
from factor_library import FactorEngine

# ==========================================
# 配置
# ==========================================

LOG_DIR = Path("logs")
STATE_FILE = Path("state.json")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / f"execution_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==========================================
# 工具函数
# ==========================================

def load_config(config_path: str = "config.yaml") -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def round_down(n: float, decimals: int = 0) -> float:
    """向下取整"""
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier


def get_alpaca_clients(paper: bool = True):
    """初始化Alpaca客户端"""
    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")
    
    if not api_key or not secret_key:
        sys.exit("[错误] .env文件中缺少API密钥")
    
    trader = TradingClient(api_key, secret_key, paper=paper)
    data_client = StockHistoricalDataClient(api_key, secret_key)
    
    return trader, data_client


def fetch_alpaca_history(data_client, symbols: list, days_back: int = 400):
    """
    从Alpaca获取历史数据
    """
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days_back)
    
    logger.info(f"下载 {len(symbols)} 个标的数据: {start_dt.date()} 至 {end_dt.date()}")
    
    chunk_size = 50
    all_bars = []
    unique_syms = list(set(symbols))
    
    for i in range(0, len(unique_syms), chunk_size):
        chunk = unique_syms[i:i+chunk_size]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame.Day,
                start=start_dt,
                end=end_dt,
                adjustment=Adjustment.ALL,
                feed=DataFeed.IEX
            )
            bars = data_client.get_stock_bars(req).df
            if not bars.empty:
                all_bars.append(bars)
                logger.info(f"  完成 {i+1}-{min(i+chunk_size, len(unique_syms))}/{len(unique_syms)}")
        except Exception as e:
            logger.warning(f"  下载失败 {chunk[0]}: {e}")
            continue
    
    if not all_bars:
        logger.error("未获取到任何数据")
        return pd.DataFrame(), pd.DataFrame()
    
    # 合并数据
    df = pd.concat(all_bars).reset_index()
    df['date'] = df['timestamp'].dt.date
    df = df.set_index('date')
    
    # 转换为宽格式
    close = df.pivot(columns='symbol', values='close').ffill()
    volume = df.pivot(columns='symbol', values='volume').fillna(0)
    
    logger.info(f"✅ 数据下载完成: {close.shape}")
    
    return close, volume


def get_current_positions(trader) -> dict:
    """获取当前持仓 {symbol: qty}"""
    try:
        positions = trader.get_all_positions()
        return {p.symbol: float(p.qty) for p in positions}
    except Exception as e:
        logger.error(f"获取持仓失败: {e}")
        return {}


def get_position_details(trader) -> pd.DataFrame:
    """获取持仓详细信息"""
    try:
        positions = trader.get_all_positions()
        data = []
        for p in positions:
            data.append({
                'symbol': p.symbol,
                'qty': float(p.qty),
                'avg_entry': float(p.avg_entry_price),
                'current_price': float(p.current_price),
                'market_value': float(p.market_value),
                'unrealized_pl': float(p.unrealized_pl),
                'unrealized_plpc': float(p.unrealized_plpc)
            })
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"获取持仓详情失败: {e}")
        return pd.DataFrame()


def cancel_open_orders(trader):
    """取消所有未成交订单"""
    try:
        orders = trader.get_orders(filter=GetOrdersRequest(status=QueryOrderStatus.OPEN))
        for order in orders:
            trader.cancel_order_by_id(order.id)
            logger.info(f"  已取消订单: {order.symbol} {order.side}")
        logger.info(f"✅ 取消了 {len(orders)} 个未成交订单")
    except Exception as e:
        logger.warning(f"取消订单失败: {e}")


def check_stop_loss_take_profit(trader, config: dict):
    """
    检查止损/止盈条件
    
    这是你原代码缺少的部分!
    """
    stop_loss_cfg = config['portfolio']['stop_loss']
    take_profit_cfg = config['portfolio']['take_profit']
    
    if not (stop_loss_cfg['enabled'] or take_profit_cfg['enabled']):
        return []
    
    positions_df = get_position_details(trader)
    
    if positions_df.empty:
        return []
    
    orders_to_place = []
    
    for _, pos in positions_df.iterrows():
        symbol = pos['symbol']
        qty = pos['qty']
        pl_pct = pos['unrealized_plpc']
        
        # 止损检查
        if stop_loss_cfg['enabled'] and pl_pct <= stop_loss_cfg['threshold']:
            logger.warning(f"🛑 止损触发: {symbol} 盈亏={pl_pct:.2%}")
            orders_to_place.append({
                'symbol': symbol,
                'side': OrderSide.SELL,
                'qty': qty,
                'reason': 'stop_loss'
            })
        
        # 止盈检查
        elif take_profit_cfg['enabled'] and pl_pct >= take_profit_cfg['threshold']:
            logger.info(f"💰 止盈触发: {symbol} 盈亏={pl_pct:.2%}")
            orders_to_place.append({
                'symbol': symbol,
                'side': OrderSide.SELL,
                'qty': qty,
                'reason': 'take_profit'
            })
    
    return orders_to_place


def is_rebalance_day(trader, force: bool = False) -> tuple:
    """
    判断是否为调仓日
    
    返回: (是否调仓, 原因说明)
    """
    if force:
        return True, "强制执行"
    
    today = datetime.now().date()
    
    # 计算本周范围
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=4)
    
    try:
        # 查询本周交易日历
        cal_req = GetCalendarRequest(start=start_of_week, end=end_of_week)
        calendar = trader.get_calendar(cal_req)
        
        if not calendar:
            logger.warning("未获取到交易日历")
            return today.weekday() == 4, "使用周五作为默认"
        
        # 本周最后一个交易日
        last_trading_day = calendar[-1].date
        
        if today == last_trading_day:
            return True, f"本周最后交易日 ({last_trading_day})"
        else:
            return False, f"非调仓日 (下次: {last_trading_day})"
            
    except Exception as e:
        logger.error(f"获取日历失败: {e}")
        # Fallback: 周五
        return today.weekday() == 4, "日历失败,使用周五"


# ==========================================
# 核心策略逻辑
# ==========================================

def calculate_target_weights(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    config: dict
) -> dict:
    """
    计算目标权重
    
    返回: {symbol: weight}
    """
    # 1. 初始化因子引擎
    engine = FactorEngine()
    
    # 2. 分离股票和基准
    benchmarks = config['data']['benchmarks']
    defensive = config['data']['defensive_assets']
    exclude_cols = benchmarks + defensive
    
    stock_cols = [c for c in close_df.columns if c not in exclude_cols]
    close_stocks = close_df[stock_cols]
    volume_stocks = volume_df[stock_cols]
    
    # 3. 计算因子
    logger.info("计算因子...")
    factors = engine.compute_all_factors(close_stocks, volume_stocks)
    
    # 4. 复合得分
    scores = engine.compute_composite_score(factors)
    latest_scores = scores.iloc[-1].dropna().sort_values(ascending=False)
    
    logger.info(f"  有效因子得分: {len(latest_scores)} 个股票")
    
    # 5. 风险开关检查
    regime_cfg = config['regime']
    risk_on = True
    
    if regime_cfg['enabled']:
        spy_close = close_df[regime_cfg['indicator']]
        spy_sma = spy_close.rolling(regime_cfg['sma_length']).mean().iloc[-1]
        spy_mom = spy_close.pct_change(regime_cfg['momentum_length']).iloc[-1]
        spy_price = spy_close.iloc[-1]
        
        risk_on = (spy_price > spy_sma) and (spy_mom > 0)
        
        logger.info(f"风险状态: SPY=${spy_price:.2f}, SMA=${spy_sma:.2f}, Mom={spy_mom:.2%} -> {'🟢 RISK ON' if risk_on else '🔴 RISK OFF'}")
    
    # 6. 生成目标权重
    target_weights = {}
    
    if not risk_on:
        # 防御模式
        defensive_alloc = config['regime']['defensive_allocation']
        logger.info("💤 防御模式: 使用防御资产配置")
        return defensive_alloc
    
    # 7. 主动模式 - 应用过滤器
    filter_cfg = config['filters']
    
    latest_price = close_stocks.iloc[-1]
    latest_volume = volume_stocks.iloc[-1]
    avg_dollar_volume = (latest_price * latest_volume.rolling(filter_cfg['adv_window']).mean()).iloc[-1]
    
    # 过滤
    valid_stocks = latest_scores.index.tolist()
    
    # 价格过滤
    valid_stocks = [s for s in valid_stocks if latest_price.get(s, 0) > filter_cfg['min_price']]
    logger.info(f"  价格过滤后: {len(valid_stocks)} 个")
    
    # 流动性过滤
    valid_stocks = [s for s in valid_stocks if avg_dollar_volume.get(s, 0) > filter_cfg['min_adv_dollar']]
    logger.info(f"  流动性过滤后: {len(valid_stocks)} 个")
    
    # 8. 选择Top N
    portfolio_cfg = config['portfolio']
    top_n = portfolio_cfg['top_n']
    
    if len(valid_stocks) < portfolio_cfg['min_names_to_trade']:
        logger.warning(f"⚠️  有效股票不足 ({len(valid_stocks)} < {portfolio_cfg['min_names_to_trade']}), 转防御")
        return config['regime']['defensive_allocation']
    
    top_picks = valid_stocks[:top_n]
    
    # 9. 等权重 + 上限
    base_weight = 1.0 / len(top_picks)
    max_weight = portfolio_cfg['max_weight']
    
    for symbol in top_picks:
        target_weights[symbol] = min(base_weight, max_weight)
    
    # 10. 重新归一化
    total_weight = sum(target_weights.values())
    if total_weight > 0:
        target_weights = {k: v/total_weight for k, v in target_weights.items()}
    
    logger.info(f"✅ 目标组合: {len(target_weights)} 个股票")
    logger.info(f"  Top 5: {list(target_weights.keys())[:5]}")
    
    return target_weights


def generate_orders(
    target_weights: dict,
    current_positions: dict,
    account_equity: float,
    current_prices: dict,
    config: dict
) -> list:
    """
    生成订单列表
    
    返回: [{symbol, side, qty/notional, reason}]
    """
    orders = []
    min_trade_amt = config['costs']['min_trade_amount']
    max_order_pct = config['execution']['max_order_size_pct']
    
    # 1. 卖出不在目标中的持仓
    for symbol, current_qty in current_positions.items():
        if symbol not in target_weights:
            orders.append({
                'symbol': symbol,
                'side': OrderSide.SELL,
                'qty': current_qty,
                'reason': 'not_in_target'
            })
            logger.info(f"  卖出 {symbol}: 不在目标中")
    
    # 2. 调整目标持仓
    for symbol, target_weight in target_weights.items():
        target_value = account_equity * target_weight
        current_qty = current_positions.get(symbol, 0)
        current_price = current_prices.get(symbol, 0)
        
        if current_price == 0:
            logger.warning(f"  跳过 {symbol}: 无价格数据")
            continue
        
        current_value = current_qty * current_price
        diff_value = target_value - current_value
        
        # 安全检查: 单笔订单不超过账户一定比例
        if abs(diff_value) > account_equity * max_order_pct:
            logger.warning(f"  限制 {symbol}: 订单过大 ${abs(diff_value):,.0f} > {max_order_pct:.0%} 账户")
            diff_value = np.sign(diff_value) * account_equity * max_order_pct
        
        # 买入
        if diff_value > min_trade_amt:
            orders.append({
                'symbol': symbol,
                'side': OrderSide.BUY,
                'notional': round(diff_value, 2),
                'reason': 'rebalance_buy'
            })
        
        # 卖出
        elif diff_value < -min_trade_amt:
            qty_to_sell = abs(diff_value) / current_price
            qty_to_sell = round_down(qty_to_sell, 2)
            
            if qty_to_sell > 0:
                orders.append({
                    'symbol': symbol,
                    'side': OrderSide.SELL,
                    'qty': qty_to_sell,
                    'reason': 'rebalance_sell'
                })
    
    return orders


def execute_orders(trader, orders: list, dry_run: bool = False):
    """
    执行订单
    """
    if not orders:
        logger.info("没有需要执行的订单")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"准备执行 {len(orders)} 个订单")
    logger.info(f"{'='*60}")
    
    for i, order in enumerate(orders, 1):
        symbol = order['symbol']
        side = order['side']
        reason = order.get('reason', 'unknown')
        
        if side == OrderSide.SELL:
            qty = order['qty']
            logger.info(f"[{i}/{len(orders)}] 卖出 {symbol} x{qty} ({reason})")
            
            if not dry_run:
                try:
                    req = MarketOrderRequest(
                        symbol=symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY
                    )
                    trader.submit_order(req)
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"  ❌ 订单失败: {e}")
        
        else:  # BUY
            notional = order['notional']
            logger.info(f"[{i}/{len(orders)}] 买入 {symbol} ${notional:,.2f} ({reason})")
            
            if not dry_run:
                try:
                    req = MarketOrderRequest(
                        symbol=symbol,
                        notional=notional,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY
                    )
                    trader.submit_order(req)
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"  ❌ 订单失败: {e}")
    
    if dry_run:
        logger.info("\n🔵 模拟模式: 未实际下单")
    else:
        logger.info("\n✅ 订单提交完成")


# ==========================================
# 主程序
# ==========================================

def main(args):
    """主执行流程"""
    
    logger.info("\n" + "="*60)
    logger.info("Alpaca执行器启动")
    logger.info("="*60)
    
    # 1. 加载配置
    config = load_config()
    
    # 2. 初始化客户端
    trader, data_client = get_alpaca_clients(paper=args.paper)
    
    # 3. 检查是否为调仓日
    should_rebalance, reason = is_rebalance_day(trader, args.force)
    logger.info(f"调仓检查: {reason}")
    
    if not should_rebalance:
        logger.info("⏸️  今日无需调仓")
        
        # 即使不调仓,也检查止损/止盈
        stop_orders = check_stop_loss_take_profit(trader, config)
        if stop_orders:
            logger.info(f"发现 {len(stop_orders)} 个止损/止盈触发")
            execute_orders(trader, stop_orders, args.dry_run)
        
        return
    
    logger.info("🔄 开始调仓流程...")
    
    # 4. 获取股票池
    dm = DataManager()
    universe = dm.get_universe_list()
    logger.info(f"股票池: {len(universe)} 个标的")
    
    # 5. 下载数据
    all_symbols = list(set(
        universe + 
        config['data']['benchmarks'] + 
        config['data']['defensive_assets']
    ))
    
    close_df, volume_df = fetch_alpaca_history(
        data_client,
        all_symbols,
        days_back=config['data']['lookback_days']
    )
    
    if close_df.empty:
        logger.error("❌ 数据获取失败")
        return
    
    # 6. 计算目标权重
    target_weights = calculate_target_weights(close_df, volume_df, config)
    
    # 7. 获取账户信息
    account = trader.get_account()
    equity = float(account.equity)
    cash = float(account.cash)
    buying_power = float(account.buying_power)
    
    logger.info(f"\n账户状态:")
    logger.info(f"  权益: ${equity:,.2f}")
    logger.info(f"  现金: ${cash:,.2f}")
    logger.info(f"  购买力: ${buying_power:,.2f}")
    
    # 8. 获取当前持仓
    current_positions = get_current_positions(trader)
    logger.info(f"  当前持仓: {len(current_positions)} 个")
    
    # 9. 取消未成交订单
    if not args.dry_run:
        cancel_open_orders(trader)
    
    # 10. 生成订单
    current_prices = close_df.iloc[-1].to_dict()
    
    orders = generate_orders(
        target_weights=target_weights,
        current_positions=current_positions,
        account_equity=equity,
        current_prices=current_prices,
        config=config
    )
    
    # 11. 执行订单
    execute_orders(trader, orders, args.dry_run)
    
    logger.info("\n" + "="*60)
    logger.info("执行完成")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alpaca量化交易执行器")
    parser.add_argument("--paper", action="store_true", help="使用模拟账户")
    parser.add_argument("--dry-run", action="store_true", help="模拟运行(不下单)")
    parser.add_argument("--force", action="store_true", help="强制执行(忽略日期检查)")
    
    args = parser.parse_args()
    
    try:
        main(args)
    except KeyboardInterrupt:
        logger.info("\n用户中断")
    except Exception as e:
        logger.exception("致命错误")
        sys.exit(1)