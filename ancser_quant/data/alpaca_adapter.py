from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed, Adjustment
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
import polars as pl
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv() # Force load
from datetime import datetime
from typing import List, Dict
from .schema import MARKET_DATA_SCHEMA

class AlpacaAdapter:
    """
    Fetches data from Alpaca (Paid/Free) and adapts to Titan Schema.
    Also handles Account info.
    """
    def __init__(self):
        self.api_key = os.getenv("APCA_API_KEY_ID")
        self.secret_key = os.getenv("APCA_API_SECRET_KEY")
        if not self.api_key:
            raise ValueError("Alpaca API keys missing.")
            
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper=True) # Default to paper

    def get_account(self) -> Dict:
        """Fetch account information."""
        try:
            acct = self.trading_client.get_account()
            return {
                'equity': float(acct.equity),
                'buying_power': float(acct.buying_power),
                'cash': float(acct.cash),
                'daytrade_count': int(acct.daytrade_count),
                'status': acct.status,
                'currency': acct.currency
            }
        except Exception as e:
            print(f"[AlpacaAdapter] Get Account Error: {e}")
            return {'equity': 0.0, 'buying_power': 0.0, 'status': 'Error'}


    def fetch_history(self, symbols: List[str], start_date: str, end_date: str = None) -> pl.LazyFrame:
        print(f"[AlpacaAdapter] Fetching {len(symbols)} symbols...")
        
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=datetime.strptime(start_date, '%Y-%m-%d'),
            end=datetime.strptime(end_date, '%Y-%m-%d'),
            adjustment=Adjustment.ALL,
            feed=DataFeed.IEX # or SIP if paid
        )
        
        try:
            bars = self.data_client.get_stock_bars(req).df
        except Exception as e:
            print(f"[AlpacaAdapter] Error: {e}")
            return pl.LazyFrame({})
            
        if bars.empty:
            return pl.LazyFrame({})
            
        # Reset index to get timestamp and symbol as columns
        bars = bars.reset_index()
        
        # Convert to Polars
        df = pl.from_pandas(bars)
        
        # Rename and Cast
        # Alpaca: timestamp, symbol, open, high, low, close, volume, trade_count, vwap
        
        df = df.with_columns([
            pl.col("timestamp").alias("timestamp"), # already correct name
            pl.col("symbol").cast(pl.Categorical),
            pl.col("open").cast(pl.Float32),
            pl.col("high").cast(pl.Float32),
            pl.col("low").cast(pl.Float32),
            pl.col("close").cast(pl.Float32),
            pl.col("volume").cast(pl.Float32),
            pl.col("vwap").cast(pl.Float32),
            pl.col("trade_count").cast(pl.UInt32)
        ])
        
        # Select standard columns
        df = df.select(list(MARKET_DATA_SCHEMA.keys()))
        
        return df.lazy()

    def get_positions(self) -> List[Dict]:
        """Fetch current open positions."""
        try:
            positions = self.trading_client.get_all_positions()
            return [{
                'Symbol': p.symbol,
                'Qty': float(p.qty),
                'Market Value': float(p.market_value),
                'Avg Entry': float(p.avg_entry_price),
                'Current Price': float(p.current_price),
                'Unrealized P&L': float(p.unrealized_pl),
                'P&L %': float(p.unrealized_plpc) * 100
            } for p in positions]
        except Exception as e:
            print(f"[AlpacaAdapter] Get Positions Error: {e}")
            return []

    def get_portfolio_history(self, period: str = "1M", timeframe: str = "1D") -> pd.DataFrame:
        """Fetch portfolio history (Equity Curve)."""
        from alpaca.trading.requests import GetPortfolioHistoryRequest
        # TimeFrame enum from alpaca.trading.enums does not exist. Using strings directly.
        # GetPortfolioHistoryRequest takes period, timeframe, date_start, date_end, pnl_reset, extended_hours.
        # The error said unexpected keyword argument 'period' in get_portfolio_history, suggesting it expects a REQUEST OBJECT.
        
        try:
            # Construct Request Object
            req = GetPortfolioHistoryRequest(period=period, timeframe=timeframe, extended_hours=True)
            history = self.trading_client.get_portfolio_history(req)
            
            # Check if history is valid
            if not history or not hasattr(history, 'timestamp'):
                return pd.DataFrame()

            # Alpaca returns lists
            data = {
                'timestamp': history.timestamp,
                'equity': history.equity,
                'profit_loss': history.profit_loss,
                'profit_loss_pct': history.profit_loss_pct
            }
            
            df = pd.DataFrame(data)
            
            # Convert timestamp (Unix epoch seconds)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            return df.set_index('timestamp')
        except Exception as e:
            print(f"[AlpacaAdapter] Get History Error: {e}")
            return pd.DataFrame()
    def get_orders(self, status: str = 'all', limit: int = 20) -> List[Dict]:
        """Fetch recent orders."""
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import OrderSide, QueryOrderStatus

        try:
            req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit, nested=True)
            orders = self.trading_client.get_orders(filter=req)
            
            return [{
                'id': str(o.id),
                'symbol': o.symbol,
                'qty': float(o.qty) if o.qty else 0.0,
                'side': o.side.value,
                'type': o.type.value,
                'status': o.status.value,
                'filled_avg_price': float(o.filled_avg_price) if o.filled_avg_price else None,
                'notional': float(o.notional) if o.notional else None,
                'created_at': o.created_at,
                'filled_qty': float(o.filled_qty) if o.filled_qty else 0.0
            } for o in orders]
        except Exception as e:
            print(f"[AlpacaAdapter] Get Orders Error: {e}")
            return []

    def get_clock(self) -> Dict:
        """Fetch market clock (is_open, next_open, next_close)."""
        try:
            clock = self.trading_client.get_clock()
            return {
                'is_open': clock.is_open,
                'next_open': clock.next_open,
                'next_close': clock.next_close,
                'timestamp': clock.timestamp
            }
        except Exception as e:
            print(f"[AlpacaAdapter] Get Clock Error: {e}")
            return {'is_open': False}

    def submit_order(self, symbol: str, qty: float, side: str, type: str = 'market', time_in_force: str = 'day', notional: float = None):
        """
        Submit an order to Alpaca.
        Supports both quantity-based and notional-based (dollar amount) orders.
        """
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        
        try:
            # Determine Side
            side_enum = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
            
            req_params = {
                "symbol": symbol,
                "side": side_enum,
                "time_in_force": TimeInForce.DAY
            }
            
            if notional and notional > 0:
                req_params["notional"] = round(notional, 2)
            else:
                req_params["qty"] = qty
                
            req = MarketOrderRequest(**req_params)
                
            order = self.trading_client.submit_order(order_data=req)
            print(f"[AlpacaAdapter] Order Submitted: {side.upper()} {symbol} (ID: {order.id})")
            return order
            
        except Exception as e:
            print(f"[AlpacaAdapter] Submit Order Error: {e}")
            raise e

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch latest trade price for a list of symbols."""
        from alpaca.data.requests import StockLatestTradeRequest
        try:
            req = StockLatestTradeRequest(symbol_or_symbols=symbols)
            trades = self.data_client.get_stock_latest_trade(req)
            return {sym: float(trade.price) for sym, trade in trades.items()}
        except Exception as e:
            print(f"[AlpacaAdapter] Get Latest Prices Error: {e}")
            return {}

    def cancel_all_orders(self):
        """Cancel all open orders."""
        try:
            self.trading_client.cancel_orders()
            print("[AlpacaAdapter] All open orders cancelled.")
        except Exception as e:
            print(f"[AlpacaAdapter] Cancel Orders Error: {e}")
