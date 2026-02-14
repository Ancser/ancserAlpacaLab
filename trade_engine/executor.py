import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus

logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    qty: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float

class Executor(ABC):
    """
    Abstract Base Class for executing trades.
    """
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]:
        """Returns {symbol: Position}"""
        pass

    @abstractmethod
    def cancel_all_orders(self):
        pass

    @abstractmethod
    def submit_orders(self, orders: List[Dict]):
        """
        Submit a list of orders. 
        Order dict format: {'symbol': str, 'side': 'buy'/'sell', 'qty': float, 'notional': float, 'reason': str}
        """
        pass

class AlpacaExecutor(Executor):
    def __init__(self, paper: bool = True):
        api_key = os.getenv("APCA_API_KEY_ID")
        secret_key = os.getenv("APCA_API_SECRET_KEY")
        if not api_key or not secret_key:
             raise ValueError("Missing Alpaca API credentials")
             
        self.client = TradingClient(api_key, secret_key, paper=paper)

    def get_account_info(self) -> Dict:
        acct = self.client.get_account()
        return {
            'equity': float(acct.equity),
            'cash': float(acct.cash),
            'buying_power': float(acct.buying_power),
            'id': acct.account_number # PII warning: usually safe to log ID, but be careful
        }

    def get_positions(self) -> Dict[str, Position]:
        try:
            positions = self.client.get_all_positions()
            result = {}
            for p in positions:
                result[p.symbol] = Position(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    current_price=float(p.current_price),
                    market_value=float(p.market_value),
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_plpc=float(p.unrealized_plpc)
                )
            return result
        except Exception as e:
            logger.error(f"Failed to get positions: {e}")
            return {}

    def cancel_all_orders(self):
        try:
            self.client.cancel_orders()
            logger.info("Cancelled all open orders.")
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")

    def submit_orders(self, orders: List[Dict]):
        if not orders:
            return

        logger.info(f"Submitting {len(orders)} orders...")
        
        for order in orders:
            symbol = order['symbol']
            side = order['side'] # 'buy' or 'sell'
            reason = order.get('reason', '')
            
            try:
                # Map side
                alpaca_side = OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL
                
                req = None
                if 'notional' in order and order['notional'] > 0:
                     req = MarketOrderRequest(
                        symbol=symbol,
                        notional=order['notional'],
                        side=alpaca_side,
                        time_in_force=TimeInForce.DAY
                    )
                     logger.info(f"  {side.upper()} {symbol} ${order['notional']} ({reason})")
                
                elif 'qty' in order and order['qty'] > 0:
                    req = MarketOrderRequest(
                        symbol=symbol,
                        qty=order['qty'],
                        side=alpaca_side,
                        time_in_force=TimeInForce.DAY
                    )
                    logger.info(f"  {side.upper()} {symbol} {order['qty']} shares ({reason})")
                
                if req:
                    self.client.submit_order(req)
                
            except Exception as e:
                logger.error(f"  Failed to submit order for {symbol}: {e}")
