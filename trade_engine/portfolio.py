import logging
import math
import numpy as np
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class PortfolioManager:
    """
    Manages portfolio state and generates orders to match target weights.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        config: Dict containing 'costs' and 'execution' settings.
        e.g.
        costs:
            min_trade_amount: 50
        execution:
            max_order_size_pct: 0.1
        """
        self.config = config
        self.min_trade_amt = config.get('costs', {}).get('min_trade_amount', 10.0)
        self.max_order_pct = config.get('execution', {}).get('max_order_size_pct', 0.1)

    def generate_orders(
        self,
        target_weights: Dict[str, float],
        current_positions: Dict[str, Any], # Dict[str, Position] or similar obj with .qty .current_price
        account_equity: float,
        current_prices: Dict[str, float]
    ) -> List[Dict]:
        """
        Generate list of orders to converge current positions to target weights.
        """
        orders = []

        # 1. Sell positions not in target (or where target is 0)
        for symbol, pos in current_positions.items():
            if symbol not in target_weights or target_weights[symbol] <= 0:
                # If Position object
                qty = getattr(pos, 'qty', pos) # Handle if passed simple qty or Position obj
                if qty > 0:
                    orders.append({
                        'symbol': symbol,
                        'side': 'sell',
                        'qty': float(qty),
                        'reason': 'not_in_target'
                    })

        # 2. Rebalance existing or new positions
        for symbol, target_weight in target_weights.items():
            if target_weight <= 0:
                continue

            target_val = account_equity * target_weight
            
            # Get current state
            pos = current_positions.get(symbol)
            curr_qty = getattr(pos, 'qty', 0) if pos else 0
            
            curr_price = current_prices.get(symbol, 0)
            if curr_price == 0:
                logger.warning(f"No price for {symbol}, skipping.")
                continue

            curr_val = curr_qty * curr_price
            diff_val = target_val - curr_val

            # Guard 1: Max order size
            if abs(diff_val) > account_equity * self.max_order_pct:
                 # Cap it
                 logger.warning(f"Order for {symbol} capped at {self.max_order_pct:.0%} of equity")
                 diff_val = np.sign(diff_val) * (account_equity * self.max_order_pct)

            # Guard 2: Min trade amount
            if abs(diff_val) < self.min_trade_amt:
                continue

            # Generate Order
            if diff_val > 0:
                # Buying
                orders.append({
                    'symbol': symbol,
                    'side': 'buy',
                    'notional': round(diff_val, 2),
                    'reason': 'rebalance_buy'
                })
            else:
                # Selling
                # Calc qty
                qty_to_sell = abs(diff_val) / curr_price
                # Round down, maybe floor? Alpaca allows fractional but safe to be precise
                # If using fractional, we can just pass notional for selling? 
                # Alpaca sell usually requires qty.
                qty_to_sell = math.floor(qty_to_sell * 100) / 100
                
                if qty_to_sell > 0:
                    orders.append({
                        'symbol': symbol,
                        'side': 'sell',
                        'qty': qty_to_sell,
                        'reason': 'rebalance_sell'
                    })

        return orders
