import logging
import pandas as pd
from ancser_quant.data.alpaca_adapter import AlpacaAdapter

logger = logging.getLogger("AncserExecution")

class OrderManagementSystem:
    """
    Execution Layer (Body).
    Takes Target Weights -> Generates Orders -> Submits to Alpaca.
    Orders are submitted as qty-based (shares), rounded to 2 decimal places.
    """
    def __init__(self):
        self.alpaca = AlpacaAdapter() # Use adapter for API access

    def generate_and_execute_orders(self, target_weights: dict) -> list:
        """
        1. Get Current Portfolio (Positions + Cash)
        2. Fetch Latest Prices for all involved symbols
        3. Calculate Target Qty per Asset
        4. Calculate Diff Qty (Orders), rounded to 2 decimal places
        5. Log order qty as % of target position
        6. Execute Orders (Sell first, then Buy)
        """
        # 0. Cancel Open Orders (Free up Buying Power & Shares)
        self.alpaca.cancel_all_orders()

        # 1. Get Account Info
        acct = self.alpaca.get_account()
        if not acct:
            logger.error("Failed to get account info. Aborting rebalance.")
            return []

        equity = float(acct.get('equity', 0.0))
        buying_power = float(acct.get('buying_power', 0.0))
        logger.info(f"Account Equity: ${equity:,.2f}, Buying Power: ${buying_power:,.2f}")

        # Get Current Positions
        positions = self.alpaca.get_positions()
        current_holdings = {p['Symbol']: float(p['Market Value']) for p in positions}
        current_qtys = {p['Symbol']: float(p['Qty']) for p in positions}
        current_prices = {p['Symbol']: float(p['Current Price']) for p in positions}

        logger.info(f"Current Holdings: {list(current_holdings.keys())}")

        # 2. Determine all involved symbols and fetch latest prices
        all_symbols = set(current_holdings.keys()) | set(target_weights.keys())

        # Fetch prices for symbols not already in positions
        missing_symbols = [s for s in all_symbols if s not in current_prices]
        if missing_symbols:
            fetched_prices = self.alpaca.get_latest_prices(missing_symbols)
            current_prices.update(fetched_prices)

        # 3. Calculate Orders
        orders = []

        for sym in all_symbols:
            current_val = current_holdings.get(sym, 0.0)
            target_pct = target_weights.get(sym, 0.0)
            target_val = equity * target_pct

            diff_val = target_val - current_val

            # Threshold: Ignore trades < $10 to avoid noise/fees
            if abs(diff_val) < 10.0:
                continue

            price = current_prices.get(sym, 0.0)
            if price <= 0:
                logger.warning(f"No valid price for {sym}, skipping order.")
                continue

            # Calculate qty rounded to 2 decimal places
            current_qty = current_qtys.get(sym, 0.0)
            target_qty = round(target_val / price, 2) if target_val > 0 else 0.0

            if diff_val < 0:
                # SELL: if fully closing (target=0), use exact current qty to avoid
                # fractional residual errors (e.g. have 12.3456 but round gives 12.35)
                if target_pct == 0 and current_qty > 0:
                    order_qty = current_qty  # exact close, no rounding
                else:
                    order_qty = round(abs(diff_val) / price, 2)
                    # Cap at actual holding to prevent oversell of fractional shares
                    order_qty = min(order_qty, current_qty)
            else:
                # BUY: straightforward
                order_qty = round(abs(diff_val) / price, 2)

            if order_qty <= 0:
                continue

            # Percentage of target position this order represents
            if target_qty > 0:
                pct_of_target = (order_qty / target_qty) * 100
            elif current_qty > 0:
                pct_of_target = (order_qty / current_qty) * 100
            else:
                pct_of_target = 100.0

            side = 'buy' if diff_val > 0 else 'sell'

            orders.append({
                'symbol': sym,
                'side': side,
                'qty': order_qty,
                'price': price,
                'target_qty': target_qty,
                'pct_of_target': pct_of_target,
                'type': 'market'
            })

        # 4. Execution (Sell First, Then Buy)
        sell_orders = [o for o in orders if o['side'] == 'sell']
        buy_orders = [o for o in orders if o['side'] == 'buy']

        executed_orders = []

        logger.info(f"Generated {len(sell_orders)} SELL orders and {len(buy_orders)} BUY orders.")

        # Execute Sells
        for order in sell_orders:
            try:
                logger.info(
                    f"Submitting SELL: {order['symbol']} "
                    f"qty={order['qty']:.2f} shares @ ~${order['price']:.2f} "
                    f"({order['pct_of_target']:.1f}% of target {order['target_qty']:.2f} shares)"
                )
                self.alpaca.submit_order(
                    symbol=order['symbol'],
                    qty=order['qty'],
                    side='sell',
                    notional=None
                )
                executed_orders.append(order)
            except Exception as e:
                logger.error(f"Failed to execute SELL {order['symbol']}: {e}")

        # Execute Buys
        for order in buy_orders:
            try:
                logger.info(
                    f"Submitting BUY: {order['symbol']} "
                    f"qty={order['qty']:.2f} shares @ ~${order['price']:.2f} "
                    f"({order['pct_of_target']:.1f}% of target {order['target_qty']:.2f} shares)"
                )
                self.alpaca.submit_order(
                    symbol=order['symbol'],
                    qty=order['qty'],
                    side='buy',
                    notional=None
                )
                executed_orders.append(order)
            except Exception as e:
                logger.error(f"Failed to execute BUY {order['symbol']}: {e}")

        return executed_orders
