import json
import logging
import os
import pandas as pd
from datetime import datetime
from ancser_quant.data.alpaca_adapter import AlpacaAdapter

REBALANCE_SNAPSHOT_PATH = "logs/last_rebalance.json"
REBALANCE_HISTORY_PATH = "logs/rebalance_history.json"

logger = logging.getLogger("AncserExecution")

class OrderManagementSystem:
    """
    Execution Layer (Body).
    Takes Target Weights -> Generates Orders -> Submits to Alpaca.
    Orders are submitted as qty-based (shares), rounded to 2 decimal places.
    """
    def __init__(self):
        self.alpaca = AlpacaAdapter() # Use adapter for API access

    def generate_and_execute_orders(self, target_weights: dict, strategy_config: dict = None) -> list:
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
            current_qty = current_qtys.get(sym, 0.0)
            target_pct = target_weights.get(sym, 0.0)

            price = current_prices.get(sym, 0.0)
            if price <= 0:
                logger.warning(f"No valid price for {sym}, skipping order.")
                continue

            if target_pct == 0.0:
                # Full exit: sell entire actual qty directly — bypass $10 threshold
                if current_qty <= 0:
                    continue
                order_qty = current_qty
                side = 'sell'
                target_qty = 0.0
                pct_of_target = 100.0
            else:
                # Partial rebalance: work in qty-space to avoid market-value drift
                target_qty = round((equity * target_pct) / price, 2)
                diff_qty = round(target_qty - current_qty, 2)

                # Threshold: Ignore trades worth < $10 to avoid noise/fees
                if abs(diff_qty) * price < 10.0:
                    continue
                if diff_qty == 0:
                    continue

                order_qty = abs(diff_qty)
                side = 'buy' if diff_qty > 0 else 'sell'

                if target_qty > 0:
                    pct_of_target = (order_qty / target_qty) * 100
                elif current_qty > 0:
                    pct_of_target = (order_qty / current_qty) * 100
                else:
                    pct_of_target = 100.0

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

        # Save rebalance snapshot: prices at execution time for dashboard P&L tracking
        try:
            snapshot_positions = {}
            for sym in all_symbols:
                qty = current_qtys.get(sym, 0.0)
                target_pct = target_weights.get(sym, 0.0)
                # Use target qty if we just bought into it, else current qty
                final_qty = round((equity * target_pct) / current_prices[sym], 2) if target_pct > 0 and sym in current_prices and current_prices[sym] > 0 else qty
                if final_qty > 0 and sym in current_prices:
                    snapshot_positions[sym] = {
                        "entry_price": round(current_prices[sym], 4),
                        "qty": final_qty,
                        "value": round(current_prices[sym] * final_qty, 2)
                    }
            snapshot = {
                "rebalance_date": datetime.now().strftime('%Y-%m-%d'),
                "rebalance_time": datetime.now().isoformat(),
                "positions": snapshot_positions
            }
            # Save strategy config for backtest-vs-live comparison
            if strategy_config:
                snapshot["strategy_config"] = {
                    "active_factors": strategy_config.get("active_factors", []),
                    "use_mwu": strategy_config.get("use_mwu", False),
                    "leverage": strategy_config.get("leverage", 1.0),
                    "use_vol_target": strategy_config.get("use_vol_target", False),
                    "vol_target": strategy_config.get("vol_target", 0.20),
                    "strategy_mode": strategy_config.get("strategy_mode", "long_only"),
                }
            os.makedirs(os.path.dirname(REBALANCE_SNAPSHOT_PATH), exist_ok=True)
            # Write latest snapshot
            with open(REBALANCE_SNAPSHOT_PATH, 'w') as f:
                json.dump(snapshot, f, indent=2)
            # Append to history (load existing, append, save)
            history = []
            if os.path.exists(REBALANCE_HISTORY_PATH):
                try:
                    with open(REBALANCE_HISTORY_PATH, 'r') as f:
                        history = json.load(f)
                except Exception:
                    history = []
            # Avoid duplicate entries for the same date (overwrite same-day)
            history = [h for h in history if h.get('rebalance_date') != snapshot['rebalance_date']]
            history.append(snapshot)
            with open(REBALANCE_HISTORY_PATH, 'w') as f:
                json.dump(history, f, indent=2)
            logger.info(f"Rebalance snapshot saved. History now has {len(history)} entries.")
        except Exception as snap_e:
            logger.warning(f"Failed to write rebalance snapshot: {snap_e}")

        return executed_orders
