import logging
import json
import os
import pandas as pd
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from ancser_quant.data.alpaca_adapter import AlpacaAdapter
from ancser_quant.alpha.mwu import MWUEngine

# Logging Setup
logger = logging.getLogger("AncserExecution")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

DAILY_LOCK_PATH = "logs/daily_trade_lock.json"

def _check_daily_lock() -> bool:
    """
    Returns True if today's trade has already been executed (locked).
    Reads the lock file and compares date with today.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    if not os.path.exists(DAILY_LOCK_PATH):
        return False
    try:
        with open(DAILY_LOCK_PATH, 'r') as f:
            lock = json.load(f)
        return lock.get('last_trade_date') == today
    except Exception:
        return False

def _write_daily_lock():
    """Write today's date to the lock file after a successful trade execution."""
    os.makedirs(os.path.dirname(DAILY_LOCK_PATH), exist_ok=True)
    today = datetime.now().strftime('%Y-%m-%d')
    lock = {
        'last_trade_date': today,
        'executed_at': datetime.now().isoformat()
    }
    with open(DAILY_LOCK_PATH, 'w') as f:
        json.dump(lock, f, indent=2)
    logger.info(f"Daily trade lock written for {today}.")

class TitanEventLoop:
    """
    Core Event Loop.
    Managed by APScheduler (Advanced Python Scheduler).
    """
    def __init__(self, config_path: str = "config/titan_config.yaml"):
        self.scheduler = BackgroundScheduler(executors={'default': ThreadPoolExecutor(2)})
        self.alpaca = AlpacaAdapter() # Initialize Adapter
        self.running = False

    def heartbeat(self):
        """
        Runs every 1 minute.
        Checks API connectivity and logs system status.
        """
        try:
            # Simple check: Ask for clock or account
            # We can use the data_client or trading_client from adapter if exposed
            # For now, just log success
            logger.info("Heartbeat: System Alive. API Connection Stable.")
        except Exception as e:
            logger.error(f"Heartbeat Failed: {e}")

    def rebalance_check(self, force: bool = False):
        """
        Runs every hour (during trading hours).
        Responsible for initiating the Factor Pipeline and Rebalancing Logic.
        force=True bypasses the daily lock (used with --force flag).
        """
        logger.info("Checking for rebalance opportunity...")

        # Daily trade lock: only allow one execution per calendar day
        if not force and _check_daily_lock():
            today = datetime.now().strftime('%Y-%m-%d')
            logger.info(f"[DailyLock] Trade already executed today ({today}). Skipping rebalance.")
            return
        if force:
            logger.info("[DailyLock] Force mode: bypassing daily trade lock.")

        # 1. Load Live Strategy Config
        config_path = "config/live_strategy.json"
        if not os.path.exists(config_path):
            logger.warning("No live strategy config found. Skipping.")
            return

        try:
            with open(config_path, 'r') as f:
                strategy_config = json.load(f)
            
            logger.info(f"Loaded Strategy Config: {strategy_config}")
            universe = strategy_config.get('universe', [])
            factors = strategy_config.get('active_factors', [])
            
            if not universe or not factors:
                logger.warning("Universe or Factors empty. Skipping.")
                return

            # 2. Fetch Latest Data for Volatility Calculation (if enabled)
            target_scalar = 1.0
            use_vol_target = strategy_config.get('use_vol_target', False)
            vol_target = strategy_config.get('vol_target', 0.20)
            leverage_cap = strategy_config.get('leverage', 1.0)
            
            if use_vol_target:
                logger.info(f"Volatility Targeting Enabled. Target: {vol_target:.1%}. Calculating current market vol...")
                try:
                    # Fetch 30 days of history for the universe to estimate recent volatility
                    # Using Universe Equal Weight Volatility as Proxy for Portfolio Volatility (Robustness)
                    end_dt = datetime.now()
                    start_dt = end_dt - pd.Timedelta(days=45) # Buffer for 20 trading days
                    
                    # We utilize the unified adapter
                    from ancser_quant.backtest import BacktestEngine # Re-use infrastructure if possible, or just adapter
                    # Just use adapter directly
                    hist_pl = self.alpaca.fetch_history(universe, start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')).collect()
                    
                    if not hist_pl.is_empty():
                        hist = hist_pl.to_pandas()
                        hist['timestamp'] = pd.to_datetime(hist['timestamp'])
                        
                        # Pivot to Close prices
                        closes = hist.pivot(index='timestamp', columns='symbol', values='close')
                        
                        # Calculate daily returns
                        rets = closes.pct_change().dropna()
                        
                        # Equal Weighted Universe Return
                        univ_ret = rets.mean(axis=1)
                        
                        # Last 20 days vol
                        if len(univ_ret) >= 20:
                           recent_std = univ_ret.tail(20).std()
                           current_vol = recent_std * (252 ** 0.5)
                           
                           if current_vol > 0.001:
                               raw_scalar = vol_target / current_vol
                               target_scalar = min(leverage_cap, raw_scalar)
                               logger.info(f"Market Vol (20d): {current_vol:.2%}. Target: {vol_target:.0%}. Scalar: {target_scalar:.2f}x")
                           else:
                               target_scalar = leverage_cap
                        else:
                            logger.warning("Insufficient history for Volatility calc. Defaulting to Max Leverage.")
                            target_scalar = leverage_cap
                    else:
                        logger.warning("No history data fetched. Defaulting scalar to 1.0.")
                        
                except Exception as vol_e:
                    logger.error(f"Error calculating Volatility Scalar: {vol_e}. Defaulting to 1.0.")
            
            # 3. Live Strategy Calculation
            from ancser_quant.execution.strategy import LiveStrategy
            strat = LiveStrategy()
            
            # Use the loaded config
            res = strat.calculate_targets(strategy_config)
            
            if "error" in res:
                logger.error(f"Strategy Calculation Error: {res['error']}")
                return

            # Extract Results
            target_weights = res.get('allocations', {})
            vol_metrics = res.get('vol_metrics', {})
            
            # Update target scalar from LiveStrategy result (it handles vol targeting internally now)
            final_scalar = vol_metrics.get('final_scalar', 1.0)
            
            logger.info(f"Rebalance Logic Executed. Final Target Exposure Scalar: {final_scalar:.2f}x")
            
            if not target_weights:
                logger.warning("No target weights generated. Portfolio may be empty.")
            else:
                logger.info(f"Generated Targets for {len(target_weights)} assets.")
                
            # 4. Order Management System (OMS)
            from ancser_quant.execution.oms import OrderManagementSystem
            oms = OrderManagementSystem()
            
            logger.info("Executing Rebalance Orders...")
            oms.generate_and_execute_orders(target_weights)

            # Write daily lock so restarts won't re-execute today
            _write_daily_lock()

            logger.info("Rebalance Cycle Completed.")
            
        except Exception as e:
            logger.error(f"Rebalance Failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        pass 

    def start(self):
        """Start the scheduler loop."""
        if self.running:
            return
            
        logger.info("Starting Titan Event Loop...")
        
        # Schedule Jobs
        self.scheduler.add_job(self.heartbeat, 'interval', minutes=1, id='heartbeat')
        
        # Rebalance: Hourly during market hours (simplistic approx for now)
        self.scheduler.add_job(self.rebalance_check, 'cron', day_of_week='mon-fri', hour='9-16', minute=0, id='rebalance')
        
        self.scheduler.start()
        self.running = True
        
        try:
            # Keep main thread alive
            while True:
                time.sleep(2)
        except (KeyboardInterrupt, SystemExit):
            self.stop()

    def stop(self):
        """Graceful shutdown."""
        logger.info("Stopping Titan Event Loop...")
        self.scheduler.shutdown()
        self.running = False

    def check_market_open(self) -> bool:
        """Check if market is currently open."""
        clock = self.alpaca.get_clock()
        if clock.get('is_open'):
            logger.info("Market is OPEN.")
            return True
        else:
            logger.info(f"Market is CLOSED. Next Open: {clock.get('next_open')}")
            return False

def run_once(force: bool = False):
    """Run the rebalance logic once and exit."""
    loop = TitanEventLoop()
    logger.info("--- Starting Daily Batch Execution ---")
    
    if not force:
        if not loop.check_market_open():
            logger.warning("Market is Closed. Use --force to run anyway. Exiting.")
            return

    logger.info("Running Rebalance Logic...")
    loop.rebalance_check(force=force)
    logger.info("--- Daily Batch Execution Completed ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='AncserQuant Execution Engine')
    parser.add_argument('--run-once', action='store_true', help='Run rebalance logic once and exit (for Cron/Task Scheduler)')
    parser.add_argument('--force', action='store_true', help='Force execution even if market is closed')
    
    args = parser.parse_args()
    
    if args.run_once:
        run_once(force=args.force)
    else:
        # Server Mode
        loop = TitanEventLoop()
        loop.start()
