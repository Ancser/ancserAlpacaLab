import logging
import json
import os
import time
import subprocess
import pandas as pd
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor
from ancser_quant.data.alpaca_adapter import AlpacaAdapter
from ancser_quant.alpha.mwu import MWUEngine
from ancser_quant.utils.accounts import get_configured_accounts

# Logging Setup
logger = logging.getLogger("AncserExecution")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)

DAILY_LOCK_PATH = "logs/daily_trade_lock.json"
PID_FILE_PATH = "logs/ancser_daemon.pid"

def _pid_is_running(pid: int) -> bool:
    """Check if a PID is still alive using Windows tasklist."""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}', '/NH'],
            capture_output=True, text=True
        )
        return str(pid) in result.stdout
    except Exception:
        return False

def _check_single_instance() -> bool:
    """
    Returns True if another instance is already running (abort current).
    Uses a PID file to detect duplicate processes.
    """
    if os.path.exists(PID_FILE_PATH):
        try:
            with open(PID_FILE_PATH, 'r') as f:
                old_pid = int(f.read().strip())
            if _pid_is_running(old_pid):
                logger.warning(f"[SingleInstance] Another instance already running (PID {old_pid}). Exiting.")
                return True
        except Exception:
            pass  # Stale or corrupt PID file — proceed
    # Write current PID
    os.makedirs(os.path.dirname(PID_FILE_PATH), exist_ok=True)
    with open(PID_FILE_PATH, 'w') as f:
        f.write(str(os.getpid()))
    return False

def _remove_pid_file():
    """Remove PID file on clean shutdown."""
    try:
        if os.path.exists(PID_FILE_PATH):
            os.remove(PID_FILE_PATH)
    except Exception:
        pass

def _check_daily_lock(account_name: str) -> bool:
    """
    Returns True if today's trade has already been executed (locked).
    Reads the lock file and compares date with today.
    """
    today = datetime.now().strftime('%Y-%m-%d')
    path = f"logs/daily_trade_lock_{account_name}.json" if account_name != "Main" else "logs/daily_trade_lock.json"
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'r') as f:
            lock = json.load(f)
        return lock.get('last_trade_date') == today
    except Exception:
        return False

def _write_daily_lock(account_name: str):
    """Write today's date to the lock file after a successful trade execution."""
    path = f"logs/daily_trade_lock_{account_name}.json" if account_name != "Main" else "logs/daily_trade_lock.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    today = datetime.now().strftime('%Y-%m-%d')
    lock = {
        'last_trade_date': today,
        'executed_at': datetime.now().isoformat()
    }
    with open(path, 'w') as f:
        json.dump(lock, f, indent=2)
    logger.info(f"Daily trade lock written for Account [{account_name}] on {today}.")

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
        Responsible for initiating the Factor Pipeline and Rebalancing Logic for each Account.
        """
        logger.info("Checking for rebalance opportunity...")
        
        accounts = get_configured_accounts()
        if not accounts:
            logger.error("No valid accounts configured. Aborting.")
            return

        for account_name in accounts:
            logger.info(f"--- Processing Account: {account_name} ---")
            
            # Daily trade lock: only allow one execution per calendar day
            if _check_daily_lock(account_name):
                today = datetime.now().strftime('%Y-%m-%d')
                if force:
                    logger.warning(f"[DailyLock] Account {account_name} trade already executed today ({today}), but --force is set. Overriding lock.")
                else:
                    logger.info(f"[DailyLock] Account {account_name} trade already executed today ({today}). Skipping rebalance.")
                    continue

            # 1. Load Live Strategy Config
            config_path = f"config/live_strategy_{account_name}.json" if account_name != "Main" else "config/live_strategy.json"
            if not os.path.exists(config_path):
                # Always fallback to Main if specific config not found
                fallback_path = "config/live_strategy.json"
                if os.path.exists(fallback_path):
                    logger.info(f"Using default strategy config for {account_name}.")
                    config_path = fallback_path
                else:
                    logger.warning(f"No strategy config found for {account_name}. Skipping.")
                    continue

            try:
                with open(config_path, 'r') as f:
                    strategy_config = json.load(f)
                
                logger.info(f"Loaded Strategy Config for {account_name}")
                universe = strategy_config.get('universe', [])
                factors = strategy_config.get('active_factors', [])
                
                if not universe or not factors:
                    logger.warning(f"Universe or Factors empty for {account_name}. Skipping.")
                    continue

                # 3. Live Strategy Calculation 
                # (Volatility fetched optimally inside the strategy method now anyway)
                from ancser_quant.execution.strategy import LiveStrategy
                strat = LiveStrategy(account_name=account_name)
                
                # Use the loaded config
                res = strat.calculate_targets(strategy_config)
                
                if "error" in res:
                    logger.error(f"Strategy Calculation Error for {account_name}: {res['error']}")
                    continue

                # Extract Results
                target_weights = res.get('allocations', {})
                vol_metrics = res.get('vol_metrics', {})
                final_scalar = vol_metrics.get('final_scalar', 1.0)
                
                logger.info(f"Rebalance Logic Executed for {account_name}. Target Exposure Scalar: {final_scalar:.2f}x")
                
                if not target_weights:
                    logger.warning(f"No target weights generated for {account_name}. Portfolio may be empty.")
                else:
                    logger.info(f"Generated Targets for {len(target_weights)} assets for {account_name}.")
                    
                # 4. Order Management System (OMS)
                from ancser_quant.execution.oms import OrderManagementSystem
                oms = OrderManagementSystem(account_name=account_name)
                
                logger.info(f"Executing Rebalance Orders for {account_name}...")
                oms.generate_and_execute_orders(target_weights, strategy_config=strategy_config)

                # --- Inject Tracker Here ---
                try:
                    from ancser_quant.execution.tracker import LiveTracker
                    tracker = LiveTracker(account_name=account_name)
                    
                    # Fetch latest account equity and today's P&L from Alpaca to log
                    acc_adapter = AlpacaAdapter(account_name=account_name)
                    acc = acc_adapter.get_account()
                    equity = float(acc.get('equity', 0.0))
                    
                    # Use daily P&L. If history is not available, just default to 0
                    portfolio_history = acc_adapter.get_portfolio_history(period="1D", timeframe="1D")
                    pl_vals = portfolio_history.get('profit_loss', [0])
                    pl_pcts = portfolio_history.get('profit_loss_pct', [0])
                    
                    day_pnl = pl_vals[-1] if pl_vals else 0.0
                    total_pnl_pct = pl_pcts[-1] if pl_pcts else 0.0
                    
                    today_str = datetime.now().strftime('%Y-%m-%d')
                    
                    tracker.record_daily_state(
                        date_str=today_str,
                        equity=equity,
                        day_pnl=day_pnl,
                        total_pnl_pct=total_pnl_pct,
                        allocations=target_weights,
                        factors=factors,
                        target_scalar=final_scalar
                    )
                except Exception as e_track:
                    logger.error(f"Failed to record tracker state for {account_name}: {e_track}")
                    import traceback
                    logger.error(traceback.format_exc())
                # --- End Tracker ---

                # Write daily lock so restarts won't re-execute today
                _write_daily_lock(account_name)

                logger.info(f"Rebalance Cycle Completed for {account_name}.")
                
            except Exception as e:
                logger.error(f"Rebalance Failed for {account_name}: {e}")
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
        
        # Rebalance: Hourly during market hours, timezone locked to US/Eastern regardless of server location
        self.scheduler.add_job(self.rebalance_check, 'cron', day_of_week='mon-fri', hour='9-16', minute=0, id='rebalance', timezone='America/New_York')
        
        self.scheduler.start()
        self.running = True

        # Run once immediately on startup so we don't miss the current window
        logger.info("Running initial rebalance check on startup...")
        self.rebalance_check()

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
        _remove_pid_file()

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
    """Run the rebalance logic once and exit. Always skips market-open check so pre-market orders work."""
    loop = TitanEventLoop()
    logger.info("--- Starting Daily Batch Execution ---")

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
        # Server Mode — guard against duplicate instances
        if _check_single_instance():
            exit(0)
        loop = TitanEventLoop()
        loop.start()
