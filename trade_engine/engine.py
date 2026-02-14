import time
import logging
from datetime import datetime
from typing import List

from quant_core.data_loader import DataLoader
from quant_core.strategy import Strategy
from trade_engine.executor import Executor
from trade_engine.portfolio import PortfolioManager

logger = logging.getLogger(__name__)

class TradingEngine:
    """
    The Body. Connects Layer 1 (Brain) with Layer 2 (Execution).
    """
    def __init__(
        self,
        data_loader: DataLoader,
        executor: Executor,
        portfolio_manager: PortfolioManager
    ):
        self.data_loader = data_loader
        self.executor = executor
        self.portfolio = portfolio_manager
        self.running = False

    def run_once(self, strategy: Strategy, universe: List[str]):
        """
        Single iteration of the trading loop.
        """
        logger.info("Starting trading cycle...")

        # 1. Fetch Data
        end_date = datetime.now().strftime('%Y-%m-%d')
        # Simple lookback logic, strategy might need more
        start_date = "2023-01-01" # TODO: make dynamic based on config
        
        close, volume = self.data_loader.get_data(universe, start_date, end_date)
        
        if close.empty:
            logger.error("No data fetched. Aborting cycle.")
            return

        # 2. Generate Target Weights (Layer 1)
        target_weights = strategy.generate_target_weights(close, volume)
        logger.info(f"Target Weights: {target_weights}")

        # 3. Get Current State (Layer 2)
        current_positions = self.executor.get_positions()
        account_info = self.executor.get_account_info()
        equity = account_info.get('equity', 0.0)

        # 4. Generate Orders
        # We need current prices to calculate diffs. Use latest close.
        current_prices = close.iloc[-1].to_dict()
        
        orders = self.portfolio.generate_orders(
            target_weights, 
            current_positions, 
            equity, 
            current_prices
        )
        
        # 5. Execute
        self.executor.submit_orders(orders)
        logger.info("Cycle complete.")

    def run_loop(self, strategy: Strategy, universe: List[str], interval_seconds: int = 60):
        """
        Main Event Loop.
        """
        self.running = True
        logger.info(f"Starting main loop with interval {interval_seconds}s")
        
        while self.running:
            try:
                self.run_once(strategy, universe)
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
            
            time.sleep(interval_seconds)
