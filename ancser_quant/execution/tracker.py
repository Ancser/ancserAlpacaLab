import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class LiveTracker:
    """
    Tracks and records the daily state of the live portfolio, including equity, 
    daily P&L, target allocations, and the specific factors used to generate them.
    This provides a persistent history to compare against model predictions.
    """
    
    def __init__(self, account_name: str = "Main", log_path=None):
        if not log_path:
            self.log_path = f"logs/live_performance_log_{account_name}.json" if account_name != "Main" else "logs/live_performance_log.json"
        else:
            self.log_path = log_path
            
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, 'w') as f:
                json.dump([], f)

    def record_daily_state(self, date_str, equity, day_pnl, total_pnl_pct, allocations, factors, target_scalar=1.0):
        """
        Record the state of the live portfolio after a rebalance execution.
        """
        try:
            with open(self.log_path, 'r') as f:
                history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            history = []

        # Check if we already recorded today's state (update it if we re-run)
        for entry in history:
            if entry.get('date') == date_str:
                logger.info(f"Tracker: {date_str} already recorded. Updating entry.")
                entry['equity'] = equity
                entry['day_pnl'] = day_pnl
                entry['total_pnl_pct'] = total_pnl_pct
                entry['allocations'] = allocations
                entry['factors'] = factors
                entry['target_scalar'] = target_scalar
                entry['timestamp'] = datetime.now().isoformat()
                self._save(history)
                return

        new_entry = {
            'date': date_str,
            'timestamp': datetime.now().isoformat(),
            'equity': equity,
            'day_pnl': day_pnl,
            'total_pnl_pct': total_pnl_pct,
            'allocations': allocations,
            'factors': factors,
            'target_scalar': target_scalar
        }
        history.append(new_entry)
        self._save(history)
        logger.info(f"Tracker: Recorded new state for {date_str}. Equity: ${equity:,.2f}")

    def _save(self, history):
        with open(self.log_path, 'w') as f:
            json.dump(history, f, indent=4)
            
    def get_history(self):
        try:
            with open(self.log_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
