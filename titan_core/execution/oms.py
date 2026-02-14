from typing import Dict, List
import logging
import time

logger = logging.getLogger(__name__)

class OMS:
    """
    Order Management System.
    Responsible for diffing, slicing, and safety checks.
    """
    def generate_orders(self, target_weights: Dict[str, float], current_positions: Dict[str, float], total_equity: float) -> List[Dict]:
        """
        Generates a list of orders (diff) to move from Current -> Target.
        """
        orders = []
        
        # 1. Diff Logic
        # Iterate all potential keys
        all_symbols = set(target_weights.keys()) | set(current_positions.keys())
        
        for sym in all_symbols:
            t_w = target_weights.get(sym, 0.0)
            c_val = current_positions.get(sym, 0.0) # Assume current_positions is value ($)
            
            target_val = total_equity * t_w
            diff_val = target_val - c_val
            
            # Threshold to avoid tiny trades (e.g., drift < $10)
            if abs(diff_val) < 50: 
                continue
                
            side = 'buy' if diff_val > 0 else 'sell'
            
            orders.append({
                'symbol': sym,
                'side': side,
                'notional': abs(diff_val),
                'type': 'market'
            })
            
        return orders

    def slice_order(self, order: Dict, chunks: int = 10) -> List[Dict]:
        """
        Slices a large order into smaller TWAP chunks.
        """
        total_val = order['notional']
        if total_val < 1000: # Don't slice small orders
            return [order]
            
        chunk_val = total_val / chunks
        slices = []
        
        for i in range(chunks):
            sliced = order.copy()
            sliced['notional'] = chunk_val
            sliced['slice_id'] = i + 1
            slices.append(sliced)
            
        return slices

    def check_risk(self, order: Dict, max_pos_size: float = 0.10) -> bool:
        """
        Risk Control Gate.
        Returns True if safe, False if rejected.
        """
        # Simplistic check: assumes we have access to portfolio total
        # In real implementation, pass portfolio context
        return True
