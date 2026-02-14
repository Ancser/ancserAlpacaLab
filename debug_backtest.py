import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from titan_core.backtest import BacktestEngine

def run_debug_test():
    print("=== Starting Debug Backtest ===")
    
    # 1. Setup
    universe = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'AMD', 'SPY', 'QQQ']
    # Shorten range for quick debug test
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365) # 1 Year
    
    s_str = start_date.strftime('%Y-%m-%d')
    e_str = end_date.strftime('%Y-%m-%d')
    
    print(f"Range: {s_str} to {e_str}")
    print(f"Universe: {universe}")
    
    # 2. Initialize Engine
    try:
        engine = BacktestEngine(initial_capital=100000.0)
    except Exception as e:
        print(f"Failed to init engine: {e}")
        return

    # 3. Run
    factors = ['Momentum', 'Reversion']
    print(f"Active Factors: {factors}")
    
    try:
        results, weights = engine.run(universe, s_str, e_str, factors, leverage=1.0, use_mwu=True)
        
        print("\n=== Results ===")
        if results.empty:
            print("FAILURE: Results DataFrame is empty.")
        else:
            print(f"SUCCESS: Generated {len(results)} days of return.")
            print(f"Final Equity: {results['equity'].iloc[-1]:.2f}")
            print("\nHead of Results:")
            print(results.head())
            
        print("\n=== Weights ===")
        if weights.empty:
            print("WARNING: Weights DataFrame is empty.")
        else:
            print(f"Generated {len(weights)} weight records.")
            print(weights.head())
            
    except Exception as e:
        print(f"CRITICAL ERROR during run: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_debug_test()
