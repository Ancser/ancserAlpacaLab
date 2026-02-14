from typing import List, Dict, Callable, Any, Optional
import pandas as pd
import numpy as np
from .factors import zscore

class FactorPipeline:
    """
    Processes data through a sequence of factors to generate a combined signal.
    """
    def __init__(self):
        self.steps: List[Dict[str, Any]] = []

    def add(self, 
            name: str, 
            func: Callable, 
            params: Dict[str, Any] = {}, 
            weight: float = 1.0,
            normalize: bool = True):
        """
        Add a factor step to the pipeline.
        
        Args:
            name: Unique identifier for the factor.
            func: Function that takes (close, volume, ...) and returns a DataFrame.
            params: Dictionary of parameters to pass to the function.
            weight: Weight of this factor in the final composite score.
            normalize: Whether to z-score normalize this factor before combining.
        """
        self.steps.append({
            'name': name,
            'func': func,
            'params': params,
            'weight': weight,
            'normalize': normalize
        })

    def run(self, close: pd.DataFrame, volume: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Execute the pipeline and return the composite score.
        
        Returns:
            pd.DataFrame: Composite score (Date x Symbol). Higher is better.
        """
        final_score = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        total_weight = 0.0

        for step in self.steps:
            name = step['name']
            func = step['func']
            params = step['params']
            weight = step['weight']
            norm = step['normalize']

            if weight == 0:
                continue

            # Check function signature to pass correct arguments
            # Ideally, detailed inspection, but simplistic approach here:
            # If function has 'volume' arg, pass volume.
            import inspect
            sig = inspect.signature(func)
            
            call_args = {'close': close}
            if 'volume' in sig.parameters and volume is not None:
                call_args['volume'] = volume
            
            # Merge with params
            call_args.update(params)

            # Calculate factor
            try:
                factor_val = func(**call_args)
            except Exception as e:
                print(f"Error calculating factor {name}: {e}")
                continue

            # Normalize if requested
            if norm:
                factor_val = zscore(factor_val)

            # Add to composite
            # Handle NaNs: fill with 0 (neutral) for summation
            final_score = final_score.add(factor_val.fillna(0) * weight)
            total_weight += abs(weight)

        return final_score
