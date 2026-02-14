from typing import Dict, Optional
import pandas as pd
from .pipeline import FactorPipeline

class Strategy:
    """
    Defines how to go from Data -> Target Weights.
    Contains a FactorPipeline and a Weighting Scheme.
    """
    def __init__(self, 
                 pipeline: FactorPipeline, 
                 top_n: int = 10, 
                 universe_mode: str = 'fixed'):
        self.pipeline = pipeline
        self.top_n = top_n
        self.universe_mode = universe_mode

    def generate_target_weights(
        self, 
        close: pd.DataFrame, 
        volume: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Run pipeline and return target weights for the *next* period.
        """
        # 1. Run Pipeline
        scores = self.pipeline.run(close, volume)
        
        # 2. Get latest scores
        if scores.empty:
            return {}
            
        latest_scores = scores.iloc[-1].dropna().sort_values(ascending=False)
        
        # 3. Select Top N
        top_picks = latest_scores.head(self.top_n).index.tolist()
        
        if not top_picks:
            return {}

        # 4. Equal Weighting (for now)
        weight = 1.0 / len(top_picks)
        return {ticker: weight for ticker in top_picks}
