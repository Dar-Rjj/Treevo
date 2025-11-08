import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Multi-Timeframe Momentum with Volume Confirmation alpha factor
    
    Parameters:
    data: pandas DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    pandas Series with the alpha factor values
    """
    
    # Momentum Component
    close = data['close']
    
    # Short-term momentum (5-day)
    mom_short = (close - close.shift(5)) / close.shift(5)
    
    # Medium-term momentum (13-day)
    mom_medium = (close - close.shift(13)) / close.shift(13)
    
    # Long-term momentum (34-day)
    mom_long = (close - close.shift(34)) / close.shift(34)
    
    # Volume Confirmation Component
    volume = data['volume']
    
    # Calculate 34-day volume percentile rank
    volume_percentile = volume.rolling(window=34, min_periods=1).apply(
        lambda x: (x[-1] > x[:-1]).sum() / (len(x) - 1) if len(x) > 1 else 0.5
    )
    
    # Volume score transformation (bounded between 0.2 and 0.8)
    volume_score = np.minimum(np.maximum(volume_percentile / 100, 0.2), 0.8)
    
    # Multiplicative Combination
    # Multiply all three momentum returns
    momentum_product = mom_short * mom_medium * mom_long
    
    # Cube root normalization with sign preservation
    normalized_momentum = np.sign(momentum_product) * np.abs(momentum_product) ** (1/3)
    
    # Final Alpha Factor: Volume-Weighted Momentum
    alpha = normalized_momentum * volume_score
    
    return alpha
