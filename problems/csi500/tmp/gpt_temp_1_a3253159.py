import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Recent Price-Volume Momentum Factor
    Combines short-term price momentum with volume acceleration, adjusted for volatility
    """
    # Calculate price momentum components
    momentum_1d = df['close'] / df['close'].shift(1) - 1
    momentum_2d = df['close'] / df['close'].shift(2) - 1
    momentum_3d = df['close'] / df['close'].shift(3) - 1
    
    # Blend momentum signals
    blended_momentum = (momentum_1d + momentum_2d) / 2 + 0.3 * momentum_3d
    
    # Calculate volume acceleration
    volume_acceleration = df['volume'] / df['volume'].shift(1) - 1
    
    # Calculate 5-day price range for volatility adjustment
    high_5d = df['high'].rolling(window=5).max()
    low_5d = df['low'].rolling(window=5).min()
    price_range = high_5d - low_5d
    price_range_normalized = price_range / df['close'].shift(1)
    
    # Adjust volume acceleration by volatility
    volatility_adjusted_volume = volume_acceleration / (price_range_normalized + 1e-8)
    
    # Combine price and volume components
    factor = blended_momentum * volatility_adjusted_volume
    
    # Cross-sectional rank
    def cross_sectional_rank(series):
        return series.groupby(series.index).rank(pct=True)
    
    ranked_factor = factor.groupby(factor.index).apply(cross_sectional_rank)
    
    return ranked_factor
