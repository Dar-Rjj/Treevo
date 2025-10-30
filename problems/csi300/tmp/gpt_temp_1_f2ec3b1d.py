import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using Volume-Weighted Price Acceleration
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate Price Momentum components
    # Short-term momentum (1-day)
    short_term_momentum = (data['close'] / data['close'].shift(1) - 1)
    
    # Medium-term momentum (5-day)
    medium_term_momentum = (data['close'] / data['close'].shift(5) - 1)
    
    # Derive Acceleration Signal
    acceleration = short_term_momentum - medium_term_momentum
    
    # Normalize by medium-term absolute value (avoid division by zero)
    denominator = np.abs(medium_term_momentum)
    denominator = denominator.replace(0, np.nan)  # Avoid division by zero
    normalized_acceleration = acceleration / denominator
    
    # Fill NaN values with 0 where denominator was 0
    normalized_acceleration = normalized_acceleration.fillna(0)
    
    # Calculate Volume-Based Confidence
    # Volume trend - current volume vs 5-day average
    volume_5d_avg = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_ratio = data['volume'] / volume_5d_avg
    
    # Weight Acceleration by Volume Confidence
    volume_weighted_acceleration = normalized_acceleration * volume_ratio
    
    # Additional enhancement: Apply rolling normalization to final factor
    factor = volume_weighted_acceleration.rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    # Fill any remaining NaN values
    factor = factor.fillna(0)
    
    return factor
