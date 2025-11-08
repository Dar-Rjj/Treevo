import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Momentum Decay with Volume Confirmation factor
    
    Combines price momentum with exponential decay and volume confirmation
    to create a robust alpha factor.
    """
    close = data['close']
    volume = data['volume']
    
    # Calculate Price Momentum (5-day momentum)
    momentum = close / close.shift(5)
    
    # Apply Exponential Decay to Momentum
    decay_factor = 0.9
    decayed_momentum = momentum.copy()
    
    # Apply exponential decay weighting
    for i in range(len(momentum)):
        if i >= 5:  # Only apply decay after we have enough data
            weights = np.array([decay_factor**j for j in range(6)])[::-1]
            weights = weights / weights.sum()  # Normalize weights
            window = momentum.iloc[max(0, i-5):i+1]
            if len(window) == 6:
                decayed_momentum.iloc[i] = np.sum(window.values * weights)
    
    # Confirm with Volume Trend
    volume_ma = volume.rolling(window=5).mean()
    volume_ratio = volume / volume_ma
    
    # Combine decayed momentum with volume confirmation
    factor = decayed_momentum * volume_ratio
    
    return factor
