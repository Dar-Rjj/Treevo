import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume Momentum with Decay
    # Calculate 5-day volume momentum
    volume_momentum = df['volume'] / df['volume'].shift(5) - 1
    
    # Apply exponential decay over 20-day window
    decay_weights = np.exp(-np.arange(20) / 10)  # 10-day half-life
    decay_weights = decay_weights / decay_weights.sum()
    decayed_volume_momentum = volume_momentum.rolling(window=20, min_periods=1).apply(
        lambda x: np.sum(x * decay_weights[:len(x)]), raw=False
    )
    
    # Price Range Efficiency
    # Compute True Range (High, Low, Previous Close)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate Close-to-Close return divided by True Range
    close_return = df['close'].pct_change()
    range_efficiency = close_return / true_range
    
    # Signal Combination
    # Multiply decayed volume momentum by range efficiency
    factor = decayed_volume_momentum * range_efficiency
    
    return factor
