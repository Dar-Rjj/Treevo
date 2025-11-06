import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Compute 5-day Price Momentum
    momentum_5d = data['close'] / data['close'].shift(5) - 1
    
    # Compute 10-day Price Acceleration (momentum of momentum)
    momentum_acceleration = momentum_5d / momentum_5d.shift(5) - 1
    
    # Compute Directional Momentum (product of momentum and acceleration)
    directional_momentum = momentum_5d * momentum_acceleration
    
    # Compute Short-term Volume Trend (3-day volume change)
    volume_trend = data['volume'] / data['volume'].shift(3) - 1
    
    # Compute Medium-term Volume Baseline (median volume from t-10 to t-6)
    def rolling_median_volume(window):
        # Extract the last 5 days of the window (positions 5-9 in 10-day window)
        return window.iloc[5:10].median()
    
    volume_baseline = data['volume'].rolling(window=10, min_periods=6).apply(
        rolling_median_volume, raw=False
    )
    
    # Calculate Volume Divergence Ratio
    volume_divergence = volume_trend / volume_baseline - 1
    
    # Combine components: Momentum-Weighted Volume Divergence
    alpha_factor = directional_momentum * volume_divergence
    
    return alpha_factor
