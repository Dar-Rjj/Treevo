import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(data):
    """
    Momentum-Adjusted Volume Acceleration factor
    Combines price momentum with volume acceleration signals
    """
    df = data.copy()
    
    # Compute Price Momentum
    # Short-term momentum (5-day)
    short_momentum = df['close'] / df['close'].shift(5) - 1
    
    # Medium-term momentum (20-day)
    medium_momentum = df['close'] / df['close'].shift(20) - 1
    
    # Derive Volume Acceleration
    def calculate_volume_slope(volume_series, window):
        """Calculate linear regression slope of volume over given window"""
        slopes = pd.Series(index=volume_series.index, dtype=float)
        
        for i in range(window-1, len(volume_series)):
            if i >= window-1:
                y = volume_series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                
                # Handle cases with zero variance
                if len(np.unique(y)) > 1:
                    slope, _, _, _, _ = stats.linregress(x, y)
                    slopes.iloc[i] = slope
                else:
                    slopes.iloc[i] = 0
        
        return slopes
    
    # Recent volume acceleration (5-day slope)
    recent_volume_accel = calculate_volume_slope(df['volume'], 5)
    
    # Historical volume acceleration (20-day slope)
    historical_volume_accel = calculate_volume_slope(df['volume'], 20)
    
    # Normalize volume accelerations by their standard deviations for comparability
    recent_volume_accel_norm = recent_volume_accel / recent_volume_accel.rolling(60).std()
    historical_volume_accel_norm = historical_volume_accel / historical_volume_accel.rolling(60).std()
    
    # Combine Momentum and Volume Signals
    # Multiply momentum components by their respective volume accelerations
    short_momentum_weighted = short_momentum * recent_volume_accel_norm
    medium_momentum_weighted = medium_momentum * historical_volume_accel_norm
    
    # Sum weighted momentum components
    factor = short_momentum_weighted + medium_momentum_weighted
    
    return factor
