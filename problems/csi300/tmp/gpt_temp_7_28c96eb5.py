import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Compute Price Momentum Divergence
    # Calculate Short-term Momentum (5-day)
    short_term_momentum = data['close'] / data['close'].shift(5) - 1
    
    # Calculate Medium-term Momentum (20-day)
    medium_term_momentum = data['close'] / data['close'].shift(20) - 1
    
    # Compute Momentum Divergence (absolute difference)
    momentum_divergence = np.abs(short_term_momentum - medium_term_momentum)
    
    # Compute Volume Characteristics
    # Calculate 5-day Volume Average
    volume_5d_avg = data['volume'].rolling(window=5, min_periods=5).mean()
    
    # Calculate 10-day Volume Average
    volume_10d_avg = data['volume'].rolling(window=10, min_periods=10).mean()
    
    # Compute Volume Skewness (10-day)
    def volume_skewness(volume_series):
        if len(volume_series) < 3:
            return np.nan
        return ((volume_series - volume_series.mean()) ** 3).mean() / (volume_series.std() ** 3)
    
    volume_skew = data['volume'].rolling(window=10, min_periods=10).apply(volume_skewness, raw=False)
    
    # Generate Alpha Factor
    # Apply Volume Confirmation
    volume_confirmation = momentum_divergence * (volume_5d_avg / volume_10d_avg)
    
    # Adjust by Volume Skewness
    alpha_factor = volume_confirmation * volume_skew
    
    return alpha_factor
