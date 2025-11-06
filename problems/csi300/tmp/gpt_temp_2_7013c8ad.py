import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, window=20, decay_factor=0.9):
    """
    Rolling High-Low Momentum Divergence factor
    Compares price momentum against high-low range with exponential decay weighting
    """
    # Calculate rolling high-low range
    rolling_high = df['high'].rolling(window=window).max()
    rolling_low = df['low'].rolling(window=window).min()
    high_low_range = rolling_high - rolling_low
    
    # Calculate price momentum using rate of change
    momentum = (df['close'] - df['close'].shift(window)) / df['close'].shift(window)
    
    # Compare range and momentum (avoid division by zero)
    range_momentum_ratio = momentum / high_low_range.replace(0, np.nan)
    
    # Apply exponential decay weighting
    weights = np.array([decay_factor ** i for i in range(window)])[::-1]
    weights = weights / weights.sum()  # Normalize weights
    
    # Create weighted factor using rolling window
    factor_values = pd.Series(index=df.index, dtype=float)
    
    for i in range(window, len(df)):
        if i >= window:
            window_data = range_momentum_ratio.iloc[i-window+1:i+1]
            if not window_data.isna().all():
                valid_indices = ~window_data.isna()
                valid_weights = weights[-len(window_data):][valid_indices]
                valid_weights = valid_weights / valid_weights.sum()  # Renormalize for valid data
                factor_values.iloc[i] = np.sum(window_data[valid_indices] * valid_weights)
    
    return factor_values
