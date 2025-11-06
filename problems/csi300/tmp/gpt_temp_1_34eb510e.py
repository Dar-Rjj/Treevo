import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Momentum
    high_low_range = df['high'] - df['low']
    close_open_position = df['close'] - df['open']
    intraday_momentum = close_open_position / high_low_range.replace(0, np.nan)
    
    # Calculate Volume Trend
    volume_momentum = df['volume'] / df['volume'].shift(1).replace(0, np.nan) - 1
    volume_acceleration = volume_momentum - volume_momentum.shift(1)
    
    # Combine Momentum and Volume Signals
    combined_signal = intraday_momentum * volume_acceleration
    directional_signal = np.sign(combined_signal)
    
    # Generate Final Alpha Factor with Exponential Decay Weighting
    decay_factor = 0.9
    weights = [decay_factor ** i for i in range(5)]  # 5-day window
    weights = np.array(weights) / sum(weights)  # Normalize weights
    
    # Create rolling window and apply weights
    alpha_factor = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 4:  # Need at least 5 days of data
            window_data = directional_signal.iloc[i-4:i+1]
            weighted_sum = (window_data * weights).sum()
            alpha_factor.iloc[i] = weighted_sum
        else:
            alpha_factor.iloc[i] = np.nan
    
    return alpha_factor
