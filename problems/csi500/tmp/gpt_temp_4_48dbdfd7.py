import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Price Change
    df['price_change'] = df['close'].diff()
    
    # Compute Volume Adjusted Return
    df['volume_adjusted_return'] = df['price_change'] / df['volume']
    
    # Initialize Gain and Loss Arrays
    gains = np.zeros(len(df))
    losses = np.zeros(len(df))
    
    # Calculate EMA of Volume Adjusted Returns
    ema_var = pd.Series(index=df.index)
    ema_var[0] = df['volume_adjusted_return'].iloc[0]
    n_days = 20
    for i in range(1, len(df)):
        std_dev = df['volume_adjusted_return'].iloc[max(0, i-n_days):i].std()
        if std_dev > 0.01:  # High volatility
            N = 5
        else:  # Low volatility
            N = 20
        multiplier = 2 / (N + 1)
        ema_var[i] = (df['volume_adjusted_return'].iloc[i] * multiplier) + (ema_var[i-1] * (1 - multiplier))
    
    # Generate Alpha Factor
    df['high_low_range'] = df['high'] - df['low']
    df['vol_weighted_high_low_range'] = df['high_low_range'] * df['volume']
    
    # Calculate Exponential Moving Average of Volume Weighted High-Low Range
    ema_vol_weighted_high_low = pd.Series(index=df.index)
    ema_vol_weighted_high_low[0] = df['vol_weighted_high_low_range'].iloc[0]
    for i in range(1, len(df)):
        std_dev = df['vol_weighted_high_low_range'].iloc[max(0, i-n_days):i].std()
        if std_dev > 0.01:  # High volatility
            N = 5
        else:  # Low volatility
            N = 20
        multiplier = 2 / (N + 1)
        ema_vol_weighted_high_low[i] = (df['vol_weighted_high_low_range'].iloc[i] * multiplier) + (ema_vol_weighted_high_low[i-1] * (1 - multiplier))
    
    # Generate Final Alpha Factor
    alpha_factor = ema_var - ema_vol_weighted_high_low
    return alpha_factor
