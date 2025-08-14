import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Determine Volatility
    df['price_range'] = df['high'] - df['low']
    df['volatility'] = df['price_range'].rolling(window=20).std()
    
    # Adjust Window Size based on Volatility
    def adjust_window(volatility, low_vol_threshold=0.5, high_vol_threshold=1.5, low_vol_window=60, high_vol_window=20):
        if volatility < low_vol_threshold:
            return low_vol_window
        elif volatility > high_vol_threshold:
            return high_vol_window
        else:
            return 40  # Neutral window size
    
    adaptive_windows = df['volatility'].apply(adjust_window)
    
    # Calculate Rolling Statistics with Adaptive Windows
    rolling_mean = []
    rolling_std = []
    for i in range(len(df)):
        window = adaptive_windows.iloc[i]
        data = df['volume_weighted_return'].iloc[max(0, i+1-window):i+1]
        rolling_mean.append(np.mean(data))
        rolling_std.append(np.std(data))
    
    factor_values = (pd.Series(rolling_mean, index=df.index) 
                     / (pd.Series(rolling_std, index=df.index) + 1e-8))  # Avoid division by zero
    return factor_values
