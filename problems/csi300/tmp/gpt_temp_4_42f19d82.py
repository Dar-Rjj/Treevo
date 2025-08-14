import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate close-to-open return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Volatility calculation (using Garman-Klass volatility estimator)
    df['volatility'] = 0.5 * np.log(df['high'] / df['low'])**2 - (2 * np.log(2) - 1) * np.log(df['close'] / df['open'])**2
    
    # Define a function to adjust window size based on volatility
    def adaptive_window(volatility, high_vol_threshold=0.01, low_vol_threshold=0.001, max_window=30, min_window=5):
        if volatility > high_vol_threshold:
            return min_window
        elif volatility < low_vol_threshold:
            return max_window
        else:
            return int((max_window - min_window) * (volatility - low_vol_threshold) / (high_vol_threshold - low_vol_threshold)) + min_window
    
    # Apply adaptive window adjustment
    df['window_size'] = df['volatility'].apply(lambda x: adaptive_window(x))
    
    # Initialize the rolling statistics
    df['rolling_mean'] = np.nan
    df['rolling_std'] = np.nan
    
    # Calculate rolling mean and standard deviation with adaptive window
    for i in range(len(df)):
        if i + 1 < len(df):
            window = df.iloc[i]['window_size']
            start = max(i - window + 1, 0)
            end = i + 1
            subset = df['volume_weighted_return'].iloc[start:end]
            df.at[df.index[i], 'rolling_mean'] = subset.mean()
            df.at[df.index[i], 'rolling_std'] = subset.std()

    # Factor value is the rolling mean of the volume-weighted close-to-open return
    factor_values = df['rolling_mean']
    
    return factor_values
