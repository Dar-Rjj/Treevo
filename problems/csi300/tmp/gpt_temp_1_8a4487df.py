import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Close_to_Open_Return'] * df['volume']
    
    # Determine Volatility
    df['Volatility'] = df[['high', 'low', 'close']].std(axis=1).rolling(window=30).std()
    
    # Adjust Window Size Based on Volatility
    def adaptive_window(volatility, low_vol_threshold=0.5, high_vol_threshold=1.5, min_window=5, max_window=60):
        if volatility < low_vol_threshold:
            return max_window
        elif volatility > high_vol_threshold:
            return min_window
        else:
            return int(max_window - (volatility - low_vol_threshold) / (high_vol_threshold - low_vol_threshold) * (max_window - min_window))
    
    # Apply Adaptive Window
    df['Window_Size'] = df['Volatility'].apply(adaptive_window)
    
    # Rolling Statistics
    def rolling_stats(group, window_size):
        return group.rolling(window=window_size).agg(['mean', 'std'])

    df['Rolling_Stats'] = df.groupby(df.index.date)['Volume_Weighted_Return'].transform(lambda x: rolling_stats(x, df.loc[x.index, 'Window_Size'][0]))
    
    # Generate Alpha Factor
    df['Mean'] = df['Rolling_Stats']['mean']
    df['Std'] = df['Rolling_Stats']['std']
    df['Alpha_Factor'] = df['Mean'] / (df['Std'] + 1e-8)  # Add a small constant to avoid division by zero
    
    return df['Alpha_Factor']
