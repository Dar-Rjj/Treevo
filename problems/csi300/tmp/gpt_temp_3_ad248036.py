import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Volatility Calculation using High, Low, and Close prices
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x) - min(x), axis=1)
    volatility_window = 20  # Fixed window for initial volatility calculation
    df['volatility'] = df['true_range'].rolling(window=volatility_window).std()
    
    # Adaptive Window Sizing based on Volatility
    def adaptive_window(volatility, high_vol_threshold=0.5, low_vol_threshold=0.2, default_window=60, high_window=30, low_window=90):
        if volatility > high_vol_threshold:
            return high_window
        elif volatility < low_vol_threshold:
            return low_window
        else:
            return default_window
    
    # Apply Adaptive Window Sizing
    df['window_size'] = df['volatility'].apply(adaptive_window)
    
    # Rolling Statistics with Adaptive Window
    rolling_mean = []
    rolling_std = []
    for i, row in df.iterrows():
        window = int(row['window_size'])
        start = max(0, df.index.get_loc(i) - window + 1)
        end = df.index.get_loc(i) + 1
        mean = df.iloc[start:end]['volume_weighted_return'].mean()
        std = df.iloc[start:end]['volume_weighted_return'].std()
        rolling_mean.append(mean)
        rolling_std.append(std)
    
    df['rolling_mean'] = rolling_mean
    df['rolling_std'] = rolling_std
    
    # Final Alpha Factor
    df['alpha_factor'] = (df['volume_weighted_return'] - df['rolling_mean']) / df['rolling_std']
    
    return df['alpha_factor'].dropna()
