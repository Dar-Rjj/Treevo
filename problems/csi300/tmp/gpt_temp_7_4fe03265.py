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
    df['volatility'] = df[['high', 'low', 'close']].std(axis=1).rolling(window=30).std()
    
    # Adjust Window Size based on Volatility
    def adjust_window_size(volatility, low_vol_threshold=0.05, high_vol_threshold=0.15, base_window=60):
        if volatility > high_vol_threshold:
            return max(10, int(base_window * 0.5))  # Decrease window size
        elif volatility < low_vol_threshold:
            return min(120, int(base_window * 1.5))  # Increase window size
        else:
            return base_window
    
    df['window_size'] = df['volatility'].apply(adjust_window_size)
    
    # Calculate Rolling Statistics with Adaptive Window
    df['rolling_mean'] = df.apply(lambda row: df.loc[:row.name, 'volume_weighted_return'].rolling(window=row['window_size']).mean().iloc[-1], axis=1)
    df['rolling_std'] = df.apply(lambda row: df.loc[:row.name, 'volume_weighted_return'].rolling(window=row['window_size']).std().iloc[-1], axis=1)
    
    # Combine the two rolling statistics to form a final factor
    df['alpha_factor'] = (df['rolling_mean'] - df['rolling_std']) / df['rolling_std']
    
    return df['alpha_factor']
