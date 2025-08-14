import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Determine Volatility using High, Low, and Close prices
    df['log_high_close'] = np.log(df['high'] / df['close'])
    df['log_low_close'] = np.log(df['low'] / df['close'])
    df['volatility'] = df[['log_high_close', 'log_low_close']].rolling(window=20).std().mean(axis=1)
    
    # Adjust Window Size based on Volatility
    def get_window_size(volatility):
        if volatility > df['volatility'].quantile(0.75):
            return 10
        else:
            return 30
    
    df['window_size'] = df['volatility'].apply(get_window_size)
    
    # Rolling Statistics with Adaptive Window
    rolling_mean = []
    rolling_std = []
    for i in range(len(df)):
        window = int(df['window_size'].iloc[i])
        mean = df['volume_weighted_return'].iloc[max(0, i-window+1):i+1].mean()
        std = df['volume_weighted_return'].iloc[max(0, i-window+1):i+1].std()
        rolling_mean.append(mean)
        rolling_std.append(std)
    
    df['rolling_mean'] = rolling_mean
    df['rolling_std'] = rolling_std
    
    # Volatility Adjustment
    df['alpha_factor'] = df['volume_weighted_return'] / df['rolling_std']
    
    # Return the final alpha factor
    return df['alpha_factor']
