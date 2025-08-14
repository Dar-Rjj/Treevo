import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Volatility Calculation (using Parkinson's volatility estimator)
    df['daily_volatility'] = np.sqrt((1/4*np.log(df['high']/df['low'])**2))
    
    # Adaptive Window Size
    def adaptive_window(volatility, high_vol=0.05, low_vol=0.01, max_window=60, min_window=20):
        return max(min(max_window, int((high_vol - volatility) / (high_vol - low_vol) * (max_window - min_window) + min_window)), min_window)
    
    df['window_size'] = df['daily_volatility'].apply(adaptive_window)
    
    # Rolling Statistics with Adaptive Window
    df['rolling_mean'] = df.groupby('volume_weighted_return')['volume_weighted_return'].transform(lambda x: x.rolling(x.name, min_periods=1).mean())
    df['rolling_std'] = df.groupby('volume_weighted_return')['volume_weighted_return'].transform(lambda x: x.rolling(x.name, min_periods=1).std())
    
    # Final Alpha Factor
    df['alpha_factor'] = (df['volume_weighted_return'] - df['rolling_mean']) / df['rolling_std']
    
    return df['alpha_factor']
