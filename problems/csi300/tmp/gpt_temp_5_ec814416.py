import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['close'] - df['open']) / df['open']
    
    # Calculate Volume-Weighted Return
    df['volume_weighted_return'] = (df['volume'] * df['close']) / df['volume'].rolling(window=10).sum()
    
    # Calculate Amount-Weighted Return
    df['amount_weighted_return'] = (df['amount'] * df['close']) / df['amount'].rolling(window=10).sum()
    
    # Compute recent volatility (10-day standard deviation of returns)
    df['log_returns'] = np.log(df['close']).diff()
    df['volatility'] = df['log_returns'].rolling(window=10).std()
    
    # Define volatility thresholds
    high_vol_threshold = df['volatility'].quantile(0.75)
    low_vol_threshold = df['volatility'].quantile(0.25)
    
    # Assign dynamic weights based on volatility
    conditions = [
        (df['volatility'] > high_vol_threshold),
        (df['volatility'] <= high_vol_threshold) & (df['volatility'] > low_vol_threshold),
        (df['volatility'] <= low_vol_threshold)
    ]
    
    choices = [
        [0.4, 0.2, 0.2, 0.2],
        [0.3, 0.3, 0.2, 0.2],
        [0.2, 0.4, 0.2, 0.2]
    ]
    
    weights = np.select(conditions, choices)
    
    # Combine factors with dynamic weights
    df['factor_value'] = (
        df['intraday_range'] * weights[:, 0] +
        df['close_to_open_return'] * weights[:, 1] +
        df['volume_weighted_return'] * weights[:, 2] +
        df['amount_weighted_return'] * weights[:, 3]
    )
    
    return df['factor_value']
