import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Price Difference
    df['high_low_diff'] = df['high'] - df['low']
    
    # Calculate Cumulative High-Low Difference over 15 days
    df['cum_high_low_diff'] = df['high_low_diff'].rolling(window=15).sum()
    
    # Calculate Daily Close-to-Open Return
    df['close_to_open_return'] = df['close'] / df['open']
    
    # Calculate Cumulative Close-to-Open Return over 15 days
    df['cum_close_to_open_return'] = df['close_to_open_return'].rolling(window=15).sum()
    
    # Identify Volume Surges
    df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_surge'] = (df['volume'] > 1.5 * df['vol_ma_20']).astype(int)
    
    # Combine Indicators
    df['combined_indicator'] = df['cum_high_low_diff'] * df['cum_close_to_open_return'] * df['volume_surge']
    
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()
    
    # Compute Cumulative Weighted Return
    lookback_period = 15
    half_life = 7
    decay_factor = 0.5 ** (1 / half_life)
    weights = np.array([decay_factor ** t for t in range(lookback_period)])
    weights /= weights.sum()
    df['weighted_return'] = df['daily_return'].rolling(window=lookback_period).apply(lambda x: np.nansum(x * weights))
    
    # Adjust by Volatility
    df['true_range'] = df[['high', 'low']].sub(df['close'].shift(), axis=0).abs().max(axis=1)
    df['avg_true_range'] = df['true_range'].rolling(window=lookback_period).mean()
    df['adjusted_weighted_return'] = df['weighted_return'] / df['avg_true_range']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['combined_indicator'] * df['adjusted_weighted_return']
    
    return df['alpha_factor']
