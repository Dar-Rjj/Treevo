import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Parameters
    short_window = 5
    long_window = 20
    cum_diff_window = 10
    vol_trend_window = 10
    rel_strength_n = 5
    vol_activity_m = 5
    
    # Calculate Daily High-Low Difference
    df['high_low_diff'] = df['high'] - df['low']
    
    # Cumulate the Moving Difference
    df['cum_high_low_diff'] = df['high_low_diff'].rolling(window=cum_diff_window, min_periods=1).sum()
    
    # Calculate Volume Trend
    df['volume_trend'] = df['volume'].rolling(window=vol_trend_window, min_periods=1).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
    
    # Calculate Relative Strength
    df['relative_strength'] = df['close'] / df['close'].shift(rel_strength_n)
    
    # Measure Volume Activity Change
    df['avg_volume'] = df['volume'].rolling(window=vol_activity_m, min_periods=1).mean()
    df['volume_change'] = df['volume'] - df['avg_volume']
    
    # Combine Relative Strength and Volume Change
    df['intermediate_factor'] = df['relative_strength'] * df['volume_change']
    
    # Calculate Daily Price Momentum
    df['price_momentum'] = df['close'] - df['close'].shift(1)
    
    # Calculate Short-Term Trend
    df['short_term_trend'] = df['price_momentum'].rolling(window=short_window, min_periods=1).mean()
    
    # Calculate Long-Term Trend
    df['long_term_trend'] = df['price_momentum'].rolling(window=long_window, min_periods=1).mean()
    
    # Generate Integrated Alpha Factor
    df['alpha_factor'] = (df['long_term_trend'] - df['short_term_trend']) * df['intermediate_factor'] * df['cum_high_low_diff'] * df['volume_trend']
    
    return df['alpha_factor']
