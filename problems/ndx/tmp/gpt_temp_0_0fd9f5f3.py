import pandas as pd
import pandas as pd

def heuristics_v2(df, n=20):
    # Calculate Daily Price Range
    df['price_range'] = df['high'] - df['low']
    
    # Calculate Adjusted Relative Volume
    df['vol_ema'] = df['volume'].ewm(span=n, adjust=False).mean()
    df['vol_filter'] = df['volume'] > 2.0 * df['vol_ema']
    
    # Filter days based on volume condition
    filtered_days = df[df['vol_filter']]
    
    # Calculate High-Low Range Ratio
    df['high_low_ratio'] = (df['high'] - df['low']) / df['low']
    
    # Calculate Volume-Adjusted Close-to-Open Return
    df['vol_adj_return'] = (df['close'] - df['open']) * df['volume']
    filtered_days['vol_adj_return_sum'] = filtered_days['vol_adj_return'].rolling(window=n).sum()
    
    # Calculate Cumulative High-Low Momentum
    df['cum_high_low_momentum'] = filtered_days['price_range'].rolling(window=n).sum()
    
    # Combine Cumulative High-Low Momentum and Volume-Adjusted Close-to-Open Return
    df['alpha_factor'] = df['cum_high_low_momentum'] + df['vol_adj_return_sum'].fillna(0)
    
    return df['alpha_factor']
