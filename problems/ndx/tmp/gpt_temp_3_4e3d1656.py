import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Volume-Weighted Price
    df['volume_weighted_price'] = (df['High'] * df['Volume'] + df['Low'] * df['Volume']) / (2 * df['Volume'])
    
    # Compute Intraday Range Volatility
    df['intraday_range'] = df['High'] - df['Low']
    df['avg_volume'] = df['Volume'].rolling(window=14).mean()
    df['intraday_volatility'] = df['intraday_range'] / df['avg_volume']
    
    # Calculate High-to-Low Range
    df['high_low_range'] = df['High'] - df['Low']
    
    # Calculate Rolling Average of High-to-Low Range
    df['rolling_high_low_range_avg'] = df['high_low_range'].rolling(window=14).mean()
    
    # Combine Volume-Weighted Price, Intraday Range Volatility, and High-to-Low Range Momentum
    df['vwp_diff'] = df['volume_weighted_price'].diff()
    df['adjusted_vwp_diff'] = df['vwp_diff'] * df['intraday_volatility']
    df['alpha_factor'] = df['adjusted_vwp_diff'] - df['rolling_high_low_range_avg']
    
    return df['alpha_factor'].dropna()
