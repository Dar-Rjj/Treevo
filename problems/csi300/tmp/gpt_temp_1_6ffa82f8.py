import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Difference
    high_low_diff = df['high'] - df['low']
    
    # Volume-Weight Adjusted Momentum with VWAP Influence
    # Calculate VWAP
    vwap = (df[['open', 'high', 'low', 'close']].mean(axis=1) * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Adjust High-Low Difference by VWAP
    high_low_vwap_diff = (df['high'] - vwap).abs() + (vwap - df['low']).abs()
    volume_weighted_momentum = high_low_diff * df['volume'] * high_low_vwap_diff
    
    # Calculate Cumulative Sum Over a Rolling Window
    window_size = 7
    cumulative_momentum = volume_weighted_momentum.rolling(window=window_size).sum()
    
    # Incorporate Close-to-Open Return with Volume Adjustment
    close_open_return = df['close'] / df['open']
    volume_adjusted_high_low = high_low_diff * df['volume']
    close_open_adj = close_open_return * volume_adjusted_high_low
    
    # Measure Intraday Trading Activity and Volume Patterns
    # Calculate Intraday Volume Variance
    intraday_vol_var = df['volume'].rolling(window=5).std()
    
    # Calculate Volume Patterns
    volume_change = df['volume'].diff()
    volume_patterns = volume_change.rolling(window=5).sum()
    
    # Incorporate Amount-Volume Relationship
    amount_volume_ratio = df['amount'] / df['volume']
    amount_volume_adj = amount_volume_ratio * intraday_vol_var
    
    # Introduce Price-Volume Correlation
    price_volume_corr = df[['close', 'volume']].rolling(window=10).corr().unstack().iloc[::2, 1].reset_index(level=1, drop=True)
    price_volume_corr_adj = price_volume_corr * intraday_vol_var
    
    # Final Combined Factor
    alpha_factor = (cumulative_momentum + close_open_adj + volume_patterns + amount_volume_adj + price_volume_corr_adj) / 5
    
    return alpha_factor
