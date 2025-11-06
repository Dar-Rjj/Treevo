import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    # Volatility-normalized momentum
    ret_5 = close.pct_change(5)
    ret_10 = close.pct_change(10)
    vol_20 = close.pct_change().rolling(20).std()
    norm_momentum = (ret_5 - ret_10) / (vol_20 + 1e-8)
    
    # Volume-confirmed breakout
    avg_volume_20 = volume.rolling(20).mean()
    volume_ratio = volume / (avg_volume_20 + 1e-8)
    high_break = (high - high.rolling(20).max()) / high
    volume_breakout = high_break * volume_ratio
    
    # Relative strength mean reversion
    rs_5 = close / close.rolling(5).mean()
    rs_20 = close / close.rolling(20).mean()
    mean_reversion = (rs_5 - rs_20) / (rs_20 + 1e-8)
    
    # Combine factors
    heuristics_matrix = norm_momentum + volume_breakout - mean_reversion
    
    return heuristics_matrix
