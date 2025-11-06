import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Volatility-adjusted momentum component
    ret_5 = close.pct_change(5)
    ret_10 = close.pct_change(10)
    vol_20 = close.pct_change().rolling(20).std()
    vol_adj_momentum = (ret_5 - ret_10) / (vol_20 + 1e-8)
    
    # Volume-based mean reversion component
    vwap = amount / (volume + 1e-8)
    price_deviation = (close - vwap) / vwap
    volume_rank = volume.rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    volume_reversion = -price_deviation * volume_rank
    
    # Combined alpha factor
    heuristics_matrix = vol_adj_momentum + volume_reversion
    
    return heuristics_matrix
