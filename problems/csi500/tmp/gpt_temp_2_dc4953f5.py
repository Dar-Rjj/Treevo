import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    # Volatility-adjusted momentum
    vol_window = 20
    ret_5d = close.pct_change(5)
    vol_20d = close.pct_change().rolling(vol_window).std()
    vol_adj_momentum = ret_5d / vol_20d
    
    # Volume-confirmed reversal pattern
    price_range = (high - low) / close
    vwap = amount / volume
    price_deviation = (close - vwap) / vwap
    
    volume_rank = volume.rolling(10).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    reversal_signal = -price_deviation * volume_rank
    
    # Combine factors
    combined_factor = vol_adj_momentum.rolling(5).mean() + reversal_signal.rolling(3).mean()
    
    heuristics_matrix = combined_factor
    return heuristics_matrix
