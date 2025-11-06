import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Short-term reversal component (3-day)
    ret_3d = close.pct_change(3)
    rev_factor = -ret_3d.rolling(window=10).rank(pct=True)
    
    # Medium-term momentum component (15-day)
    mom_15d = close.pct_change(15)
    mom_factor = mom_15d.rolling(window=20).apply(lambda x: (x > x.median()).astype(float).iloc[-1])
    
    # Volume-confirmed breakout detection
    vol_ma = volume.rolling(window=10).mean()
    price_range = (high - low) / close
    range_ma = price_range.rolling(window=10).mean()
    
    breakout_signal = ((volume > vol_ma * 1.2) & 
                      (price_range > range_ma * 1.1) & 
                      (close > close.shift(3))).astype(float)
    
    # Combine factors with dynamic weighting
    heuristics_matrix = (rev_factor * 0.4 + 
                        mom_factor * 0.3 + 
                        breakout_signal * 0.3)
    
    return heuristics_matrix
