import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Short-term price acceleration (3-day momentum minus 6-day momentum)
    mom_3 = close.pct_change(3)
    mom_6 = close.pct_change(6)
    price_accel = mom_3 - mom_6
    
    # Long-term volatility-adjusted momentum (15-day return divided by 20-day volatility)
    mom_15 = close.pct_change(15)
    vol_20 = close.pct_change().rolling(20).std()
    vol_adj_momentum = mom_15 / (vol_20 + 1e-8)
    
    # Volume confirmation (volume trend vs price trend)
    vol_ma_5 = volume.rolling(5).mean()
    vol_ma_10 = volume.rolling(10).mean()
    vol_trend = vol_ma_5 / vol_ma_10 - 1
    
    # Combine components with non-linear interactions
    heuristics_matrix = price_accel * np.sign(vol_adj_momentum) * (1 + np.tanh(vol_trend))
    
    return heuristics_matrix
