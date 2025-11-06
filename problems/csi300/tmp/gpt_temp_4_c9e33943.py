import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Short-term momentum acceleration (3-day vs 6-day momentum)
    mom_3 = close.pct_change(3)
    mom_6 = close.pct_change(6)
    momentum_accel = mom_3 - mom_6
    
    # Volatility-adjusted long-term momentum (20-day)
    vol_20 = close.pct_change().rolling(20).std()
    mom_20 = close.pct_change(20)
    vol_adj_momentum = mom_20 / (vol_20 + 1e-8)
    
    # Price range efficiency (true range normalized by closing price)
    true_range = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
    range_efficiency = true_range / close
    
    # Volume confirmation (volume trend relative to price movement)
    volume_trend = volume.rolling(5).mean() / volume.rolling(20).mean()
    volume_confirmation = np.sign(mom_3) * volume_trend
    
    # Combine components with non-linear interactions
    factor = (momentum_accel * vol_adj_momentum * 
              (1 - range_efficiency.rolling(5).mean()) * 
              np.tanh(volume_confirmation))
    
    heuristics_matrix = factor
    return heuristics_matrix
