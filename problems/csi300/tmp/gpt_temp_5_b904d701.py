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
    accel_factor = mom_3 - mom_6
    
    # Volatility-adjusted long-term momentum (20-day)
    vol_20 = close.pct_change().rolling(window=20).std()
    mom_20 = close.pct_change(20)
    vol_adj_momentum = mom_20 / (vol_20 + 1e-8)
    
    # Price range efficiency (how much closing price utilizes daily range)
    range_util = (close - low) / (high - low + 1e-8)
    range_efficiency = range_util.rolling(window=10).mean()
    
    # Volume confirmation (volume trend aligned with price movement)
    volume_trend = volume.rolling(window=10).apply(lambda x: np.corrcoef(x, range(len(x)))[0,1] if len(x) > 1 else 0)
    volume_confirmation = np.sign(mom_3) * volume_trend
    
    # Combine components with non-linear interactions
    heuristics_matrix = (accel_factor * vol_adj_momentum + 
                        range_efficiency * volume_confirmation - 
                        np.tanh(accel_factor * vol_adj_momentum))
    
    return heuristics_matrix
