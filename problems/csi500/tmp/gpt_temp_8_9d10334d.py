import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Price acceleration (3-day momentum of 5-day momentum)
    mom_5 = close.pct_change(5)
    price_acceleration = mom_5.diff(3)
    
    # Volatility-adjusted long-term momentum (20-day)
    vol_20 = close.pct_change().rolling(20).std()
    mom_20 = close.pct_change(20)
    adj_momentum = mom_20 / (vol_20 + 1e-8)
    
    # Acceleration to momentum ratio
    accel_ratio = price_acceleration / (adj_momentum + 1e-8 * np.sign(adj_momentum))
    
    # Volume-weighted VWAP trend
    vwap = amount / volume
    vwap_trend = vwap.rolling(5).mean() / vwap.rolling(20).mean() - 1
    
    # Combine signals with volume confirmation
    raw_signal = accel_ratio * vwap_trend
    
    # Apply volume-based smoothing
    volume_weight = volume.rolling(10).mean() / volume.rolling(50).mean()
    smoothed_signal = raw_signal.rolling(5).apply(
        lambda x: np.average(x, weights=volume_weight.iloc[x.index.get_loc(x.index[0]):x.index.get_loc(x.index[-1])+1])
        if not x.empty else np.nan
    )
    
    heuristics_matrix = smoothed_signal
    heuristics_matrix.name = 'heuristics_v2'
    
    return heuristics_matrix
