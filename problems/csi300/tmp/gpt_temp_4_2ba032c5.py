import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    open_price = df['open']
    
    intraday_return = (close - open_price) / open_price
    overnight_gap = (open_price - close.shift(1)) / close.shift(1)
    
    gap_reversal_strength = -overnight_gap * np.sign(intraday_return)
    
    volume_profile = volume.rolling(10).apply(lambda x: np.percentile(x, 70) - np.percentile(x, 30))
    current_volume_rank = (volume - volume.rolling(10).min()) / (volume.rolling(10).max() - volume.rolling(10).min())
    
    volume_imbalance = (current_volume_rank - 0.5) * volume_profile
    
    momentum_persistence = intraday_return.rolling(3).mean() * intraday_return.rolling(3).std()
    
    heuristics_matrix = gap_reversal_strength * volume_imbalance * momentum_persistence
    return heuristics_matrix
