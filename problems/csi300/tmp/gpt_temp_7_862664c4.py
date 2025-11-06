import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    price_momentum = close.pct_change(5)
    volatility = (high - low).rolling(window=10).std()
    volume_acceleration = volume.pct_change(3)
    volume_vol_scaled = volume_acceleration / volatility.replace(0, np.nan)
    
    heuristics_matrix = price_momentum - volume_vol_scaled
    return heuristics_matrix
