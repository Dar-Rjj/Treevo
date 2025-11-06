import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Price acceleration component
    short_return = close.pct_change(periods=5)
    long_return = close.pct_change(periods=15)
    price_acceleration = short_return - long_return
    
    # Volume-confirmed trend strength
    volume_ma = volume.rolling(window=10).mean()
    volume_ratio = volume / volume_ma
    trend_strength = close.rolling(window=10).apply(lambda x: (x[-1] - x[0]) / (np.std(x) + 1e-8))
    volume_confirmation = trend_strength * volume_ratio
    
    # Volatility adjustment
    volatility = (high - low).rolling(window=10).std()
    
    # Combine components
    heuristics_matrix = price_acceleration * volume_confirmation / (volatility + 1e-8)
    
    return heuristics_matrix
