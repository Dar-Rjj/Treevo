import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    price_acceleration = close.pct_change(5) - close.pct_change(10)
    price_acceleration = price_acceleration * close.rolling(window=15).std()
    
    volatility_filter = (high - low).rolling(window=20).mean()
    volume_persistence = volume.rolling(window=10).apply(lambda x: (x[-3:].mean() > x[:7].mean()).astype(float))
    regime_filtered_volume = volume_persistence * volatility_filter
    
    price_reversal = -close.pct_change(3)
    volatility_scaling = (high - low).rolling(window=10).std()
    scaled_reversal = price_reversal * volatility_scaling
    
    heuristics_matrix = price_acceleration + regime_filtered_volume + scaled_reversal
    
    return heuristics_matrix
