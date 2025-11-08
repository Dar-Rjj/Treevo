import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    price_acceleration = close.pct_change(5) - close.pct_change(10)
    acceleration_decay = price_acceleration.rolling(window=8).apply(lambda x: x.iloc[-1] - x.mean())
    
    volatility_clustering = (high - low).rolling(window=12).apply(lambda x: x.iloc[-3:].mean() / x.iloc[:9].mean())
    
    volume_divergence = volume.rolling(window=15).apply(lambda x: (x.iloc[-5:].mean() - x.iloc[:10].mean()) / x.std())
    volume_momentum = volume_divergence * close.pct_change(7)
    
    heuristics_matrix = acceleration_decay + volatility_clustering + volume_momentum
    
    return heuristics_matrix
