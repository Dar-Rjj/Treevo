import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    mean_reversion = (close - close.rolling(window=12).mean()) / (high.rolling(window=12).max() - low.rolling(window=12).min())
    
    volume_acceleration = volume.rolling(window=8).apply(lambda x: (x[-1] - x.mean()) / x.std())
    
    volatility_breakout = (high - low) - (high.rolling(window=10).mean() - low.rolling(window=10).mean())
    
    heuristics_matrix = mean_reversion + volume_acceleration + volatility_breakout
    
    return heuristics_matrix
