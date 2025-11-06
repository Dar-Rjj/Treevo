import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    price_range = (high - low) / close
    vol_change = volume / volume.rolling(10).mean()
    
    momentum = close.pct_change(5)
    weight = momentum.rolling(5).mean()
    
    raw_factor = price_range * vol_change * weight
    heuristics_matrix = raw_factor
    
    return heuristics_matrix
