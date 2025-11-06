import pandas as pd

def heuristics_v2(df):
    close = df['close']
    
    price_median = close.rolling(window=10, min_periods=1).median()
    price_mad = close.rolling(window=10, min_periods=1).apply(lambda x: (x - x.median()).abs().median())
    
    heuristics_matrix = (close - price_median) / (price_mad + 1e-8)
    
    return heuristics_matrix
