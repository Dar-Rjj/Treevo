import pandas as pd

def heuristics_v2(df):
    # Calculate 15-day EMA of the closing price
    ema15_close = df['close'].ewm(span=15, adjust=False).mean()
    
    # Calculate 60-day EMA of the adjusted close price (assuming 'adj_close' column exists)
    ema60_adj_close = df['adj_close'].ewm(span=60, adjust=False).mean()
    
    # Calculate 20-day median of the volume
    median_volume_20 = df['volume'].rolling(window=20).median()
    
    # Positive difference between today's volume and 20-day median volume
    volume_diff = (df['volume'] - median_volume_20).apply(lambda x: x if x > 0 else 0)
    
    # Combine factors
    heuristics_matrix = (ema15_close - ema60_adj_close) + volume_diff
    
    return heuristics_matrix
