import pandas as pd

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    typical_price = (high + low + close) / 3
    price_deviation = (close - typical_price) / (high - low)
    
    volume_persistence = volume.rolling(window=10).apply(lambda x: (x > x.median()).sum())
    
    reversal_signal = -price_deviation * volume_persistence
    
    heuristics_matrix = reversal_signal.rolling(window=5).mean()
    
    return heuristics_matrix
