import pandas as pd

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    momentum = close.pct_change(5)
    volatility = (high - low).rolling(10).std()
    volume_trend = volume.rolling(10).mean()
    
    heuristics_matrix = momentum / (volatility * volume_trend)
    
    return heuristics_matrix
