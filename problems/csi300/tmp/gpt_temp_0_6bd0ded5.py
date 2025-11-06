import pandas as pd

def heuristics_v2(df):
    close = df['close']
    volume = df['volume']
    
    window = 20
    
    price_reversal = close / close.shift(1) - 1
    reversal_volatility = price_reversal.rolling(window=window).std()
    
    volume_trend = volume.rolling(window=window).apply(lambda x: pd.Series(x).pct_change().mean())
    volume_persistence = volume_trend.rolling(window=window).std()
    
    heuristics_matrix = reversal_volatility / volume_persistence
    
    return heuristics_matrix
