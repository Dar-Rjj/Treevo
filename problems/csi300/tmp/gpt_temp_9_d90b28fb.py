import pandas as pd

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    window = 15
    
    price_range = (high - low) / close
    volatility = price_range.rolling(window=window).std()
    
    volume_trend = volume.rolling(window=window).apply(lambda x: pd.Series(x).pct_change().mean())
    
    momentum = (close / close.shift(5) - 1).rolling(window=window).mean()
    
    heuristics_matrix = volatility * volume_trend * momentum
    
    return heuristics_matrix
