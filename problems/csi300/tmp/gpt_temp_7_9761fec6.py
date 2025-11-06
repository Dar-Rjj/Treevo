import pandas as pd

def heuristics_v2(df):
    close = df['close']
    volume = df['volume']
    
    window = 15
    
    price_volatility = (close - close.rolling(window=window).mean()) / close.rolling(window=window).std()
    
    volume_trend = volume.rolling(window=window).apply(lambda x: pd.Series(x).pct_change().mean(), raw=False)
    
    heuristics_matrix = price_volatility * volume_trend
    
    return heuristics_matrix
