import pandas as pd

def heuristics_v2(df):
    close = df['close']
    volume = df['volume']
    
    window = 10
    
    price_reversal = (close.shift(5) - close) / close.shift(5)
    volume_trend = volume.rolling(window=window).apply(lambda x: pd.Series(x).corr(pd.Series(range(window))), raw=True)
    
    heuristics_matrix = price_reversal * volume_trend
    
    return heuristics_matrix
