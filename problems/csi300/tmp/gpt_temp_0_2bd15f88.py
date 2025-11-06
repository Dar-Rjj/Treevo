import pandas as pd

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    window = 20
    
    upside_move = (high - close.shift(1)) / close.shift(1)
    downside_move = (close.shift(1) - low) / close.shift(1)
    
    asymmetry = upside_move.rolling(window=window).mean() - downside_move.rolling(window=window).mean()
    
    volume_trend = volume.rolling(window=window).apply(lambda x: pd.Series(x).pct_change().mean(), raw=False)
    
    heuristics_matrix = asymmetry * volume_trend
    
    return heuristics_matrix
