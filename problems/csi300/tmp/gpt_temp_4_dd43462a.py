import pandas as pd

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    window = 20
    
    upside_vol = (high - close).rolling(window=window).std()
    downside_vol = (close - low).rolling(window=window).std()
    
    volatility_asymmetry = upside_vol - downside_vol
    
    volume_trend = volume.rolling(window=window).apply(lambda x: pd.Series(x).pct_change().mean())
    
    heuristics_matrix = volatility_asymmetry * volume_trend
    
    return heuristics_matrix
