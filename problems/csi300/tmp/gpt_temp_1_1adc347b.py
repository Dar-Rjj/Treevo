import pandas as pd

def heuristics_v2(df):
    open_price, high, low, close, volume = df['open'], df['high'], df['low'], df['close'], df['volume']
    
    price_deviation = (close - open_price) / open_price
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    volatility = true_range.rolling(window=10).mean()
    
    volume_rank = volume.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    
    heuristics_matrix = -price_deviation / volatility * volume_rank
    
    return heuristics_matrix
