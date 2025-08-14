import pandas as pd

def heuristics_v2(df):
    # Calculate the Relative Strength Index (RSI)
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate the Average True Range (ATR)
    tr = df[['high', 'low']].apply(lambda x: x.diff().abs().max(), axis=1)
    atr = tr.rolling(window=14).mean()
    
    # Composite heuristic
    heuristics_matrix = (rsi + atr) / 2
    return heuristics_matrix
