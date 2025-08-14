import pandas as pd

def heuristics_v2(df):
    # Calculate 21-day Exponential Moving Average
    ema_21 = df['close'].ewm(span=21, adjust=False).mean()
    
    # Calculate 14-day Average True Range
    tr = df[['high', 'low']].apply(lambda x: x.diff().abs().max(axis=1), axis=1)
    atr_14 = tr.rolling(window=14).mean()
    
    # Momentum - Rate of Change over 14 days
    roc_14 = df['close'].pct_change(periods=14)
    
    # Composite heuristic
    heuristics_matrix = (ema_21 + atr_14 + roc_14) / 3
    return heuristics_matrix
