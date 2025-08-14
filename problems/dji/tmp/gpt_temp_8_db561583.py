import pandas as pd

def heuristics_v2(df):
    df['ema_close'] = df['close'].ewm(span=10, adjust=False).mean()
    df['daily_returns'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['mad'] = df['daily_returns'].rolling(window=10).apply(lambda x: x.abs().mean(), raw=True)
    heuristics_matrix = 0.6 * df['ema_close'] + 0.4 * df['mad']
    return heuristics_matrix
