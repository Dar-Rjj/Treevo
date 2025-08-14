import pandas as pd

def heuristics_v2(df):
    df['daily_return'] = df['close'].pct_change()
    heuristics_matrix = df['daily_return'].rolling(window=30).apply(lambda x: (x - x.mean()).abs().mean(), raw=True)
    return heuristics_matrix
