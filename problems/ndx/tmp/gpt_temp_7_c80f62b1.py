import pandas as pd

def heuristics_v2(df):
    short_window = 10
    long_window = 50
    short_ema = df['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_window, adjust=False).mean()
    ratio = short_ema / long_ema
    # Apply a weighted smoothing using the normalized volume as weights
    smoothed_factor = ratio.rolling(window=3).apply(lambda x: ((x * df['volume'][-len(x):].values).sum() / df['volume'][-len(x):].sum()), raw=False).dropna()
    
    return heuristics_matrix
