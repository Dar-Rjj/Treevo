import pandas as pd

def heuristics_v2(df):
    short_window = 10
    long_window = 50
    alpha = 0.3
    
    short_ma = df['close'].rolling(window=short_window).mean()
    long_ma = df['close'].rolling(window=long_window).mean()
    diff = short_ma - long_ma
    smoothed_diff = diff.ewm(alpha=alpha, adjust=False).mean().dropna()
    
    return heuristics_matrix
