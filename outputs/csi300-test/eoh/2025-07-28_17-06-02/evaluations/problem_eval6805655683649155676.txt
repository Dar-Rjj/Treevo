import pandas as pd

def heuristics_v2(df):
    # Simple Moving Average
    sma_10 = df['close'].rolling(window=10).mean()
    
    # Weighted Moving Average with Triangular Window
    wma_20 = df['close'].rolling(window=20).apply(lambda x: (x * list(range(1, 21))).sum() / 210, raw=False)
    
    # Modified Stochastic Oscillator
    low_min = df['low'].rolling(window=5).min()
    high_max = df['high'].rolling(window=5).max()
    k_percent = 100 * (df['close'] - low_min) / (high_max - low_min)
    
    # Chaikin Money Flow
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfm * df['volume']
    cmf = mfv.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
    
    # Composite heuristic
    heuristics_matrix = (sma_10 + wma_20 + k_percent + cmf) / 4
    return heuristics_matrix
