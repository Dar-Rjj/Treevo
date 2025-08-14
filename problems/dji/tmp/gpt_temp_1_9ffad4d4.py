import pandas as pd

def heuristics_v2(df):
    log_return = df['close'].pct_change().apply(lambda x: np.log(1 + x)).rolling(window=10).mean()
    sma_20 = df['close'].rolling(window=20).mean()
    sma_50 = df['close'].rolling(window=50).mean()
    sma_ratio = sma_20 / sma_50
    composite_factor = (log_return + sma_ratio) / 2
    heuristics_matrix = composite_factor.ewm(span=10, adjust=False).mean()
    return heuristics_matrix
