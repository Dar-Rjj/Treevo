import pandas as pd

def heuristics_v2(df):
    price_diff = (df['close'] - df['open']).abs()
    log_volume_adj = price_diff * df['volume'].apply(lambda x: x if x > 0 else 1).apply(lambda x: (x+1).log())
    momentum = df['close'].rolling(window=7).mean().pct_change(7)
    heuristics_matrix = 0.5 * log_volume_adj + 0.5 * momentum
    return heuristics_matrix
