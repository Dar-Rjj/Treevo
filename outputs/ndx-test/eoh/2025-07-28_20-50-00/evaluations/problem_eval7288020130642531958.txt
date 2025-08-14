import pandas as pd

def heuristics_v2(df):
    df['close_high_ratio'] = df['close'] / df['high']
    df['log_volume_return'] = df['volume'].pct_change().apply(lambda x: np.log(1 + x))
    ema_close_high_ratio = df['close_high_ratio'].ewm(span=10, adjust=False).mean()
    heuristics_matrix = ema_close_high_ratio * df['log_volume_return'].fillna(0)
    return heuristics_matrix
