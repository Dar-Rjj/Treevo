import pandas as pd

def heuristics_v2(df):
    df['log_close_diff'] = df['close'].apply(lambda x: np.log(x) if x > 0 else 0).diff()
    df['log_volume_change'] = df['volume'].pct_change().apply(lambda x: np.log(1 + x) if x > -1 else 0)
    heuristics_matrix = (df['log_close_diff'] / df['log_volume_change']).fillna(0)
    return heuristics_matrix
