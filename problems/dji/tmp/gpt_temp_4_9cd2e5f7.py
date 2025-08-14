import pandas as pd

def heuristics_v2(df):
    sma_high = df['high'].rolling(window=10).mean()
    adj_close_ratio = df['close'] / sma_high
    vol_std = df['volume'].rolling(window=30).std()
    volume_log_ratio = (df['volume'] / vol_std).apply(lambda x: x if x > 0 else 1).apply(np.log)
    
    heuristics_matrix = np.log(adj_close_ratio) * volume_log_ratio
    
    return heuristics_matrix
