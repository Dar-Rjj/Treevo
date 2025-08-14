import pandas as pd

def heuristics_v2(df):
    log_price_ratio = df['close'].apply(lambda x: x if x > 0 else 1e-6).apply(np.log) - df['open'].apply(lambda x: x if x > 0 else 1e-6).apply(np.log)
    roc_volume = df['volume'].pct_change().fillna(0)
    daily_return_skew = df['close'].pct_change().fillna(0).rolling(window=20).skew()
    
    heuristics_matrix = (log_price_ratio * 0.5) + (roc_volume * 0.4) - (daily_return_skew * 0.1)
    return heuristics_matrix
