def heuristics_v2(df):
    roc_close = df['close'].pct_change(periods=10)
    cumsum_log_volume = df['volume'].apply(lambda x: (x + 1e-5)).apply(np.log).rolling(window=10).sum()
    ema_high_low_diff = (df['high'] - df['low']).ewm(span=10, adjust=False).mean()
    heuristics_matrix = (roc_close * 0.4) + (cumsum_log_volume * 0.3) + (ema_high_low_diff * 0.3)
    return heuristics_matrix
