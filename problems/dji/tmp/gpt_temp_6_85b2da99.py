def heuristics_v2(df):
    ema_high_5 = df['high'].ewm(span=5, adjust=False).mean()
    ema_low_5 = df['low'].ewm(span=5, adjust=False).mean()
    ema_high_10 = df['high'].ewm(span=10, adjust=False).mean()
    ema_low_10 = df['low'].ewm(span=10, adjust=False).mean()
    diff_5 = ema_high_5 - ema_low_5
    diff_10 = ema_high_10 - ema_low_10
    vol_log_change = df['volume'].pct_change().apply(lambda x: 1 if pd.isna(x) else x).apply(lambda x: abs(x))
    weight = vol_log_change.apply(lambda x: x**(1/3))
    heuristics_matrix = pd.concat([diff_5, diff_10], axis=1).max(axis=1) * weight
    return heuristics_matrix
