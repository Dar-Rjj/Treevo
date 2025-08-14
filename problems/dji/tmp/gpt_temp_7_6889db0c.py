def heuristics_v2(df):
    ema_high_5 = df['high'].ewm(span=5, adjust=False).mean()
    ema_low_5 = df['low'].ewm(span=5, adjust=False).mean()
    ema_high_10 = df['high'].ewm(span=10, adjust=False).mean()
    ema_low_10 = df['low'].ewm(span=10, adjust=False).mean()
    diff_ratio = (ema_high_5 - ema_low_5) / (ema_high_10 - ema_low_10)
    vol_pct_change = df['volume'].pct_change().apply(lambda x: 1 if pd.isna(x) else x).apply(lambda x: abs(x))
    weight = vol_pct_change.apply(lambda x: x**0.5)
    heuristics_matrix = diff_ratio * weight
    return heuristics_matrix
