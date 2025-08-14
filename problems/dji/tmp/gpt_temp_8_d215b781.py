def heuristics_v2(df):
    ema_high_7 = df['high'].ewm(span=7, adjust=False).mean()
    ema_low_7 = df['low'].ewm(span=7, adjust=False).mean()
    diff_7 = (ema_high_7 - ema_low_7) ** 2
    vol_pct_change = df['volume'].pct_change().apply(lambda x: 0 if pd.isna(x) else x).abs()
    heuristics_matrix = diff_7 * vol_pct_change
    return heuristics_matrix
