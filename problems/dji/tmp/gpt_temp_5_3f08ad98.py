def heuristics_v2(df):
    ema_high_5 = df['high'].ewm(span=5, adjust=False).mean()
    ema_close_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_high_10 = df['high'].ewm(span=10, adjust=False).mean()
    ema_close_10 = df['close'].ewm(span=10, adjust=False).mean()
    diff_5 = ema_high_5 - ema_close_5
    diff_10 = ema_high_10 - ema_close_10
    vol_change = df['volume'].pct_change().apply(lambda x: 0 if pd.isna(x) else x)
    scaling_factor = vol_change.apply(lambda x: abs(x) ** (1/3))
    heuristics_matrix = (diff_5 * diff_10) * scaling_factor
    return heuristics_matrix
