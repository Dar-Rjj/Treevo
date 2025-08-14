def heuristics_v2(df):
    ema_close_7 = df['close'].ewm(span=7, adjust=False).mean()
    ema_open_7 = df['open'].ewm(span=7, adjust=False).mean()
    ema_close_14 = df['close'].ewm(span=14, adjust=False).mean()
    ema_open_14 = df['open'].ewm(span=14, adjust=False).mean()
    diff_7 = ema_close_7 - ema_open_7
    diff_14 = ema_close_14 - ema_open_14
    vol_log_change = df['volume'].pct_change().apply(lambda x: 0 if pd.isna(x) or x == 0 else x).apply(lambda x: abs(x))
    weight = vol_log_change.apply(lambda x: x ** (1/3) if x != 0 else 0)
    heuristics_matrix = (diff_7 + diff_14) * weight
    return heuristics_matrix
