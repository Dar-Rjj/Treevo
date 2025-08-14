def heuristics_v2(df):
    ema_high_5 = df['high'].ewm(span=5, adjust=False).mean()
    ema_low_5 = df['low'].ewm(span=5, adjust=False).mean()
    ema_high_10 = df['high'].ewm(span=10, adjust=False).mean()
    ema_low_10 = df['low'].ewm(span=10, adjust=False).mean()
    ema_high_20 = df['high'].ewm(span=20, adjust=False).mean()
    ema_low_20 = df['low'].ewm(span=20, adjust=False).mean()
    diff_5 = ema_high_5 - ema_low_5
    diff_10 = ema_high_10 - ema_low_10
    diff_20 = ema_high_20 - ema_low_20
    daily_return = df['close'].pct_change().apply(lambda x: 1 if pd.isna(x) else x)
    weight = daily_return.apply(lambda x: abs(x)**(1/3))
    heuristics_matrix = (diff_5 + diff_10 + diff_20) / 3 * weight
    return heuristics_matrix
