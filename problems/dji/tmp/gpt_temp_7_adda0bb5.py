def heuristics_v2(df):
    ema_high_5 = df['high'].ewm(span=5, adjust=False).mean()
    ema_low_5 = df['low'].ewm(span=5, adjust=False).mean()
    ema_high_10 = df['high'].ewm(span=10, adjust=False).mean()
    ema_low_10 = df['low'].ewm(span=10, adjust=False).mean()
    ratio_5 = ema_high_5 / ema_low_5
    ratio_10 = ema_high_10 / ema_low_10
    vol_change = df['volume'].diff().apply(lambda x: 1 if pd.isna(x) else abs(x))
    weight = vol_change.apply(lambda x: x ** (1/3))
    heuristics_matrix = (ratio_5 * ratio_10) * weight
    return heuristics_matrix
