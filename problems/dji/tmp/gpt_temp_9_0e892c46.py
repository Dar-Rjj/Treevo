def heuristics_v2(df):
    ema_volume_10 = df['volume'].ewm(span=10, adjust=False).mean()
    price_change_rate = df['close'].pct_change().apply(lambda x: 1 if pd.isna(x) else x)
    price_diff_ratio = (df['high'] - df['low']) / df['close']
    heuristics_matrix = (price_diff_ratio * ema_volume_10) * price_change_rate
    return heuristics_matrix
