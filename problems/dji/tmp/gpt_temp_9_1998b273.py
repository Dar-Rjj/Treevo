def heuristics_v2(df):
    ema_high_low = ((df['high'] + df['low']) / 2).ewm(span=10, adjust=False).mean()
    heuristics_matrix = (df['close'] / ema_high_low) * np.log(df['volume'])
    return heuristics_matrix
