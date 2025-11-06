def heuristics_v2(df):
    intraday_range = (df['high'] - df['low']) / df['open']
    range_autocorr = intraday_range.rolling(window=5).apply(lambda x: x.autocorr(lag=1), raw=False)
    volume_accel = df['volume'] / df['volume'].shift(1) - 1
    heuristics_matrix = range_autocorr * volume_accel
    return heuristics_matrix
