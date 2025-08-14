def heuristics_v2(df):
    df['5d_high'] = df['high'].rolling(window=5).max()
    df['5d_low'] = df['low'].rolling(window=5).min()
    df['weighted_dist_to_5d_extreme'] = (df[['close', '5d_high', '5d_low', 'volume']]
                                         .apply(lambda x: ((x[3] * abs(x[0] - x[1])) + (x[3] * abs(x[0] - x[2]))) / (2 * x[3]), axis=1))
    heuristics_matrix = df['weighted_dist_to_5d_extreme'].dropna()
    return heuristics_matrix
