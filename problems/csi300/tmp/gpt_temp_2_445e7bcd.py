def heuristics_v2(df):
    df['5d_high'] = df['high'].rolling(window=5).max()
    df['5d_low'] = df['low'].rolling(window=5).min()
    df['days_since_5d_high'] = (df.index - df[df['high'].eq(df['5d_high'])].index).days
    df['days_since_5d_low'] = (df.index - df[df['low'].eq(df['5d_low'])].index).days
    df['weight_high'] = 1 / (df['days_since_5d_high'] + 1)
    df['weight_low'] = 1 / (df['days_since_5d_low'] + 1)
    df['weighted_dist_to_5d_extreme'] = (df[['close', '5d_high', '5d_low', 'weight_high', 'weight_low']]
                                         .apply(lambda x: min(x[3] * abs(x[0] - x[1]), x[4] * abs(x[0] - x[2])), axis=1))
    heuristics_matrix = df['weighted_dist_to_5d_extreme'].dropna()
    return heuristics_matrix
