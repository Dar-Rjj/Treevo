def heuristics_v2(df):
    df['10d_high'] = df['high'].rolling(window=10).max()
    df['10d_low'] = df['low'].rolling(window=10).min()
    df['dist_to_10d_extreme'] = (df[['close', '10d_high', '10d_low']]
                                 .apply(lambda x: min(abs(x[0] - x[1]), abs(x[0] - x[2])) * (1 + 1/abs(x[0] - (x[1]+x[2])/2)), axis=1))
    heuristics_matrix = df['dist_to_10d_extreme'].dropna()
    return heuristics_matrix
