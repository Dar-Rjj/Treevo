def heuristics_v2(df):
    df['10d_high'] = df['high'].rolling(window=10).max()
    df['10d_low'] = df['low'].rolling(window=10).min()
    df['dist_to_10d_max'] = (df['10d_high'] - df['close'])
    df['dist_to_10d_min'] = (df['close'] - df['10d_low'])
    df['weight_max'] = 1 / (df['dist_to_10d_max'] + 1e-6).apply(np.exp)
    df['weight_min'] = 1 / (df['dist_to_10d_min'] + 1e-6).apply(np.exp)
    df['weighted_avg_rel_dist'] = (df['weight_max']*df['dist_to_10d_max'] + df['weight_min']*df['dist_to_10d_min']) / (df['weight_max'] + df['weight_min'])
    heuristics_matrix = df['weighted_avg_rel_dist'].dropna()
    return heuristics_matrix
