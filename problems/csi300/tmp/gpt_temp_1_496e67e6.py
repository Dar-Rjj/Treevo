def heuristics_v2(df):
    df['7d_mean_close'] = df['close'].rolling(window=7).mean()
    df['abs_dist_to_7d_mean'] = abs(df['close'] - df['7d_mean_close'])
    total_weight_inv = (1 / df['abs_dist_to_7d_mean']).replace([np.inf, -np.inf], 0).rolling(window=15).sum()
    df['weighted_volume_inv'] = df['volume'] * (1 / df['abs_dist_to_7d_mean']).replace([np.inf, -np.inf], 0)
    df['weighted_amount_inv'] = df['amount'] * (1 / df['abs_dist_to_7d_mean']).replace([np.inf, -np.inf], 0)
    df['weighted_avg_inv'] = (df['weighted_volume_inv'].rolling(window=15).sum() + df['weighted_amount_inv'].rolling(window=15).sum()) / total_weight_inv
    heuristics_matrix = df['weighted_avg_inv'].dropna()
    return heuristics_matrix
