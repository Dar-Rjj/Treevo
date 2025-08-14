def heuristics_v2(df):
    df['20d_median_close'] = df['close'].rolling(window=20).median()
    df['abs_dist_to_20d_median'] = abs(df['close'] - df['20d_median_close'])
    total_weight = df['abs_dist_to_20d_median'].rolling(window=30).sum()
    df['weighted_volume'] = (df['volume'].apply(lambda x: max(1, x)).apply(np.log)) * df['abs_dist_to_20d_median']
    df['weighted_amount'] = (df['amount'].apply(np.sqrt)) * df['abs_dist_to_20d_median']
    df['weighted_sum'] = (df['weighted_volume'].rolling(window=30).sum() + df['weighted_amount'].rolling(window=30).sum()) / total_weight
    heuristics_matrix = df['weighted_sum'].dropna()
    return heuristics_matrix
