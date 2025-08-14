def heuristics_v2(df):
    window = 20
    df['pct_change_vol'] = df['volume'].pct_change().fillna(0)
    df['log_diff_close'] = (df['close'] - df['close'].rolling(window=window).median()).apply(lambda x: np.log1p(x) if x > 0 else -np.log1p(-x))
    std_pct_change_vol = df['pct_change_vol'].rolling(window=window).std()
    std_log_diff_close = df['log_diff_close'].rolling(window=window).std()
    weight_vol = 1 / std_pct_change_vol
    weight_close = 1 / std_log_diff_close
    total_weight = weight_vol + weight_close
    heuristics_matrix = (weight_vol * df['pct_change_vol'] + weight_close * df['log_diff_close']) / total_weight
    return heuristics_matrix
