def heuristics_v2(df):
    df['sma_close_10'] = df['close'].rolling(window=10).mean()
    df['log_vol_diff'] = np.log(df['volume'].rolling(window=20).max() / df['volume'].rolling(window=20).min())
    heuristics_matrix = df['sma_close_10'] * df['log_vol_diff']
    return heuristics_matrix
