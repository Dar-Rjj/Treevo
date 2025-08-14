def heuristics_v2(df):
    close_avg_50 = df['close'].rolling(window=50).mean()
    close_avg_200 = df['close'].rolling(window=200).mean()
    vol_log_diff_10 = (df['volume'].apply(np.log)).diff(periods=10)
    heuristics_matrix = (close_avg_50 / close_avg_200) + vol_log_diff_10
    return heuristics_matrix
