def heuristics_v2(df):
    avg_vol_50 = df['volume'].rolling(window=50).mean()
    avg_vol_200 = df['volume'].rolling(window=200).mean()
    close_log_change_10_avg = (df['close'].pct_change(periods=10) + 1).apply(np.log).rolling(window=10).mean()
    heuristics_matrix = (avg_vol_50 / avg_vol_200) * (1 + close_log_change_10_avg)
    return heuristics_matrix
