def heuristics_v2(df):
    avg_vol_60 = df['volume'].rolling(window=60).mean()
    avg_vol_200 = df['volume'].rolling(window=200).mean()
    close_log_change_60 = np.log(df['close']).pct_change(periods=60)
    heuristics_matrix = (avg_vol_60 / avg_vol_200) * (1 + close_log_change_60)
    return heuristics_matrix
