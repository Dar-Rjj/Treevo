def heuristics_v2(df):
    avg_vol_30 = df['volume'].rolling(window=30).mean()
    avg_vol_150 = df['volume'].rolling(window=150).mean()
    log_vol_ratio = np.log(avg_vol_30 / avg_vol_150)
    close_return_90 = (df['close'] / df['close'].shift(90)).pow(1/90) - 1
    heuristics_matrix = log_vol_ratio + close_return_90
    return heuristics_matrix
