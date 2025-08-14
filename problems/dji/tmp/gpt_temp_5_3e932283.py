def heuristics_v2(df):
    sma_close_5 = df['close'].rolling(window=5).mean()
    sma_close_10 = df['close'].rolling(window=10).mean()
    avg_volume_5 = df['volume'].rolling(window=5).mean()
    vol_ratio = df['volume'] / avg_volume_5
    log_vol_ratio = vol_ratio.apply(lambda x: 1 if pd.isna(x) or x <= 0 else abs(math.log(x)))
    diff_sma = sma_close_5 - sma_close_10
    heuristics_matrix = diff_sma * log_vol_ratio
    return heuristics_matrix
