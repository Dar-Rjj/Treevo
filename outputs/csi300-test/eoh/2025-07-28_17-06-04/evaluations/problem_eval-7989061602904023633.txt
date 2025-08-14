def heuristics_v2(df):
    avg_vol_60 = df['volume'].rolling(window=60).mean()
    avg_vol_200 = df['volume'].rolling(window=200).mean()
    close_change_60 = df['close'].pct_change(periods=60)
    adjustment_factor = (1 + close_change_60).apply(lambda x: np.log(x) if x > 0 else 0)
    heuristics_matrix = (avg_vol_60 / avg_vol_200) * adjustment_factor
    return heuristics_matrix
