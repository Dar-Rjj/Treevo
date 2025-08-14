def heuristics_v2(df):
    sma_5 = df['close'].rolling(window=5).mean()
    sma_20 = df['close'].rolling(window=20).mean()
    vol_avg_10 = df['volume'].rolling(window=10).mean()
    volume_ratio = df['volume'] / vol_avg_10
    factor = (sma_5 - sma_20) * volume_ratio
    heuristics_matrix = pd.Series(factor, index=df.index)
    return heuristics_matrix
