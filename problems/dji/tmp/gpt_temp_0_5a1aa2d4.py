def heuristics_v2(df):
    oc_diff = (df['open'] - df['close']) / df['volume'].rolling(window=10).mean()
    sma_oc_diff = oc_diff.rolling(window=5).mean()
    heuristics_matrix = pd.Series(sma_oc_diff, index=df.index)
    return heuristics_matrix
