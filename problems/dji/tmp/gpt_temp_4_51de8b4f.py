def heuristics_v2(df):
    pv_ratio = df['close'] / df['volume']
    sma_pv_ratio = pv_ratio.rolling(window=20).mean()
    heuristics_matrix = pd.Series(sma_pv_ratio, index=df.index)
    return heuristics_matrix
