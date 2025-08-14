def heuristics_v2(df):
    ma_50 = df['close'].rolling(window=50).mean()
    ma_200 = df['close'].rolling(window=200).mean()
    ma_ratio = ma_50 / ma_200
    roc_ma_ratio = ma_ratio.pct_change(periods=10)
    heuristics_matrix = pd.Series(roc_ma_ratio, index=df.index)
    return heuristics_matrix
