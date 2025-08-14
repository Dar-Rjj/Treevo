def heuristics_v2(df):
    roc_close = df['close'].pct_change(periods=10)
    std_volume = df['volume'].rolling(window=20).std()
    heuristics_matrix = pd.Series(roc_close * std_volume, index=df.index)
    return heuristics_matrix
