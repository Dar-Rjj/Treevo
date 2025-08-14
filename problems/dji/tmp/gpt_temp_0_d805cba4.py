def heuristics_v2(df):
    close_change = df['close'].pct_change(periods=5)
    vol_std = df['volume'].rolling(window=10).std()
    adjusted_momentum = close_change / vol_std
    sma_adjusted_momentum = adjusted_momentum.rolling(window=20).mean()
    heuristics_matrix = pd.Series(sma_adjusted_momentum, index=df.index)
    return heuristics_matrix
